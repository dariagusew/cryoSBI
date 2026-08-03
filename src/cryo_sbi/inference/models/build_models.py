import torch
import torch.nn as nn
from functools import partial
import zuko
import lampe
import cryo_sbi.inference.models.estimator_models as estimator_models
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS


class NSFFix:
    """
    Wrapper around zuko.flows.NSF that forces spline bin widths to be
    uniformly spaced on a regular grid across [-B, B].
    """
    def __new__(cls, features: int, context: int = 0, bins: int = 8, verbose: bool = True, **kwargs):
        if verbose:
            print(f"\n[NSFFix Debug] Initializing NSF-FIX:")
            print(f"  ├─ features: {features}")
            print(f"  ├─ context:  {context}")
            print(f"  ├─ bins:     {bins}")
            print(f"  └─ kwargs:   {kwargs}")

        # 1. Instantiate standard zuko NSF flow
        flow = zuko.flows.NSF(features=features, context=context, bins=bins, **kwargs)

        # State container for one-time runtime debug printing
        debug_state = {"triggered": False}

        params_per_feature = 3 * bins - 1

        # 2. Function to modify hypernetwork output tensor across ALL features
        def patch_hyper_forward(original_forward, hyper_module):
            def wrapped_forward(*args, **kwargs_inner):
                output = original_forward(*args, **kwargs_inner)

                # Reshape from (..., features * params_per_feature) -> (..., features, params_per_feature)
                orig_shape = output.shape
                output_3d = output.view(*orig_shape[:-1], features, params_per_feature)

                if verbose and not debug_state["triggered"]:
                    print(f"\n[NSFFix Debug] 🚀 Intercepting Hypernetwork Forward:")
                    print(f"  ├─ Original Flat Shape: {orig_shape}")
                    print(f"  ├─ Reshaped 3D Shape:   {output_3d.shape}")
                    print(f"  ├─ Width Logits Before (All {features} features): min={output_3d[..., :bins].min().item():.4f}, max={output_3d[..., :bins].max().item():.4f}")

                # ZERO OUT WIDTH LOGITS FOR ALL FEATURES SIMULTANEOUSLY
                output_3d[..., :bins] = 0.0

                if verbose and not debug_state["triggered"]:
                    print(f"  └─ Width Logits After  (All {features} features): min={output_3d[..., :bins].min().item():.4f}, max={output_3d[..., :bins].max().item():.4f} (Uniform Grid Enforced Across ALL Features ✅)\n")
                    debug_state["triggered"] = True

                # Return in original flat shape expected by zuko
                return output_3d.view(orig_shape)

            return wrapped_forward

        # 3. Intercept `forward` on all hypernetworks
        hook_count = 0
        hooked_names = []

        for name, module in flow.named_modules():
            if hasattr(module, "hyper") and isinstance(module.hyper, nn.Module):
                module.hyper.forward = patch_hyper_forward(module.hyper.forward, module.hyper)
                hooked_names.append(f"{name}.hyper ({module.hyper.__class__.__name__})")
                hook_count += 1

        if verbose:
            print(f"[NSFFix Debug] Successfully patched {hook_count} hypernetwork forward method(s):")
            for h_name in hooked_names:
                print(f"  └─ {h_name}")

        return flow


def build_npe_flow_model(config: dict, **embedding_kwargs) -> nn.Module:
    """
    Function to build NPE estimator with embedding net
    from config_file

    Args:
        config (dict): config file
        embedding_kwargs (dict): kwargs for embedding net

    Returns:
        estimator (nn.Module): NPE estimator
    """

    if config["MODEL"] == "MAF":
        model = zuko.flows.MAF
    elif config["MODEL"] == "NSF":
        model = zuko.flows.NSF
    elif config["MODEL"] == "SOSPF":
        model = zuko.flows.SOSPF
    else:
        raise NotImplementedError(
            f"Model : {config['MODEL']} has not been implemented yet!"
        )

    try:
        embedding = partial(
            EMBEDDING_NETS[config["EMBEDDING"]], config["OUT_DIM"], **embedding_kwargs
        )
    except KeyError:
        raise NotImplementedError(
            f"Model : {config['EMBEDDING']} has not been implemented yet! \
The following embeddings are implemented : {[key for key in EMBEDDING_NETS.keys()]}"
        )

    estimator = estimator_models.NPEWithEmbedding(
        embedding_net=embedding,
        output_embedding_dim=config["OUT_DIM"],
        num_transforms=config["NUM_TRANSFORM"],
        num_hidden_flow=config["NUM_HIDDEN_FLOW"],
        hidden_flow_dim=config["HIDDEN_DIM_FLOW"],
        flow=model,
        theta_shift=config["THETA_SHIFT"],
        theta_scale=config["THETA_SCALE"],
        **{"activation": partial(nn.LeakyReLU, 0.1)},
    )

    return estimator

def build_nle_flow_model(config: dict, image_size: int, **embedding_kwargs) -> nn.Module:
    """
    Function to build NLE estimator with embedding net
    from config_file

    Args:
        config (dict): config file
        image_size (int): number of pixels
        embedding_kwargs (dict): kwargs for embedding net

    Returns:
        estimator (nn.Module): NLE estimator
    """

    if config["MODEL"] == "MAF":
        model = zuko.flows.MAF
    elif config["MODEL"] == "NSF":
        model = zuko.flows.NSF
    elif config["MODEL"] == "NSF-FIX":
        model = NSFFix
    elif config["MODEL"] == "SOSPF":
        model = zuko.flows.SOSPF
    else:
        raise NotImplementedError(
            f"Model : {config['MODEL']} has not been implemented yet!"
        )

    try:
        embedding = partial(
            EMBEDDING_NETS[config["EMBEDDING"]], config["OUT_DIM"], image_size, **embedding_kwargs
        )
    except KeyError:
        raise NotImplementedError(
            f"Model : {config['EMBEDDING']} has not been implemented yet! \
The following embeddings are implemented : {[key for key in EMBEDDING_NETS.keys()]}"
        )

    # 1. Common kwargs shared across NSF, NSF-FIX, and SOSPF
    estimator_kwargs = {
        "embedding_net": embedding,
        "output_embedding_dim": config["OUT_DIM"],
        "num_transforms": config["NUM_TRANSFORM"],
        "num_hidden_flow": config["NUM_HIDDEN_FLOW"],
        "hidden_flow_dim": config["HIDDEN_DIM_FLOW"],
        "flow": model,
        "num_models": config["NUM_MODELS"],
        "activation": nn.SiLU,
        "randperm": True,
    }

    # 2. Add model-specific parameters
    model_type = config["MODEL"].upper()

    if model_type in ["NSF", "NSF-FIX"]:
        estimator_kwargs["bins"] = config.get("NUM_BINS", 8)
    elif model_type == "SOSPF":
        # SOSPF does not take 'bins'
        pass
    else:
        raise ValueError(f"Unsupported flow model: {model_type}")

    # 3. Instantiate the estimator
    estimator = estimator_models.NLEWithEmbedding(**estimator_kwargs)

    return estimator


def build_nre_classifier_model(config: dict, **embedding_kwargs) -> nn.Module:
    raise NotImplementedError("NRE classifier model has not been implemented yet!")
