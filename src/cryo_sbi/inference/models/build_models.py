import torch
import torch.nn as nn
from functools import partial
import zuko
import lampe
import cryo_sbi.inference.models.estimator_models as estimator_models
from cryo_sbi.inference.models.embedding_nets import EMBEDDING_NETS


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

def build_nle_flow_model(config: dict, pretrained_theta_path: str = None, **embedding_kwargs) -> nn.Module:
    """
    Function to build NLE estimator with embedding net
    from config_file

    Args:
        config (dict): config file
        pretrained_theta_path (str, optional): path to pre-trained theta embedding weights
        embedding_kwargs (dict): kwargs for embedding net

    Returns:
        estimator (nn.Module): NLE estimator
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
        # Prepare theta embedding kwargs from config
        theta_kwargs = {}
        
        # Extract GNN-specific parameters from config if they exist
        if config.get("EMBEDDING_THETA") == "GNN":
            if "GNN_CUTOFF" in config:
                theta_kwargs["cutoff"] = config["GNN_CUTOFF"]
                print(f"Using GNN cutoff from config: {config['GNN_CUTOFF']}")
            if "GNN_HIDDEN_DIM" in config:
                theta_kwargs["hidden_dim"] = config["GNN_HIDDEN_DIM"]
                print(f"Using GNN hidden_dim from config: {config['GNN_HIDDEN_DIM']}")
        
        # Override with any manually provided embedding_kwargs (takes precedence)
        theta_kwargs.update(embedding_kwargs)
        
        # Build embeddings
        # For image embedding (e.g., ResNet18): only needs output_dimension
        embedding_x = partial(
            EMBEDDING_NETS[config["EMBEDDING_X"]], 
            config["OUT_DIM_X"]
        )
        
        # For theta embedding (e.g., GNN or MLP): needs output_dimension + specific kwargs
        embedding_theta = partial(
            EMBEDDING_NETS[config["EMBEDDING_THETA"]], 
            config["OUT_DIM_THETA"],  # This becomes the first positional arg (output_dimension)
            **theta_kwargs             # These become keyword args (cutoff=..., hidden_dim=...)
        )

    except KeyError:
        raise NotImplementedError(
            f"Model : {config['EMBEDDING']} has not been implemented yet! \
The following embeddings are implemented : {[key for key in EMBEDDING_NETS.keys()]}"
        )

    estimator = estimator_models.NLEWithEmbedding(
        embedding_net_x=embedding_x,
        embedding_net_theta=embedding_theta,
        output_embedding_dim_x=config["OUT_DIM_X"],
        output_embedding_dim_theta=config["OUT_DIM_THETA"],
        num_transforms=config["NUM_TRANSFORM"],
        num_hidden_flow=config["NUM_HIDDEN_FLOW"],
        hidden_flow_dim=config["HIDDEN_DIM_FLOW"],
        flow=model,
        **{"activation": partial(nn.LeakyReLU, 0.1)},
    )

    # ==================================================
    # Load pre-trained theta embedding weights if provided
    if pretrained_theta_path is not None:
        print(f"Loading pre-trained theta embedding weights from {pretrained_theta_path}")
        checkpoint = torch.load(pretrained_theta_path)
        
        # Handle both checkpoint dict and raw state_dict
        if isinstance(checkpoint, dict) and 'mlp_state_dict' in checkpoint:
            state_dict = checkpoint['mlp_state_dict']
            print(f"  Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')}, "
                  f"val_acc={checkpoint.get('val_acc', 'N/A'):.2f}%)" if 'val_acc' in checkpoint else "  Loaded from checkpoint")
        else:
            state_dict = checkpoint
        
        estimator.embedding_theta.load_state_dict(state_dict)
        print("✓ Pre-trained theta embedding weights loaded successfully")
    # ==================================================

    return estimator


def build_nre_classifier_model(config: dict, **embedding_kwargs) -> nn.Module:
    raise NotImplementedError("NRE classifier model has not been implemented yet!")
