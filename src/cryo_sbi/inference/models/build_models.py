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

    estimator = estimator_models.NLEWithEmbedding(
        embedding_net=embedding,
        output_embedding_dim=config["OUT_DIM"],
        num_transforms=config["NUM_TRANSFORM"],
        num_hidden_flow=config["NUM_HIDDEN_FLOW"],
        hidden_flow_dim=config["HIDDEN_DIM_FLOW"],
        flow=model,
        bins=config.get("NUM_BINS", 8),
        slope=config.get("SLOPE", 0.001),
        num_models=config["NUM_MODELS"],
        **{"activation": partial(nn.LeakyReLU, 0.1)},
    )

    return estimator


def build_nre_classifier_model(config: dict, **embedding_kwargs) -> nn.Module:
    raise NotImplementedError("NRE classifier model has not been implemented yet!")
