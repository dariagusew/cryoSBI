import torch
import torch.nn as nn
import zuko
from lampe.inference import NPE, NRE
import math


class NPEWithEmbedding(nn.Module):
    """Neural Posterior Estimation with embedding net

    Attributes:
        npe (NPE): NPE model
        embedding (nn.Module): embedding net
        standardize (Standardize): standardization module
    """

    def __init__(
        self,
        embedding_net: nn.Module,
        output_embedding_dim: int,
        num_transforms: int = 4,
        num_hidden_flow: int = 2,
        hidden_flow_dim: int = 128,
        flow: nn.Module = zuko.flows.MAF,
        theta_shift: float = 0.0,
        theta_scale: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Neural Posterior Estimation with embedding net.

        Args:
            embedding_net (nn.Module): embedding net
            output_embedding_dim (int): output embedding dimension
            num_transforms (int, optional): number of transforms. Defaults to 4.
            num_hidden_flow (int, optional): number of hidden layers in flow. Defaults to 2.
            hidden_flow_dim (int, optional): hidden dimension in flow. Defaults to 128.
            flow (nn.Module, optional): flow. Defaults to zuko.flows.MAF.
            theta_shift (float, optional): Shift of the theta for standardization. Defaults to 0.0.
            theta_scale (float, optional): Scale of the theta for standardization. Defaults to 1.0.
            kwargs: additional arguments for the flow

        Returns:
            None
        """

        super().__init__()

        self.npe = NPE(
            1,
            output_embedding_dim,
            transforms=num_transforms,
            build=flow,
            hidden_features=[*[hidden_flow_dim] * num_hidden_flow, 128, 64],
            **kwargs,
        )

        self.embedding = embedding_net()
        self.standardize = Standardize(theta_shift, theta_scale)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NPE model

        Args:
            theta (torch.Tensor): Conformational parameters.
            x (torch.Tensor): Image to condition the posterior on.

        Returns:
            torch.Tensor: Log probability of the posterior.
        """

        return self.npe(self.standardize(theta), self.embedding(x))

    def flow(self, x: torch.Tensor):
        """
        Conditions the posterior on an image.

        Args:
            x (torch.Tensor): Image to condition the posterior on.

        Returns:
            zuko.flows.Flow: The posterior distribution.
        """
        return self.npe.flow(self.embedding(x))

    def sample(self, x: torch.Tensor, shape=(1,)) -> torch.Tensor:
        """
        Generate samples from the posterior distribution.

        Args:
            x (torch.Tensor): Image to condition the posterior on.
            shape (tuple, optional): Shape of the samples. Defaults to (1,).

        Returns:
            torch.Tensor: Samples from the posterior distribution.
        """

        samples_standardized = self.flow(x).sample(shape)
        return self.standardize.transform(samples_standardized)


class NLEWithEmbedding(nn.Module):
    """Neural Likelihood Estimation with a learned embedding for the conditioning variable.

    This class is designed for problems where the conditioning variable `theta`
    represents a discrete set of categories (e.g., different physical models),
    which are provided as integer labels.
    """

    def __init__(
        self,
        embedding_net: nn.Module,
        output_embedding_dim: int,
        num_models: int,
        num_transforms: int = 4,
        num_hidden_flow: int = 2,
        hidden_flow_dim: int = 128,
        flow: nn.Module = zuko.flows.MAF,
        **kwargs,
    ) -> None:
        """
        Args:
            embedding_net (nn.Module):      Embedding net for the image `x`.
            output_embedding_dim (int):     Output dimension of the image embedding.
            num_models (int):               Number of distinct models/categories for `theta`.
            num_transforms (int, optional): Number of transforms. Defaults to 4.
            num_hidden_flow (int, optional): Number of hidden layers in flow. Defaults to 2.
            hidden_flow_dim (int, optional): Hidden dimension in flow. Defaults to 128.
            flow (nn.Module, optional):      Flow architecture. Defaults to zuko.flows.MAF.
            kwargs:                          Additional arguments for the flow.
        """
        
        super().__init__()

        # 0. Euristic for theta_embedding_dim
        theta_embedding_dim = max(1, min(50, num_models // 2))

        # 1. Create the learnable embedding layer for theta
        self.theta_embedding = nn.Embedding(num_models, theta_embedding_dim)

        # 2. The density estimator now takes a context of size `theta_embedding_dim`
        self.nle = NPE(
            output_embedding_dim,
            theta_embedding_dim,
            transforms=num_transforms,
            build=flow,
            hidden_features=[*[hidden_flow_dim] * num_hidden_flow, 128, 64],
            **kwargs,
        )

        # 3. Set image embedding
        self.embedding = embedding_net()
        

    def _embed_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Helper function to process and embed theta."""
        # nn.Embedding expects LongTensor of indices.
        # Assuming theta comes in as (batch_size, 1) or (batch_size,) with float/int type.
        if theta.dim() == 2 and theta.shape[1] == 1:
            theta = theta.squeeze(-1)
        
        return self.theta_embedding(theta.round().long())

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NLE model.

        Args:
            x (torch.Tensor): Image whose likelihood we model.
            theta (torch.Tensor): Integer labels for the models (conditioning variable).

        Returns:
            torch.Tensor: Log-likelihood
        """
        # Embed both the image and the conditioning labels
        x_embedded = self.embedding(x)
        theta_embedded = self._embed_theta(theta)

        return self.nle(x_embedded, theta_embedded)

    def flow(self, theta: torch.Tensor):
        """
        Conditions the likelihood on the model labels.

        Args:
            theta (torch.Tensor): Integer labels for the models.

        Returns:
            zuko.flows.Flow: The likelihood distribution p(x | theta).
        """
        # Embed the conditioning labels
        theta_embedded = self._embed_theta(theta)
        
        return self.nle.flow(theta_embedded)

