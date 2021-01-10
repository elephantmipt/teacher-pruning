from typing import Union, Tuple, List

from torch import Tensor
from torch import nn


class KDModel(nn.Module):
    def forward(
        self,
        x: Tensor,
        output_hiddens: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Forward pass.
        Args:
            x: input.
            output_hiddens: hidden states from model.

        Returns:
            logits or tuple of logits and list of hiddens.
        """
        pass
