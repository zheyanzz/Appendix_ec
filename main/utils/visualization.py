

import torch

from torch import Tensor


def make_comparison(

    source: Tensor, stylized: Tensor, style_ref: Tensor | None = None

) -> Tensor:

    parts = [source, stylized]

    if style_ref is not None:

        parts.append(style_ref)

    return torch.cat(parts, dim=-1)

