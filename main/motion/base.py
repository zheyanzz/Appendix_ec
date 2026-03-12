

from abc import ABC, abstractmethod


import torch

from torch import Tensor


class BaseFlowExtractor(ABC):


    has_displacement: bool  


    @abstractmethod

    def compute_flow(self, frames: Tensor) -> Tensor:

        ...

