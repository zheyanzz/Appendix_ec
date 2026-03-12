from __future__ import annotations




import torch

from torch import Tensor


class TotalLoss:


    def __init__(

        self,

        lambda_saml: float = 0.1,

        lambda_style: float = 0.01,

        lambda_contrast: float = 0.1,

        lambda_tstm: float = 1.0,

        auxiliary_loss_freq: int = 4,

    ):

        self.lambda_saml = lambda_saml

        self.lambda_style = lambda_style

        self.lambda_contrast = lambda_contrast

        self.lambda_tstm = lambda_tstm

        self.auxiliary_loss_freq = auxiliary_loss_freq


    def __call__(

        self,

        L_denoise: Tensor,

        L_saml: Tensor | None = None,

        L_style: Tensor | None = None,

        L_contrast: Tensor | None = None,

        L_tstm: Tensor | None = None,

        step: int = 0,

    ) -> tuple[Tensor, dict]:

        total = L_denoise.clone()

        loss_dict = {"L_denoise": L_denoise.item()}


        

        if L_saml is not None and self.lambda_saml > 0:

            total = total + self.lambda_saml * L_saml

            loss_dict["L_saml"] = L_saml.item()


        

        compute_aux = (step % self.auxiliary_loss_freq == 0)


        if compute_aux and L_style is not None and self.lambda_style > 0:

            total = total + self.lambda_style * L_style

            loss_dict["L_style"] = L_style.item()


        if compute_aux and L_contrast is not None and self.lambda_contrast > 0:

            total = total + self.lambda_contrast * L_contrast

            loss_dict["L_contrast"] = L_contrast.item()


        if compute_aux and L_tstm is not None and self.lambda_tstm > 0:

            total = total + self.lambda_tstm * L_tstm

            loss_dict["L_tstm"] = L_tstm.item()


        loss_dict["L_total"] = total.item()

        return total, loss_dict

