

import logging


import torch

import torch.nn as nn

from torch import Tensor

from omegaconf import DictConfig


from .backbone import DiTBackbone

from ..conditioning.style import WaveClipEncoder, StyleTemporalEncoder, StyleCrossAttention

from ..conditioning.motion import MotionInjectionModule

from ..trajectory import TransitionDetector, PolicyClassifier, TSTM

from ..motion.flow_factory import get_flow_extractor

from ..motion.flow_utils import prepare_flow_inputs

from ..utils.canny_utils import extract_canny_batch

from ..objectives import (

    DenoisingLoss,

    SpatialAdaptiveMotionLoss,

    StyleAppearanceLoss,

    ContrastiveContentLoss,

    TSTMLoss,

    TotalLoss,

)


logger = logging.getLogger(__name__)


class QuadStyleModel(nn.Module):


    def __init__(self, cfg: DictConfig):

        super().__init__()

        self.cfg = cfg

        attn_cfg = getattr(cfg.model, "attention_infra", None)

        self.attention_backend = (

            getattr(attn_cfg, "backend", "auto") if attn_cfg is not None else "auto"

        )


        

        self.backbone = DiTBackbone(cfg)


        

        self.flow_extractor = get_flow_extractor(cfg)


        

        wc = cfg.model.wave_clip

        self.wave_clip = WaveClipEncoder(

            clip_model_name=wc.clip_model,

            K=wc.K,

            wavelet=wc.wavelet,

            use_ll=wc.use_ll,

            D_clip=wc.D_clip,

            D_hidden=wc.D_hidden,

            P=wc.P,

        )


        self.style_temporal_enc = StyleTemporalEncoder(

            D_clip=wc.D_clip,

            D_sty=cfg.model.style_attention.D_sty,

            P_sty=cfg.model.style_attention.P_sty,

        )


        

        td = cfg.model.transition_detector

        self.transition_detector = TransitionDetector(

            clip_model=self.wave_clip.clip,

            scales=list(td.scales),

            alpha=td.alpha,

            tau=td.tau,

            W_cls=td.W_cls,

        )


        self.policy_classifier = PolicyClassifier(

            delta_min=td.delta_min,

            W_feat=td.W_feat,

            hidden_dim=td.hidden_dim,

            scales=list(td.scales),

        )


        tstm_cfg = cfg.model.tstm

        self.tstm = TSTM(

            delta=tstm_cfg.delta,

            sigma=tstm_cfg.sigma,

            lambda_decay=tstm_cfg.lambda_decay,

            w_fade=tstm_cfg.w_fade,

        )


        

        mi = cfg.model.motion_injection

        n_heads = mi.n_motion_heads

        if n_heads == -1:

            n_heads = self.backbone.H_heads


        

        default_h, default_w = 60, 90  

        self.motion_module = MotionInjectionModule(

            n_motion_heads=n_heads,

            D_h=self.backbone.D_h,

            N_src=default_h * default_w,

        )


        

        sa = cfg.model.style_attention

        self.style_attn = StyleCrossAttention(

            D_sty=sa.D_sty,

            D_h=self.backbone.D_h,

            D_tok=self.backbone.D_tok,

            N_src=default_h * default_w,

            attention_backend=self.attention_backend,

        )


        

        self.denoising_loss = DenoisingLoss()

        self.saml_loss = SpatialAdaptiveMotionLoss()

        self.style_loss = StyleAppearanceLoss()

        self.contrastive_loss = ContrastiveContentLoss()

        self.tstm_loss = TSTMLoss()


        

        self.style_loss.set_clip(self.wave_clip.clip)

        self.contrastive_loss.set_clip(self.wave_clip.clip)


        

        from diffusers import DDPMScheduler

        self.noise_scheduler = DDPMScheduler.from_pretrained(

            "THUDM/CogVideoX-5b-I2V", subfolder="scheduler"

        )


    @staticmethod

    def _merge_flow_inputs(flow_inputs_per_sample: list[dict]) -> dict:

        if len(flow_inputs_per_sample) == 1:

            return flow_inputs_per_sample[0]


        has_displacement = all(f["has_displacement"] for f in flow_inputs_per_sample)

        return {

            "M": torch.stack([f["M"] for f in flow_inputs_per_sample], dim=0),

            "W": torch.stack([f["W"] for f in flow_inputs_per_sample], dim=0),

            "f_bar": torch.stack([f["f_bar"] for f in flow_inputs_per_sample], dim=0),

            "has_displacement": has_displacement,

        }


    def _encode_style_batch(self, style_video: Tensor, T_prime: int) -> Tensor:

        B, N = style_video.shape[:2]

        style_tokens = []

        first_boundary_logits = None

        first_policy_logits = None


        for b_idx in range(B):

            style_seq = style_video[b_idx]

            style_features = self.wave_clip.encode_frames(style_seq)

            clusters, delta_s = self.transition_detector.detect(style_seq)

            transitions, policies, is_uniform = self.policy_classifier.classify(

                clusters, delta_s, N=N

            )

            if b_idx == 0:

                first_boundary_logits = self.policy_classifier.last_boundary_logits

                first_policy_logits = self.policy_classifier.last_policy_logits


            style_smoothed = self.tstm(style_features, transitions, policies)

            cut_boundaries = []

            if not is_uniform:

                cut_boundaries = [t for t, p in zip(transitions, policies) if p == "cut"]

            encoded = self.style_temporal_enc(

                style_smoothed, T_prime, cut_boundaries=cut_boundaries

            )

            d_style = encoded.shape[1]

            tokens = encoded.reshape(T_prime, d_style, -1).permute(0, 2, 1)

            style_tokens.append(tokens)


        self.policy_classifier.last_boundary_logits = first_boundary_logits

        self.policy_classifier.last_policy_logits = first_policy_logits

        return torch.stack(style_tokens, dim=0)


    def forward_training(

        self, batch: dict, loss_weights: dict, step: int = 0,

    ) -> dict:

        V_src = batch["source_video"]

        V_sty = batch["style_video"]

        B, N, C, H, W = V_src.shape

        device = V_src.device


        

        with torch.no_grad():

            Z_src = self.backbone.encode_video(V_src)

        T_prime = Z_src.shape[2]

        P_grid_h = Z_src.shape[3]

        P_grid_w = Z_src.shape[4]

        N_src = P_grid_h * P_grid_w


        self.motion_module.N_src = N_src

        self.style_attn.N_src = N_src


        

        canny = extract_canny_batch(

            V_src,

            low=self.cfg.model.canny_encoder.canny_low,

            high=self.cfg.model.canny_encoder.canny_high,

        )

        Z_canny = self.backbone.canny_enc(canny)

        assert Z_canny.shape == Z_src.shape, (

            f"Shape mismatch: Z_canny {Z_canny.shape} vs Z_src {Z_src.shape}"

        )


        

        mi = self.cfg.model.motion_injection

        flow_inputs_per_sample = []

        for b_idx in range(B):

            flow = self.flow_extractor.compute_flow(V_src[b_idx])

            flow_inputs_per_sample.append(

                prepare_flow_inputs(

                    flow, T_prime, P_grid_h, P_grid_w, N,

                    tau_m=mi.tau_m, tau_e=mi.tau_e,

                    has_displacement=self.flow_extractor.has_displacement,

                )

            )

        flow_inputs = self._merge_flow_inputs(flow_inputs_per_sample)


        

        S_tokens = self._encode_style_batch(V_sty, T_prime)


        

        timesteps = torch.randint(

            0, self.noise_scheduler.config.num_train_timesteps,

            (B,), device=device, dtype=torch.long,

        )

        noise = torch.randn_like(Z_src)

        z_noisy = self.noise_scheduler.add_noise(Z_src, noise, timesteps)


        z_input = torch.cat([z_noisy, Z_canny], dim=1)


        pred_noise = self.backbone.forward_with_hooks(

            z_input, self.style_attn, S_tokens, flow_inputs,

            timesteps, self.motion_module,

            T_prime=T_prime, N_src_spatial=N_src,

            attention_backend=self.attention_backend,

        )


        

        L_denoise = self.denoising_loss(pred_noise, noise)


        L_saml = None

        if loss_weights.get("lambda_saml", 0) > 0:

            pred_reshaped = pred_noise.reshape(B, T_prime, -1, P_grid_h, P_grid_w)

            L_saml = self.saml_loss(

                pred_reshaped, flow_inputs, T_prime, P_grid_h, P_grid_w

            )


        L_style = None

        L_contrast = None

        L_tstm = None


        aux_freq = loss_weights.get("auxiliary_loss_freq", 4)

        if step % aux_freq == 0:

            decoded = None

            need_decode = (

                loss_weights.get("lambda_style", 0) > 0

                or loss_weights.get("lambda_contrast", 0) > 0

            )

            if need_decode:

                alpha_bar = self.noise_scheduler.alphas_cumprod.to(

                    device=device, dtype=z_noisy.dtype

                )

                alpha_t = alpha_bar[timesteps].view(B, 1, 1, 1, 1)

                sqrt_alpha = alpha_t.sqrt()

                sqrt_one_minus_alpha = (1.0 - alpha_t).sqrt()

                pred_x0 = (z_noisy - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha.clamp_min(1e-6)

                decoded = self.backbone.decode_latents(pred_x0[:, :, :1, :, :])

                decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)


            if loss_weights.get("lambda_style", 0) > 0 and decoded is not None:

                if decoded.shape[1] > 0:

                    pred_frame = decoded[:, 0]

                    sty_frame = V_sty[:, 0]

                    L_style = self.style_loss(pred_frame, sty_frame)


            if loss_weights.get("lambda_contrast", 0) > 0 and decoded is not None:

                if decoded.shape[1] > 0:

                    pred_frame = decoded[:, 0]

                    src_frame = V_src[:, 0]

                    sty_frame = V_sty[:, 0]

                    L_contrast = self.contrastive_loss(pred_frame, src_frame, sty_frame)


            if loss_weights.get("lambda_tstm", 0) > 0:

                gt_transitions = batch.get("gt_transitions", None)

                if gt_transitions is not None:

                    gt_boundary = batch.get("gt_boundary_mask", None)

                    gt_policies_tensor = batch.get("gt_policies", None)

                    L_tstm = self.tstm_loss(

                        self.policy_classifier.last_boundary_logits,

                        gt_boundary,

                        self.policy_classifier.last_policy_logits,

                        gt_policies_tensor,

                    )


        

        total_loss_fn = TotalLoss(

            lambda_saml=loss_weights.get("lambda_saml", 0),

            lambda_style=loss_weights.get("lambda_style", 0),

            lambda_contrast=loss_weights.get("lambda_contrast", 0),

            lambda_tstm=loss_weights.get("lambda_tstm", 0),

            auxiliary_loss_freq=loss_weights.get("auxiliary_loss_freq", 4),

        )


        total_loss, loss_dict = total_loss_fn(

            L_denoise, L_saml, L_style, L_contrast, L_tstm, step=step

        )


        return {

            "total_loss": total_loss,

            "loss_dict": loss_dict,

            "pred_noise": pred_noise,

        }


    @torch.no_grad()

    def generate(

        self,

        source_video: Tensor,

        style_video: Tensor,

        num_steps: int = 50,

        guidance_scale: float = 7.5,

    ) -> Tensor:

        self.eval()

        B, N, C, H, W = source_video.shape

        device = source_video.device


        

        Z_src = self.backbone.encode_video(source_video)

        T_prime = Z_src.shape[2]

        P_grid_h = Z_src.shape[3]

        P_grid_w = Z_src.shape[4]

        N_src = P_grid_h * P_grid_w


        self.motion_module.N_src = N_src

        self.style_attn.N_src = N_src


        

        canny = extract_canny_batch(

            source_video,

            low=self.cfg.model.canny_encoder.canny_low,

            high=self.cfg.model.canny_encoder.canny_high,

        )

        Z_canny = self.backbone.canny_enc(canny)


        

        mi = self.cfg.model.motion_injection

        flow_inputs_per_sample = []

        for b_idx in range(B):

            flow = self.flow_extractor.compute_flow(source_video[b_idx])

            flow_inputs_per_sample.append(

                prepare_flow_inputs(

                    flow, T_prime, P_grid_h, P_grid_w, N,

                    tau_m=mi.tau_m, tau_e=mi.tau_e,

                    has_displacement=self.flow_extractor.has_displacement,

                )

            )

        flow_inputs = self._merge_flow_inputs(flow_inputs_per_sample)


        

        S_tokens = self._encode_style_batch(style_video, T_prime)


        

        scheduler = self.noise_scheduler

        scheduler.set_timesteps(num_steps)


        z = torch.randn_like(Z_src)


        for t in scheduler.timesteps:

            t_tensor = torch.tensor([t], device=device, dtype=torch.long).expand(B)


            z_input = torch.cat([z, Z_canny], dim=1)


            pred = self.backbone.forward_with_hooks(

                z_input, self.style_attn, S_tokens, flow_inputs,

                t_tensor, self.motion_module,

                T_prime=T_prime, N_src_spatial=N_src,

                attention_backend=self.attention_backend,

            )


            z = scheduler.step(pred, t, z).prev_sample


        

        output = self.backbone.decode_latents(z)

        return ((output + 1.0) / 2.0).clamp(0.0, 1.0)


QuadStyleSystem = QuadStyleModel


__all__ = ["QuadStyleModel", "QuadStyleSystem"]

