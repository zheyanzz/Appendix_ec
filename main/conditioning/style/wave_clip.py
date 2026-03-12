

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


try:
    import pywt
except ImportError:  
    pywt = None

logger = logging.getLogger(__name__)


class WaveClipEncoder(nn.Module):


    def __init__(
        self,
        clip_model_name: str = "ViT-B-14",
        pretrained: str = "openai",
        K: int = 3,
        wavelet: str = "db1",
        use_ll: bool = True,
        D_clip: int = 540,
        D_hidden: int = 540,
        P: int = 14,

    ):

        super().__init__()
        self.K = K
        self.wavelet = wavelet
        self.use_ll = use_ll
        self.D_clip = D_clip
        self.P = P
        import open_clip
        self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        n_bands = 3 * K + (1 if use_ll else 0)
        n_channels_in = n_bands * 3  
        self.W1 = nn.Conv2d(n_channels_in, D_hidden, kernel_size=1)

        self.W2 = nn.Conv2d(D_hidden, D_clip, kernel_size=1)

        self.gate_conv = nn.Conv2d(2 * D_clip, D_clip, kernel_size=3, padding=1)

    def _extract_clip_features(self, frame: Tensor) -> Tensor:
        x = F.interpolate(frame.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():      

            tokens = self.clip.encode_image(x)
            visual = self.clip.visual
            x_vis = visual(x)

        return self._clip_patch_tokens(frame)


    @torch.no_grad()

    def _clip_patch_tokens(self, frame: Tensor) -> Tensor:

        x = F.interpolate(

            frame.unsqueeze(0), size=(224, 224),

            mode="bilinear", align_corners=False,

        )


        visual = self.clip.visual


        

        x = visual.conv1(x)  

        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  


        

        cls_token = visual.class_embedding.unsqueeze(0).unsqueeze(0)  

        cls_token = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat([cls_token + visual.positional_embedding[:1], x + visual.positional_embedding[1:]], dim=1)


        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  

        x = visual.transformer(x)

        x = x.permute(1, 0, 2)  


        

        patch_tokens = x[0, 1:]  

        T_CLIP = patch_tokens.T.reshape(self.D_clip, self.P, self.P)

        return T_CLIP


    def _extract_swt_features(self, frame: Tensor) -> Tensor:

        if pywt is None:

            return self._extract_swt_features_fallback(frame)


        device = frame.device

        C, H, W = frame.shape


        

        pad_h = (2 ** self.K - H % (2 ** self.K)) % (2 ** self.K)

        pad_w = (2 ** self.K - W % (2 ** self.K)) % (2 ** self.K)

        frame_np = frame.cpu().numpy()


        if pad_h > 0 or pad_w > 0:

            frame_np = np.pad(

                frame_np, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"

            )


        bands = []

        ll_band = None


        for k in range(1, self.K + 1):

            detail_bands_k = {"HL": [], "LH": [], "HH": []}

            for ch in range(C):

                coeffs = pywt.swt2(frame_np[ch], wavelet=self.wavelet, level=self.K)

                

                

                idx = self.K - k

                cA, (cH, cV, cD) = coeffs[idx]

                detail_bands_k["HL"].append(cH[:H, :W])

                detail_bands_k["LH"].append(cV[:H, :W])

                detail_bands_k["HH"].append(cD[:H, :W])


                if k == self.K and ch == C - 1 and self.use_ll:

                    

                    ll_all = []

                    for c2 in range(C):

                        coeffs2 = pywt.swt2(frame_np[c2], wavelet=self.wavelet, level=self.K)

                        cA2, _ = coeffs2[0]  

                        ll_all.append(cA2[:H, :W])

                    ll_band = np.stack(ll_all, axis=0)  


            for band_name in ["HL", "LH", "HH"]:

                b = np.stack(detail_bands_k[band_name], axis=0)  

                bands.append(b)


        

        pooled_bands = []

        for b in bands:

            b_t = torch.from_numpy(b).float().to(device)  

            

            for c in range(C):

                s_c = torch.quantile(b_t[c].abs(), 0.99)

                b_t[c] = b_t[c] / (s_c + 1e-4)

            b_pooled = F.adaptive_avg_pool2d(b_t.unsqueeze(0), (self.P, self.P)).squeeze(0)

            pooled_bands.append(b_pooled)


        

        if self.use_ll and ll_band is not None:

            ll_t = torch.from_numpy(ll_band).float().to(device)  

            for c in range(C):

                mu_c = ll_t[c].mean()

                s_c = torch.quantile((ll_t[c] - mu_c).abs(), 0.99)

                ll_t[c] = (ll_t[c] - mu_c) / (s_c + 1e-4)

            ll_pooled = F.adaptive_avg_pool2d(ll_t.unsqueeze(0), (self.P, self.P)).squeeze(0)

            pooled_bands.append(ll_pooled)


        E_pix = torch.cat(pooled_bands, dim=0)  

        return E_pix


    def _extract_swt_features_fallback(self, frame: Tensor) -> Tensor:

        C, H, W = frame.shape

        pooled_bands = []


        for k in range(1, self.K + 1):

            stride = 2 ** (k - 1)

            smooth = F.avg_pool2d(

                frame.unsqueeze(0),

                kernel_size=3,

                stride=1,

                padding=1,

            ).squeeze(0)


            gx = frame - torch.roll(frame, shifts=stride, dims=2)

            gy = frame - torch.roll(frame, shifts=stride, dims=1)

            gxy = gx * gy


            for band in (gx, gy, gxy):

                band_pooled = F.adaptive_avg_pool2d(band.unsqueeze(0), (self.P, self.P)).squeeze(0)

                pooled_bands.append(band_pooled)


            if self.use_ll and k == self.K:

                ll_pooled = F.adaptive_avg_pool2d(smooth.unsqueeze(0), (self.P, self.P)).squeeze(0)

                pooled_bands.append(ll_pooled)


        return torch.cat(pooled_bands, dim=0)


    def encode_frames(self, style_frames: Tensor) -> Tensor:

        N = style_frames.shape[0]

        device = style_frames.device

        outputs = []


        for t in range(N):

            frame = style_frames[t]  


            

            T_CLIP_t = self._clip_patch_tokens(frame)  


            

            E_pix = self._extract_swt_features(frame)  


            

            M_mod = torch.tanh(

                self.W2(F.relu(self.W1(E_pix.unsqueeze(0))))

            ).squeeze(0)  


            

            gate_input = torch.cat([T_CLIP_t, M_mod], dim=0).unsqueeze(0)  

            gamma = torch.sigmoid(self.gate_conv(gate_input)).squeeze(0)  


            T_out_t = T_CLIP_t + gamma * M_mod  

            outputs.append(T_out_t)


        return torch.stack(outputs, dim=0)  

