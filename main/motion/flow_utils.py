

import torch

import torch.nn.functional as F

from torch import Tensor


def prepare_flow_inputs(

    flow: Tensor,

    T_prime: int,

    P_grid_h: int,

    P_grid_w: int,

    N: int,

    tau_m: float = 0.1,

    tau_e: float = 0.1,

    has_displacement: bool = True,

) -> dict:

    device = flow.device

    _, _, H, W = flow.shape


    

    pi = torch.linspace(0, N - 1, T_prime).long()  

    f_aligned = flow[pi]  


    

    ph = H // P_grid_h

    pw = W // P_grid_w

    f_bar = F.avg_pool2d(

        f_aligned.reshape(T_prime * 2, 1, H, W),

        kernel_size=(ph, pw),

        stride=(ph, pw),

    ).reshape(T_prime, 2, P_grid_h, P_grid_w)

    


    

    patches = f_aligned.unfold(2, ph, ph).unfold(3, pw, pw)

    

    patch_mean = patches.mean(dim=(-2, -1), keepdim=True)  

    sigma2 = (patches - patch_mean).pow(2).sum(dim=1).mean(dim=(-2, -1))  

    sigma = sigma2.sqrt().reshape(T_prime, -1)  


    

    u = f_bar[:, 0].reshape(T_prime, -1)  

    v = f_bar[:, 1].reshape(T_prime, -1)  

    M = torch.stack([u, v, sigma], dim=-1)  


    

    mag = torch.norm(f_bar, dim=1)  


    

    sobel_x = torch.tensor(

        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device

    ).reshape(1, 1, 3, 3) / 8.0

    sobel_y = sobel_x.transpose(-2, -1)


    mag_4d = mag.unsqueeze(1)  

    gx = F.conv2d(mag_4d, sobel_x, padding=1)

    gy = F.conv2d(mag_4d, sobel_y, padding=1)

    boundary = (gx ** 2 + gy ** 2).sqrt().squeeze(1)  


    

    p70_mag = torch.quantile(

        mag.reshape(T_prime, -1), 0.70, dim=1

    ).reshape(T_prime, 1, 1)

    p70_edge = torch.quantile(

        boundary.reshape(T_prime, -1), 0.70, dim=1

    ).reshape(T_prime, 1, 1)


    w_hi = torch.sigmoid((mag - p70_mag) / tau_m)  

    w_edge = torch.sigmoid((boundary - p70_edge) / tau_e)  

    w_lo = 1.0 - torch.max(w_hi, w_edge)  


    W = torch.stack([w_hi, w_edge, w_lo], dim=-1)  

    W = W.reshape(T_prime, -1, 3)  


    return {

        "M": M,          

        "W": W,          

        "f_bar": f_bar,  

        "has_displacement": has_displacement,

    }

