# Anisotropic Kuwahara filter
# adapted from https://github.com/GarrettGunnell/Post-Processing/blob/main/Assets/Kuwahara%20Filter

from torch import load, no_grad, autocast
from torch import split, squeeze, stack
import torch
from numpy import sqrt, exp, sin, cos
from PIL import Image

from kornia.filters import SpatialGradient

from scripts.johnim.images import *
from scripts.johnim.state import State

PI = 3.14159265358979323

with no_grad():
    with autocast('cuda'):
        def gaussian(sigma, pos):
            return (1.0 / sqrt(2.0 * PI * sigma * sigma)) * exp(-(pos * pos) / (2.0 * sigma * sigma))

        s = State
        s.seed = 'random'

        t_img = load_torch_image('./outputs/lhs.png')
        #t_img = load_torch_image('output/68798888721_038.png')

        print(f't_img shape {t_img.shape}')

        img = ImageResultB(t_img)

        sobel = SpatialGradient()

        t_img_sbl = sobel(t_img[None])[0]

        horiz, vert = split(t_img_sbl, 1, 1)
        horiz = squeeze(horiz)
        vert = squeeze(vert)

        save_torch_b(horiz, 'HORIZ')
        save_torch_b(vert, 'VERT')
        
        # "Structure tensor"
        # S = [[E,F],[F,G]]
        hh = horiz * horiz
        vv = vert * vert
        hv = horiz * vert

        E = hh.sum(dim=0)
        G = vv.sum(dim=0)
        F = hv.sum(dim=0)

        sbl_final = stack((E, G, F))

        save_torch_b(sbl_final, 'SOBEL_FINAL')

        # do 2 gaussian blur passes; skipping for now

        EG = E + G
        EmG = E - G
        F2 = F * F
        srEG4F2 = torch.sqrt(EmG*EmG + 4*F2)

        lambda1 = 0.5 * (EG + srEG4F2)
        lambda2 = 0.5 * (EG - srEG4F2)

        save_torch_b(stack((lambda1, lambda1, lambda1)), 'LAMBDA1')
        save_torch_b(stack((lambda2, lambda2, lambda2)), 'LAMBDA2')

        t_hat_0 = lambda1 - E
        t_hat_1 = -F

        A = (lambda1 - lambda2) / (lambda1 + lambda2)

        A = torch.nan_to_num(A, 0, 0, 0)

        v = stack((lambda1 - E, -F))

        vnorm = torch.norm(v, dim=0, p=2)

        t = v / vnorm

        tsum = torch.sum(t, dim=0)

        tmask = tsum * torch.zeros_like(tsum) # Everything but NaN is now 0
        tmask = torch.nan_to_num(tmask, 1, 1, 1) # NaNs are now 1

        # Replace all NaNs with (0, 1)
        t0, t1 = split(t, 1, dim=0)
        t0 = squeeze(t0)
        t1 = squeeze(t1)
        print(t.shape)
        print(t0.shape)
        print(tmask.shape)
        t0 = (t0 * (1 - tmask)) + (torch.zeros_like(tmask) * tmask)
        t1 = (t1 * (1 - tmask)) + (torch.ones_like(tmask) * tmask)

        phi = -torch.atan2(t1, t0)
        phi = torch.nan_to_num(phi, 0, 0, 0) # 0 is a guess, idk if it's an appropriate value here.
        print(f'phi shape: {phi.shape}')

        t = stack((t0, t1))

        # Settings
        alpha = 1.0
        kernelSize = 2
        sharpness = 8
        hardness = 8
        zeroCrossing = 0.58
        useZeta = False
        zeta = 1.0 if useZeta else 2.0 / 2.0 / (kernelSize / 2.0)
        passes = 1
        # end Settings

        kernelRadius = kernelSize / 2

        a = kernelRadius * torch.clamp((alpha + A) / alpha, 0.1, 2.0)
        b = kernelRadius * torch.clamp(alpha / (alpha + A), 0.1, 2.0)

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        R = stack((stack((cos_phi, -sin_phi)),stack((sin_phi, cos_phi))))
        print(f'R shape: {R.shape}')

        S = stack((stack((0.5 / a, torch.zeros_like(a))), stack((torch.zeros_like(b), 0.5 / b))))
        print(f'S shape: {S.shape}')

        SR = torch.matmul(R, S)
        print(f'SR shape: {SR.shape}')

        aa = a * a
        bb = b * b
        cos_phi_2 = cos_phi * cos_phi
        sin_phi_2 = sin_phi * sin_phi

        max_x = torch.sqrt(aa * cos_phi_2 + bb * sin_phi_2).int()
        max_y = torch.sqrt(aa * sin_phi_2 + bb * cos_phi_2).int()

        sinZeroCross = sin(zeroCrossing)
        eta = (zeta + cos(zeroCrossing)) / (sinZeroCross * sinZeroCross)

        h,w = phi.shape

        m = torch.zeros((4,8,h,w))
        s = torch.zeros((3,8,h,w))

        print(f'max_x shape: {max_x.shape}')
        print(f'max_y shape: {max_y.shape}')

        #print(phi)

        t_channel_split = stack([t_img[n][None] for n in range(3)])

        print(f't_c_s shape: {t_channel_split.shape}')

        SR_t = torch.transpose(torch.transpose(SR, 0, 2), 1, 3)

        print(f'SR_t shape: {SR_t.shape}')

        max_max_x = int(torch.max(max_x).item())
        max_max_y = int(torch.max(max_y).item())


        # TODO
        # Do the SR_t @ v matmuls individually and *then* construct the kernel from them

        kern = torch.tensor([[[x,y] for x in range(-max_max_x, max_max_x + 1)] for y in range(-max_max_y, max_max_y + 1)])

        print(f'kern shape: {kern.shape}')

        #print(kern)

        kern = torch.transpose(kern, 0, 2)

        #print(kern)

        hnng = SR @ kern

        print(f'hnng shape: {hnng.shape}')

        print(hnng)