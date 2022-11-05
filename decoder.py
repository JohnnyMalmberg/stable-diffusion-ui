from ldm.models.autoencoder import AutoencoderKL as Coder
from torch.nn import Identity
from scripts.johnim.images import *
from torch import load, split, cuda, no_grad, clamp, autocast, cat, zeros, zeros_like, ones_like
from torchvision.transforms import GaussianBlur as Blur
from einops import rearrange
import numpy as np
from PIL import Image

import gc, time

with no_grad():
    with autocast('cuda'):

        def mem():
            gc.collect()
            cuda.empty_cache()

        ddconf = {
            'double_z': True,
            'z_channels': 4,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': (1,2,4,4),
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0
        }

        lossconf = {
            'target': 'torch.nn.Identity'
        }

        ckpt = '~/ml-quickload/vae-ft-mse-840000-ema-pruned.ckpt'

        mem()

        coder = Coder(ddconfig=ddconf, lossconfig=lossconf, embed_dim=4, ckpt_path=ckpt, monitor='val/rec_loss').to('cuda')

        impath = '/home/johnim/data-0/media/images/ai_art/output/testo2_cityb.pt'

        # this magic number 0.18215 is the "scale factor" of stable diffusion,
        # it was found in the *.yaml model config
        latent = 1. / 0.18215 * load(impath)

        quarters = split(latent, int(latent.shape[3]/4), 3)

        half0 = cat(quarters[:2], dim=3)
        half1 = cat(quarters[2:], dim=3)
        mid = cat(quarters[1:3], dim=3)

        def tpi(torch_image):
            clamped_image = clamp((torch_image + 1.0) / 2.0, min=0.0, max=1.0)
            np_image = 255. * rearrange(clamped_image.cpu().numpy(), 'c h w -> h w c')
            pil_image = Image.fromarray(np_image.astype(np.uint8))
            return pil_image

        mem()

        res = coder.decode(half0.to('cuda'))
        r1 = res.clone().cpu()
        pilim = tpi(res[0])
        pilim.save('/home/johnim/data-0/media/images/ai_art/output/DECODE_half0.png')
        del res, pilim
        mem()

        res = coder.decode(half1.to('cuda'))
        r2 = res.clone().cpu()
        pilim = tpi(res[0])
        pilim.save('/home/johnim/data-0/media/images/ai_art/output/DECODE_half1.png')
        del res, pilim
        mem()

        res = coder.decode(mid.to('cuda'))
        rmid = res.clone().cpu()
        pilim = tpi(res[0])
        pilim.save('/home/johnim/data-0/media/images/ai_art/output/DECODE_mid.png')
        del res, pilim
        mem()

        r = cat((r1, r2), dim=3).to('cuda')
        pilim = tpi(r[0])
        pilim.save('/home/johnim/data-0/media/images/ai_art/output/DECODE_halves.png')
        del pilim
        mem()

        rsize = [s if i != 3 else int(s/2) for i,s in enumerate(rmid.shape)]
        print(rsize)

        naughts = zeros(rsize).to('cuda')

        print(naughts.shape)
        print(rmid.shape)

        rmid_expand = cat((naughts, rmid.to('cuda'), naughts), dim=3)

        singles = ones_like(naughts).to('cuda')

        singles_sliver = split(singles, 16, dim=3)[0]
        naughts_chunk = split(naughts, naughts.shape[3] - 16, dim=3)[0]

        mask = cat((singles, singles_sliver, naughts_chunk, naughts_chunk, singles_sliver, singles), dim=3)

        b = Blur(15, sigma=5.5).to('cuda')

        mask = b.forward(mask)

        mask_inv = ones_like(mask) - mask

        rfin = r * mask + rmid_expand * mask_inv
        pilim = tpi(rfin[0])
        pilim.save('/home/johnim/data-0/media/images/ai_art/output/DECODE_b_fin.png')
        del pilim, rfin, r, rmid_expand, r1, r2, rmid, naughts, singles
        mem()

        #mem()
        #res = coder.decode(quarters0[1].to('cuda'))
        #pilim = to_pil_image(res[0])
        #pilim.save('/home/johnim/data-0/media/images/ai_art/output/DECODE_1.png')
        #del res, pilim

        #mem()
        #res = coder.decode(quarters1[0].to('cuda'))
        #pilim = to_pil_image(res[0])
        #pilim.save('/home/johnim/data-0/media/images/ai_art/output/DECODE_3.png')
        #del res, pilim

        #mem()
        #res = coder.decode(quarters1[1].to('cuda'))
        #pilim = to_pil_image(res[0])
        #pilim.save('/home/johnim/data-0/media/images/ai_art/output/DECODE_4.png')
        #del res, pilim
