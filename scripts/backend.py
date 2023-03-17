
import traceback, argparse, os, sys, glob, time, gc, threading
import numpy as np
from omegaconf import OmegaConf
from PIL import ImageTk, Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from contextlib import contextmanager, nullcontext
from random import randint
from queue import Queue
from torch import autocast, load, no_grad, clamp, stack, randn, lerp, cuda, device, from_numpy, tensor, nn, cat, einsum, save, transpose, rot90, norm
from time import sleep

import k_diffusion as K
from k_diffusion.sampling import BrownianTreeNoiseSampler

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog as fd

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from johnim.images import *
from johnim.state import *
from johnim.sock import *

from io import BytesIO

v1_4_latent_preview_matrix = [
    [0.298, 0.207, 0.208],
    [0.187, 0.286, 0.173],
    [-0.158, 0.189, 0.264],
    [-0.184, -0.271, -0.473]
    ]

def patch_conv(**patch):
    c = nn.Conv2d
    init = c.__init__
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    c.__init__ = __init__

#patch_conv(padding_mode='circular')

class CmdCallback():
    def __init__(self, on_done=None, on_start=None):
        self.done = on_done
        self.start = on_start

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond1, cond2, scale, interp):
        # x.shape = 1, 4, (xResolution / 8), (yResolution / 8)
        # sigma.shape = 1
        # cond.shape = uncond.shape = 1, 77, 768
        mult = 3 if interp != 0 and interp != 1 else 2
        x_in = cat([x]*mult)
        sigma_in = cat([sigma]*mult)
        if interp == 0:
            cond_in = cat([uncond, cond1])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        elif interp == 1:
            cond_in = cat([uncond, cond2])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        else:
            cond_in = cat([uncond, cond1, cond2])
            uncond, cond1, cond2 = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(3)
            cond = cond1 + (cond2 - cond1) * interp
        return uncond + (cond - uncond) * scale

class CFGDenoiserCartesian(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond1, cond2, scale, interp):
        # x.shape = 1, 4, (xResolution / 8), (yResolution / 8)
        # sigma.shape = 1
        # cond.shape = uncond.shape = 1, 77, 768
        mult = 3 if interp != 0 and interp != 1 else 2
        x_in = cat([x]*mult)
        sigma_in = cat([sigma]*mult)
        
        if interp == 0:
            cond_in = cat([uncond, cond1])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            origin = uncond
        elif scale == 0:
            cond_in = cat([uncond, cond2])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            origin = uncond
        else:
            cond_in = cat([uncond, cond1, cond2])
            uncond, cond, cond2 = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(3)
            origin = uncond + (cond - uncond) * scale
            return origin + (cond2 - uncond) * interp
        return uncond + (cond - uncond) * scale

class CFGDenoiserExperimental(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond1, cond2, scale, interp):
        # x.shape = 1, 4, (xResolution / 8), (yResolution / 8)
        # sigma.shape = 1
        # cond.shape = uncond.shape = 1, 77, 768
        mult = 3 if interp != 0 and interp != 1 else 2
        x_in = cat([x]*mult)
        sigma_in = cat([sigma]*mult)
        
        print('='*20)
        print(f'sigma {sigma}')

        if interp == 0:
            cond_in = cat([uncond, cond1])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            origin = uncond
        elif interp == 1:
            cond_in = cat([uncond, cond2])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            origin = uncond
        else:
            cond_in = cat([uncond, cond1, cond2])
            uncond, cond, cond2 = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(3)
            origin = uncond + (cond - uncond) * scale * sigma
            return origin + (cond2 - uncond) * interp * sigma
        return uncond + (cond - uncond) * scale * sigma

class CFGDenoiserDerhy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def in40(self, v):
        return v if v < 40 else 40

    def forward(self, x, sigma, uncond, cond1, cond2, scale, interp):
        # x.shape = 1, 4, (xResolution / 8), (yResolution / 8)
        # sigma.shape = 1
        # cond.shape = uncond.shape = 1, 77, 768
        mult = 3 if interp != 0 and interp != 1 else 2
        x_in = cat([x]*mult)
        sigma_in = cat([sigma]*mult)
        
        if interp == 0:
            cond_in = cat([uncond, cond1])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            origin = uncond
        elif interp == 1:
            cond_in = cat([uncond, cond2])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            origin = uncond
        else:
            cond_in = cat([uncond, cond1, cond2])
            uncond, cond, cond2 = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(3)
            nuncond = norm(uncond)
            dif = cond - uncond
            
            vv1 = self.in40(nuncond / norm(dif))

            origin = uncond + scale*dif*vv1
            dif = cond2 - uncond
            vv2 = self.in40(nuncond / norm(dif))
            return origin + interp*dif*vv2
            
        dif = cond-uncond
        ndif = norm(dif)
        nun = norm(uncond)
        vv = self.in40(nun / ndif)
        return uncond + scale*dif*vv

class CFGMaskedDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond1, cond2, scale, mask, x0, xi, interp):
        mask_inv = 1. - mask
        
        x_in = x#(x0 * mask_inv) + (mask * x)

        x_in = cat([x_in] * 2)
        sigma_in = cat([sigma] * 2)
        cond_in = cat([uncond, cond1])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        denoised = uncond + (cond - uncond) * scale

        denoised = (x0 * mask_inv) + (mask * denoised)

        return denoised

def sample_init(state, c1, c2, uc):
    def i_callback(o):
        state.engine_status.config(text=f'{state.temp_status_text}, step {o["i"]+1}', fg='green')
        if o["i"] % state.update_modulus == 0:
            resized = nn.functional.interpolate(o["denoised"], size=(state.h , state.w), mode='bilinear')
            state.progress_image = ImageResult(einsum('...lhw,lr -> ...rhw', resized, state.preview_matrix)[0], state, is_preview=True)
            state.update_progress_canvas()
            w,h = state.progress_image.pil_image.size
            state.transmission_queue.put((CODE_IMAGE_RESULT, w.to_bytes(4, 'little') + h.to_bytes(4, 'little') + state.progress_image.pil_image.tobytes()))
            #state.progress_image = ImageResult(state.model.decode_first_stage(o['denoised'])[0], state, is_preview=True)

    (init_w, init_h) = state.init_image.size
    if init_w != state.w or init_h != state.h:
        init = resize_torch_image(state.init_image.torch_image, state.w, state.h).to(device("cuda"))
    else:
        init = state.init_image.torch_image.to(device("cuda"))
    init = init[None]
    init = repeat(init, '1 ... -> b ...', b=state.batch_size)
    init_latent = state.model.get_first_stage_encoding(state.model.encode_first_stage(init))  # move to latent space
    
    assert 0. <= state.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(state.strength * state.steps)
    if state.sampler_name.startswith('k_'):
        kmodel_wrap = K.external.CompVisDenoiser(state.model)
        sigmas = kmodel_wrap.get_sigmas(state.steps)
        sigma_sched = sigmas[state.steps - t_enc - 1:]

        noise = state.start_code * sigmas[state.steps - t_enc - 1]

        init_noisy = init_latent + noise

        if state.use_mask:
            
            mask = state.mask_image.torch_image.cpu()

            mask = nn.functional.interpolate(mask[0][None][None], size=init_latent.shape[-2:])
            mask = from_numpy(np.tile(mask, (4, 1, 1))).to(device('cuda'))

            if state.obliterate:
                init_noisy = (mask * noise) + ((1.-mask) * init_noisy)

            kmodel_wrap_cfg = CFGMaskedDenoiser(kmodel_wrap)
            samples = K.sampling.__dict__[f'sample_{state.sampler_name[2:]}'](kmodel_wrap_cfg, init_noisy, sigma_sched, 
                        extra_args={'cond1': c1, 'cond2': c2, 'uncond': uc, 'scale': state.scale, 'mask': mask, 'x0': init_latent, 'xi': init_noisy, 'interp': state.interp}, 
                        disable=False)#, 
                        #callback=i_callback)
        else:
            kmodel_wrap_cfg = CFGDenoiserCartesian(kmodel_wrap)
            samples = K.sampling.__dict__[f'sample_{state.sampler_name[2:]}'](kmodel_wrap_cfg, init_noisy, sigma_sched, #noise_sampler=BrownianTreeNoiseSampler(), 
                        extra_args={'cond1': c1, 'cond2': c2, 'uncond': uc, 'scale': state.scale, 'interp': state.interp}, 
                        disable=False)#, 
                        #callback=i_callback)
    else:
        if state.use_mask:
            print('Masking not yet implemented for non-k diffusers :c')
        if state.sampler_name == 'plms':
            raise Exception('No inits with PLMS :c')
        else:
            sampler = state.ddim_sampler
        sampler.make_schedule(ddim_num_steps=state.steps, ddim_eta=state.ddim_eta, verbose=False)
        z_enc = sampler.stochastic_encode(init_latent, tensor([t_enc]*state.batch_size).to(device("cuda")))
        samples = sampler.decode(z_enc, c1, t_enc, unconditional_guidance_scale=state.scale,
                                                unconditional_conditioning=uc, step_callback=lambda i:state.engine_status.config(text=f'{state.temp_status_text}, step {i}', fg='green'))#, img_callback=i_callback_b)
    return samples

# "CFG" = classifier-free guidance

def sample(state, c1, c2, uc):
    def i_callback(o):
        state.engine_status.config(text=f'{state.temp_status_text}, step {o["i"]+1}', fg='green')
        if o["i"] % state.update_modulus == 0:
            resized = nn.functional.interpolate(o["denoised"], size=(state.h , state.w), mode='bilinear')
            state.progress_image = ImageResult(einsum('...lhw,lr -> ...rhw', resized, state.preview_matrix)[0], state, is_preview=True)
            state.update_progress_canvas()
            w,h = state.progress_image.pil_image.size
            state.transmission_queue.put((CODE_IMAGE_RESULT, w.to_bytes(4, 'little') + h.to_bytes(4, 'little') + state.progress_image.pil_image.tobytes()))
    def i_callback_b(img, i):
        state.engine_status.config(text=f'{state.temp_status_text}, step {i+1}', fg='green')
        if i % state.update_modulus == 0:
            state.progress_image = ImageResult(state.model.decode_first_stage(img)[0], state, is_preview=True)
            state.update_progress_canvas()

    if state.sampler_name.startswith('k_'):
        kmodel_wrap = K.external.CompVisDenoiser(state.model)
        sigmas = kmodel_wrap.get_sigmas(state.steps)
        x = state.start_code * sigmas[0]
        kmodel_wrap_cfg = CFGDenoiserCartesian(kmodel_wrap)
        samples = K.sampling.__dict__[f'sample_{state.sampler_name[2:]}'](kmodel_wrap_cfg, x, sigmas, 
                                                                            extra_args={
                                                                                'cond1': c1,
                                                                                'cond2': c2, 
                                                                                'uncond':uc, 
                                                                                'scale': state.scale,
                                                                                'interp': state.interp},
                                                                            disable=False)#,
                                                                            #callback=i_callback)
    else:
        if state.sampler_name == 'plms':
            sampler = state.plms_sampler
        else:
            sampler = state.ddim_sampler
        shape = [state.c, state.h // state.f, state.w // state.f]
        samples, losses_maybe_idk = sampler.sample(S=state.steps,
                                    conditioning=c1,
                                    batch_size=state.batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=state.scale,
                                    unconditional_conditioning=uc,
                                    eta=state.ddim_eta,
                                    x_T=state.start_code)#,
                                    #step_callback=lambda i:state.engine_status.config(text=f'{state.temp_status_text}, step {i}', fg='green'),
                                    #img_callback=i_callback_b)
    return samples

async def do_run(state):
    if not state.fixed_seed:
        state.seed = randint(1, 100_000_000)
    current_seed = seed_everything(state.seed)
    set_start_code(state)

    for prompt_index, prompts in enumerate(tqdm(state.data, desc="data")):
        uc = None
        
        if state.data_neg[prompt_index][0].startswith('#'):
            prompt_split = state.data_neg[prompt_index][0][1:].split('@')
            p = prompt_split[1]
            n = int(prompt_split[0])
            uc = stack([load(f'clip_embeds/{p}.pt')[n].to('cuda')]*state.batch_size)
        elif state.data_neg[prompt_index][0].startswith('@'):
            uc = cat([load(f'clip_embeds/{state.data_neg[prompt_index][0][1:]}.pt').to('cuda')] * state.batch_size)
        else:
            uc = state.model.get_learned_conditioning(state.data_neg[prompt_index])
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        if prompts[0].startswith('#'):
            prompt_split = prompts[0][1:].split('@')
            p = prompt_split[1]
            n = int(prompt_split[0])
            c1 = stack([load(f'clip_embeds/{p}.pt')[n].to('cuda')]*state.batch_size)
        elif prompts[0].startswith('@'):
            c1 = cat([load(f'clip_embeds/{prompts[0][1:]}.pt').to('cuda')]*state.batch_size)
        else:
            c1 = state.model.get_learned_conditioning(prompts)
        
        if state.data2[prompt_index][0].startswith('#'):
            prompt_split = state.data2[prompt_index][0][1:].split('@')
            p = prompt_split[1]
            n = int(prompt_split[0])
            c2 = stack([load(f'clip_embeds/{p}.pt')[n].to('cuda')]*state.batch_size)
        elif state.data2[prompt_index][0].startswith('@'):
            c2 = cat([load(f'clip_embeds/{state.data2[prompt_index][0][1:]}.pt').to('cuda')]*state.batch_size)
        else:
            c2 = state.model.get_learned_conditioning(state.data2[prompt_index])
            
        #c = lerp(c1, c2, state.interp)

        gc.collect()
        cuda.empty_cache()

        if state.use_init and (state.init_image is not None):
            samples = sample_init(state, c1, c2, uc)
        else:
            samples = sample(state, c1, c2, uc)

        save(samples, f'{state.outdir}/testo2.pt')

        try:
            x_samples = state.model.decode_first_stage(samples)
        
            for index, x_sample in enumerate(x_samples):
                result = ImageResult(x_sample, state)
                if not state.mass_mode:
                    state.image_results.append(result)
                    w,h = result.pil_image.size
                    #state.transmission_queue.put((CODE_IMAGE_RESULT, w.to_bytes(4, 'little') + h.to_bytes(4, 'little') + result.pil_image.tobytes()))
        except Exception as e:
            for index, x_sample in enumerate(samples):
                result = ImageResult(x_sample, state)
                if not state.mass_mode:
                    state.image_results.append(result)
                    w,h = result.pil_image.size
                    #state.transmission_queue.put((CODE_IMAGE_RESULT, w.to_bytes(4, 'little') + h.to_bytes(4, 'little') + result.pil_image.tobytes()))
        


        #if (not state.use_init):
        #    for index, x_sample in enumerate(x_samples_TESTO_0):
        #        result = ImageResult(x_sample, state)
        #        if not state.mass_mode:
        #            state.image_results.append(result)
        
        state.update_ui_images()

def do_ucommand(command, args, state):
    print("Urgent command!")
    return True

def do_file_command(command, args, state):
    if command == 'save_thumb':
        state.save_thumb(int(args))
    elif command == 'load_image_file':
        image = load_torch_image(args)
        state.image_results.append(ImageResult(image, state, loaded=True))
        state.update_ui_images()
    elif command == 'load_image_dir':
        files = os.listdir(args)
        files = [os.path.join(args, f) for f in files if f.endswith(".png") or f.endswith(".jpg")]
        for filename in files:
            image = load_torch_image(filename)
            state.image_results.append(ImageResult(image, state, loaded=True))
        state.update_ui_images()
    elif command == 'exit':
        return False
    return True

async def do_command(command, args, callbacks, state):
    print(f"Command [{command}] args [{args}]")
    if callbacks.start is not None:
        state.discord_callbacks += [callbacks.start()]
    state.transmission_queue.put((CODE_COMMAND, f'[{command}] [{args}]'.encode('ascii')))
    if command == "exit":
        return False
    #state.engine_status.config(text='Command [{}] with args [{}]'.format(command, args), fg='blue')
    if command == 'w':
        state.w = int(args) * 64
    #elif command == 'eval':
    #    eval(args) # TODO delete this if this app ever accepts commands over the network
    elif command == 'h':
        state.h = int(args) * 64
    elif command == 'wh':
        wh = args.split(' ')
        state.w = int(wh[0]) * 64
        state.h = int(wh[1]) * 64
    elif command == 'scale':
        state.scale = float(args)
    elif command == 'steps':
        state.steps = int(args)
    elif command == 'batch_size':
        state.batch_size = int(args)
    elif command == 'seed':
        set_seed(state, args)
    elif command == 'interp':
        state.interp = float(args)
    elif command == 'update_modulus':
        state.update_modulus = int(args)
    elif command == 'p':
        set_p1(state, args)
    elif command == 'p_secondary':
        set_p2(state, args)
    elif command == 'p_negative':
        set_neg_p(state, args)
    elif command == 'strength':
        state.strength = float(args)
    elif command == 'aes_steps':
        state.model.cond_stage_model.T = int(args)
    elif command == 'aes_rate':
        state.model.cond_stage_model.lr = float(args)
    elif command == 'aes_path':
        state.model.cond_stage_model.aesthetic_embedding_path = args
    elif command == 'init':
        state.set_init_from_thumbs(int(args))
    elif command == 'clear':
        state.clear_thumb(int(args))
    elif command == 'save':
        state.save_thumb(int(args))
    elif command == "sampler":
        state.sampler_name = args
    elif command == "color_match":
        indices = [int(x) for x in args.split(' ')]
        result = color_match(state, indices[0], indices[1])
        state.image_results.append(result)
        state.update_ui_images()
    elif command == "sharpen":
        args = args.split(' ')
        result = sharpen(state, int(args[0]), float(args[1]))
        state.image_results.append(result)
        state.update_ui_images()
    elif command == "refresh":
        gc.collect()
        cuda.empty_cache()
    elif command == "go":
        runs = 1 if args == "" else int(args)
        for run in range(runs):
            state.temp_status_text = f'Sample {run+1} of {runs}'
            #state.engine_status.config(text=state.temp_status_text, fg='green')
            if not state.urgent_queue.empty():
                return True # if there are weird reaction emoji shenanigans in the discord bot, look here
            await do_run(state)
            gc.collect()
            cuda.empty_cache()
    elif command == "go-interp":
        runs = int(args)
        original_interp = state.interp
        interps = [x / (runs-1) for x in range(runs)]
        for index, interp in enumerate(interps):
            state.temp_status_text = f'Sample {index+1} of {runs}'
            #state.engine_status.config(text=state.temp_status_text, fg='green')
            if not state.urgent_queue.empty():
                return True # if there are weird reaction emoji shenanigans in the discord bot, look here
            state.interp = interp
            await do_run(state)
            gc.collect()
            cuda.empty_cache()
        state.interp = original_interp

    else:
        print("[Unrecognized command]")
    if callbacks.done is not None:
        state.discord_callbacks += [callbacks.done()]
    print('Command Done.')
    #state.engine_status.config(text='Idle')
    return True

def get_model(state):
    cuda.empty_cache()

    config = OmegaConf.load(f"{state.config}")
    model = load_model_from_config(config, f"{state.ckpt}")

    if (state.embedding_path is not None):
        model.embedding_manager.load(state.embedding_path)

    dev = device("cuda") if cuda.is_available() else device("cpu")
    model = model.to(dev)

    if state.precision == 'autocast':
        # Half precision
        model.half()

    cuda.empty_cache()
    return model

def set_start_code(state):
    state.start_code = randn([state.batch_size, state.c, state.h // state.f, state.w // state.f], device=device("cuda"))

async def backend_thread(state):
    state.preview_matrix = tensor(v1_4_latent_preview_matrix).to(device('cuda'))

    state.model = get_model(state)

    state.plms_sampler = PLMSSampler(state.model)

    state.ddim_sampler = DDIMSampler(state.model)

    set_start_code(state)

    precision_scope = autocast if state.precision=="autocast" else nullcontext

    should_continue = True

    with no_grad():
        with precision_scope("cuda"):
            with state.model.ema_scope():
                while should_continue:
                    #state.engine_status.config(text='Idle', fg='yellow')
                    if not state.urgent_queue.empty():
                        (ucommand, uargs) = state.urgent_queue.get()
                        should_continue = do_ucommand(ucommand, uargs, state)
                    (command, args, callbacks) = state.command_queue.get()
                    try:
                        should_continue = await do_command(command, args, callbacks, state)
                    except Exception as ex:
                        #state.engine_status.config(text='[ERROR]', fg='red')
                        print(f'[Exception | Backend | {ex}] {traceback.format_exc()}')
                    gc.collect()
                    cuda.empty_cache()

def do_file_command(command, args, state):
    if command == 'save_thumb':
        state.save_thumb(int(args))
    elif command == 'load_image_file':
        image = load_torch_image(args)
        state.image_results.append(ImageResult(image, state, loaded=True))
        state.update_ui_images()
    elif command == 'load_image_dir':
        files = os.listdir(args)
        files = [os.path.join(args, f) for f in files if f.endswith(".png") or f.endswith(".jpg")]
        for filename in files:
            image = load_torch_image(filename)
            state.image_results.append(ImageResult(image, state, loaded=True))
        state.update_ui_images()
    elif command == 'exit':
        return False
    return True

def file_thread(state):
    should_continue = True
    while should_continue:
        (command, args) = state.file_queue.get()
        try:
            should_continue = do_file_command(command, args, state)
        except Exception as ex:
            print('[Exception | FileThread | {}] {}'.format(ex, traceback.format_exc()))

def watcher_thread(state):
    tmp_init_folder = '/tmp/stable_input'
    os.makedirs(tmp_init_folder, exist_ok=True)
    while state.running:
        inits = os.listdir(tmp_init_folder)
        if len(inits) > 0:
            do_file_command('load_image_dir', tmp_init_folder, state)
        for f in os.listdir(tmp_init_folder):
            os.remove(os.path.join(tmp_init_folder, f))
        sleep(2.)


def main():
    print('todo')
    pass

if __name__ == '__main__':
    main()
