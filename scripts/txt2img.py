# TODO
# -- multiple such buttons to save to different folders?
# buttons for saving and loading prompts/settings
# more than 2 prompts, more complicated interps?
# clean up the code a bit
# source control
# compositing
# inpainting
# screenshotting (using scrot?)
# k sampler
# -- k sampler w/ clip guidance?
# -- lms for inits?
# add tex inversion in this repo
# prompt term weighting
# file i/o on another thread, to prevent UI freezes when saving and loading data
# source control!!
# multiple inits, blended together?
# more programmable buffers and more robust command language
# buffers and pipelines?
# -- eg: [Generate] -> [Change settings, use output as init] -> [Change more settings, use output as init]
# setting presets, save + load, quick changes
# portion of an init? toggle cropping/outpainting vs downscaling/upscaling?
# perform simple image manipulations on inits?

import traceback
import argparse, os, sys, glob
import numpy as np
from omegaconf import OmegaConf
from PIL import ImageTk, Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from random import randint
import gc
import threading
from queue import Queue
from torch import autocast, load, no_grad, clamp, stack, randn, lerp, cuda, device, from_numpy, tensor

import k_diffusion as K

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog as fd

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from johnim.images import *
from johnim.state import *

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

def update_ui_images(state):
    unshown_images = [x for x in state.image_results if not x.is_shown]
    for index, canvas in enumerate(state.thumbnail_canvases):
        if state.thumbnail_content[index] is not None:
            continue
        elif len(unshown_images) > 0:
            result = unshown_images.pop()
            state.thumbnail_content[index] = result
            result.is_shown = True
            (w, h) = result.thumb_size
            canvas.create_image(w // 2, h // 2, image=result.tk_thumb)
            canvas.update()

from torch import nn, cat

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = cat([x]*2)
        sigma_in = cat([sigma]*2)
        cond_in = cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

def sample_init(state, c, uc):
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

        kmodel_wrap_cfg = CFGDenoiser(kmodel_wrap)
        samples = K.sampling.__dict__[f'sample_{state.sampler_name[2:]}'](kmodel_wrap_cfg, init_noisy, sigma_sched, extra_args={'cond': c, 'uncond': uc, 'cond_scale': state.scale}, disable=False)

    else:
        if state.sampler_name == 'plms':
            raise Exception('No inits with PLMS :c')
        else:
            sampler = state.ddim_sampler
        sampler.make_schedule(ddim_num_steps=state.steps, ddim_eta=state.ddim_eta, verbose=False)
        z_enc = sampler.stochastic_encode(init_latent, tensor([t_enc]*state.batch_size).to(device("cuda")))
        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=state.scale,
                                                unconditional_conditioning=uc,)
    return samples

def sample(state, c, uc):
    if state.sampler_name.startswith('k_'):
        kmodel_wrap = K.external.CompVisDenoiser(state.model)
        sigmas = kmodel_wrap.get_sigmas(state.steps)
        x = state.start_code * sigmas[0]
        kmodel_wrap_cfg = CFGDenoiser(kmodel_wrap)
        samples = K.sampling.__dict__[f'sample_{state.sampler_name[2:]}'](kmodel_wrap_cfg, 
                                                                            x, 
                                                                            sigmas, 
                                                                            extra_args={
                                                                                'cond': c, 
                                                                                'uncond':uc, 
                                                                                'cond_scale': state.scale}, 
                                                                            disable=False,
                                                                            callback=lambda o:state.engine_status.config(text=f'{state.temp_status_text}, step {o["i"]+1}', fg='green'))
    else:
        if state.sampler_name == 'plms':
            sampler = state.plms_sampler
        else:
            sampler = state.ddim_sampler
        shape = [state.c, state.h // state.f, state.w // state.f]
        samples, losses_maybe_idk = sampler.sample(S=state.steps,
                                    conditioning=c,
                                    batch_size=state.batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=state.scale,
                                    unconditional_conditioning=uc,
                                    eta=state.ddim_eta,
                                    x_T=state.start_code,
                                    step_callback=lambda i:state.engine_status.config(text=f'{state.temp_status_text}, step {i}', fg='green'))
    return samples

def do_run(state):
    if not state.fixed_seed:
        state.seed = randint(1, 100_000_000)
    current_seed = seed_everything(state.seed)
    set_start_code(state)

    for prompt_index, prompts in enumerate(tqdm(state.data, desc="data")):
        uc = None
        if state.scale != 1.0:
            uc = state.model.get_learned_conditioning(state.batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)
        c1 = state.model.get_learned_conditioning(prompts)
        c2 = state.model.get_learned_conditioning(state.data2[prompt_index])
        c = lerp(c1, c2, state.interp)

        if state.use_init and (state.init_image is not None):
            samples = sample_init(state, c, uc)
        else:
            samples = sample(state, c, uc)

        x_samples = state.model.decode_first_stage(samples)

        for index, x_sample in enumerate(x_samples):
            result = ImageResult(x_sample, state)
            if not state.mass_mode:
                state.image_results.append(result)
        
        update_ui_images(state)

def do_ucommand(command, args, state):
    print("Urgent command!")
    return True

def do_file_command(command, args, state):
    if command == 'save_thumb':
        save_thumb(state, int(args))
    elif command == 'load_image_file':
        image = load_torch_image(args)
        state.image_results.append(ImageResult(image, state, loaded=True))
        update_ui_images(state)
    elif command == 'load_image_dir':
        files = os.listdir(args)
        files = [os.path.join(args, f) for f in files if f.endswith(".png") or f.endswith(".jpg")]
        for filename in files:
            image = load_torch_image(filename)
            state.image_results.append(ImageResult(image, state, loaded=True))
        update_ui_images(state)
    elif command == 'exit':
        return False
    return True

def do_command(command, args, state):
    if command == "exit":
        return False
    state.engine_status.config(text='Command [{}] with args [{}]'.format(command, args), fg='blue')
    if command == 'w':
        state.w = int(args) * 64
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
        return set_seed(state, args)
    elif command == 'interp':
        state.interp = float(args)
    elif command == 'p':
        return set_p1(state, args)
    elif command == 'p2':
        return set_p2(state, args)
    elif command == 'strength':
        state.strength = float(args)
    elif command == 'init':
        set_init_from_thumbs(state, int(args))
    elif command == 'clear':
        clear_thumb(state, int(args))
    elif command == 'save':
        save_thumb(state, int(args))
    elif command == "sampler":
        state.sampler_name = args
    elif command == "color_match":
        indices = [int(x) for x in args.split(' ')]
        result = color_match(state, indices[0], indices[1])
        state.image_results.append(result)
        update_ui_images(state)
    elif command == "sharpen":
        args = args.split(' ')
        result = sharpen(state, int(args[0]), float(args[1]))
        state.image_results.append(result)
        update_ui_images(state)
    elif command == "refresh":
        gc.collect()
        cuda.empty_cache()
    elif command == "go":
        runs = 1 if args == "" else int(args)
        for run in range(runs):
            state.temp_status_text = f'Sample {run+1} of {runs}'
            state.engine_status.config(text=state.temp_status_text, fg='green')
            if not state.urgent_queue.empty():
                return True
            do_run(state)
            gc.collect()
            cuda.empty_cache()
    elif command == "go-interp":
        runs = int(args)
        original_interp = state.interp
        interps = [x / (runs-1) for x in range(runs)]
        for index, interp in enumerate(interps):
            state.temp_status_text = f'Sample {index+1} of {runs}'
            state.engine_status.config(text=state.temp_status_text, fg='green')
            if not state.urgent_queue.empty():
                return True
            state.interp = interp
            do_run(state)
            gc.collect()
            cuda.empty_cache()
        state.interp = original_interp

    else:
        print("[Unrecognized command]")
    state.engine_status.config(text='Idle')
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

def backend_thread(state):
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
                    state.engine_status.config(text='Idle', fg='yellow')
                    if not state.urgent_queue.empty():
                        (ucommand, uargs) = state.urgent_queue.get()
                        should_continue = do_ucommand(ucommand, uargs, state)
                    (command, args) = state.command_queue.get()
                    try:
                        should_continue = do_command(command, args, state)
                    except Exception as ex:
                        state.engine_status.config(text='[ERROR]', fg='red')
                        print('[Exception | Backend | {}] {}'.format(ex, traceback.format_exc()))
                    gc.collect()
                    cuda.empty_cache()

def file_thread(state):
    should_continue = True
    while should_continue:
        (command, args) = state.file_queue.get()
        try:
            should_continue = do_file_command(command, args, state)
        except Exception as ex:
            print('[Exception | FileThread | {}] {}'.format(ex, traceback.format_exc()))

def set_init_from_thumbs(state, index):
    if state.thumbnail_content[index] is None:
        return
    state.init_image = state.thumbnail_content[index]
    (w, h) = state.init_image.size
    state.init_canvas.config(width=w, height=h)
    state.init_canvas.delete('all')
    state.init_canvas.create_image(w // 2, h // 2, image=state.init_image.tk)
    state.init_canvas.update()
    state.use_init = True

def clear_thumb(state, index):
    if state.thumbnail_content[index] is None:
        return
    state.thumbnail_content[index] = None
    state.thumbnail_canvases[index].delete('all')
    update_ui_images(state)

def save_thumb(state, index):
    if state.thumbnail_content[index] is None:
        return
    state.thumbnail_content[index].save()

def main():
    state = State()
    
    state.interp = 0.0
    state.seed = 42069
    state.fixed_seed = False
    state.w = 512
    state.h = 512
    state.image_results = []
    state.outdir = 'outputs'
    state.mass_mode = False
    state.use_init = False
    state.strength = 0.5
    state.precision = 'autocast'#'full'
    state.plms = True
    state.embedding_path = None
    state.ckpt = 'models/sd-v1-4-o.ckpt'
    state.config = 'configs/stable-diffusion/v1-inference.yaml'
    state.scale = 10
    state.batch_size = 1
    state.f = 8
    state.c = 4
    state.ddim_eta = 0.0
    state.fixed_code = True
    state.steps = 50
    state.sampler_name = 'k_euler_ancestral'
    # Options are lms, dpm_2_ancestral, dpm_2, euler_ancestral, euler, heun, plms, ddim
    
    os.makedirs(state.outdir, exist_ok=True)
    state.base_count = len(os.listdir(state.outdir))

    state.command_queue = Queue()
    state.urgent_queue = Queue()
    state.file_queue = Queue()

    backend = threading.Thread(target=lambda:backend_thread(state))
    file_th = threading.Thread(target=lambda:file_thread(state))

    main_window = tk.Tk()
    main_window.configure(bg='black')

    prompt_frame = tk.Frame(main_window)
    left_panel = tk.Frame(main_window)
    right_panel = tk.Frame(main_window)
    main_frame = tk.Frame(main_window)

    prompt_frame.pack(side=tk.TOP, anchor=tk.NW)
    left_panel.pack(side=tk.LEFT, anchor=tk.NW)
    right_panel.pack(side=tk.RIGHT, anchor=tk.E)
    main_frame.pack(anchor=tk.CENTER)

    prompt_frame.configure(bg="black")
    left_panel.configure(bg="black")
    right_panel.configure(bg="black")
    main_frame.configure(bg="black")

    urgent_commands = ["cancel"]

    # Return true if the command should not pass on to the backend
    def intercept_command(command, args):
        if command.startswith('#'):
            return True # Comments
        return False

    def do(command, args):
        if intercept_command(command, args):
            return
        if command in urgent_commands:
            state.urgent_queue.put((command, args))
        else:
            state.command_queue.put((command, args))
    
    def file_do(command, args):
        state.file_queue.put((command, args))

    def doo(user_input):
        split_input = user_input.split(' ', 1)
        command = split_input[0].lower()
        if len(split_input) > 1:
            args = split_input[1]
        else:
            args = ""
        do(command, args)

    def font(n):
        return ('Consolas', n)

    p1_input = ttk.Entry(prompt_frame, width=220, font=font(16))
    p2_input = ttk.Entry(prompt_frame, width=220, font=font(16))
    p1_input.grid(row=0,column=0, columnspan=20, sticky=tk.W, pady=2)
    p2_input.grid(row=2,column=0, columnspan=20, sticky=tk.W, pady=2)
    
    command_input = scrolledtext.ScrolledText(left_panel, width=25, height=35, wrap=tk.WORD, font=font(16))

    panel_labels = State()
    panel_inputs = State()

    def toggle_mass():
        state.mass_mode = not state.mass_mode
    def toggle_init():
        state.use_init = not state.use_init

    inputs = [
        ('scale', 'Guidance Scale', 'entry', None),
        ('w', "Width", 'entry', None),
        ('h', "Height", 'entry', None),
        ('steps', "Steps", 'entry', None),
        ('batch_size', "Batch Size", 'entry', None),
        ('seed', "Seed", 'entry', None),
        ('interp', "Interp", 'entry', None),
        ('mass', "Mass Mode", 'check', toggle_mass),
        ('init', "Use Init", 'check', toggle_init),
        ('strength', "Init Liberty", 'entry', None),
        ]

    for (varname, text, input_type, com) in inputs:
        panel_labels[varname] = tk.Label(left_panel, text=text, font=font(16))
        panel_labels[varname].configure(bg="black", fg="white")
        if input_type == 'entry':
            panel_inputs[varname] = ttk.Entry(left_panel, width=7, font=font(16))
            if varname == 'seed':
                start_value = 'random'
            elif varname == 'w':
                start_value = state.w // 64
            elif varname == 'h':
                start_value = state.h // 64
            else:
                start_value = state[varname]
            panel_inputs[varname].insert(1, str(start_value))
        elif input_type == 'check':
            panel_inputs[varname] = ttk.Checkbutton(left_panel, command=com)
        else:
            print(f'Input type {input_type} not implemented')



    p1_input.insert(1, "the cutest dog in the world")
    p2_input.insert(1, "an angry man shaking his fist at a computer")


    command_input.configure(bg="black", fg="white", insertbackground="white")
    command_input.grid(row=len(inputs)+3, column=0, columnspan=2, sticky=tk.S, pady=2)

    status_label = tk.Label(left_panel, text='Status:', font=font(10))
    engine_status = tk.Label(left_panel, text='Loading...', font=font(16))

    status_label.configure(bg='black', fg='white')
    engine_status.configure(bg='black', fg='red')

    status_label.grid(row=0, column=0, sticky=tk.W, pady=2)
    engine_status.grid(row=1, column=0, columnspan=2, sticky=tk.E, pady=2)

    state.engine_status = engine_status

    for (index, (setting, z, zz, zzz)) in enumerate(inputs):
        panel_labels[setting].grid(row=index+2, column=0, sticky=tk.W, pady=2)
    for (index, (setting, z, zz, zzz)) in enumerate(inputs):
        panel_inputs[setting].grid(row=index+2, column=1, sticky=tk.E, pady=2)

    def send_settings():
        for (varname, z, input_type, zz) in inputs:
            if input_type == 'entry':
                do(varname, panel_inputs[varname].get())
        do('p', p1_input.get())
        do('p2', p2_input.get())

    def command_block_go():
        user_input = command_input.get('1.0', tk.END).split('\n')
        user_input = [x for x in user_input if x != ""]
        command_input.delete('1.0', tk.END)
        command_input.insert(tk.INSERT, '\n'.join(user_input))
        for command_string in user_input:
            doo(command_string)

    def just_go(n):
        send_settings()
        do('go', str(n))

    def just_go_ev(ev):
        just_go(1)

    setting_button = tk.Button(left_panel, text="Send Settings", command=send_settings, font=font(10))
    setting_button.grid(row=len(inputs)+2, column=0, columnspan=2, sticky=tk.S, pady=2)
        
    p1_input.bind('<Return>', lambda e: just_go(1))
    p2_input.bind('<Return>', lambda e: just_go(1))
    command_input.bind('<Control_L><Shift_L><Return>', lambda e: command_block_go())

    script_button = tk.Button(left_panel, text="Run Script", command=command_block_go, font=font(10))
    script_button.grid(row=len(inputs)+4, column=0, columnspan=2, sticky=tk.S, pady=2)

    go_numbers = [1, 5, 10, 50, 100, 500]

    def make_gobtn_cmd(x):
        def gobtn_cmd():
            just_go(x)
        return gobtn_cmd

    go_buttons = [tk.Button(prompt_frame, text="Go (x{})".format(n), command=make_gobtn_cmd(n), font=font(10)) for index, n in enumerate(go_numbers)]

    state.thumbnail_canvases = []

    def make_lclick_cmd(i):
        def lclick_cmd(event):
            if state.thumbnail_content[i] is None:
                return
            state.main_image = state.thumbnail_content[i]
            (w, h) = state.main_image.size
            state.selection_canvas.config(width=w, height=h)
            state.selection_canvas.delete('all')
            state.selection_canvas.create_image(w // 2, h // 2, image=state.main_image.tk)
            state.selection_canvas.update()
        return lclick_cmd
    def make_mclick_cmd(i):
        def mclick_cmd(event):
            clear_thumb(state, i)
        return mclick_cmd
    def make_rclick_cmd(i):
        def load_image():
            filename = fd.askopenfilename()
            file_do('load_image_file', filename)
        def load_folder():
            image_folder = fd.askdirectory()
            file_do('load_image_dir', image_folder)
        def set_init():
            set_init_from_thumbs(state, i)
        def rclick_cmd(event):
            c_menu.delete(0, 100)
            if state.thumbnail_content[i] is None:
                c_menu.add_command(label='load', command=load_image)
                #c_menu.add_command(label='load multi', command=load_image)
                c_menu.add_command(label='load folder', command=load_folder)
            else:
                c_menu.add_command(label='save', command=lambda:file_do('save_thumb', i))
                c_menu.add_command(label='gimp', command=state.thumbnail_content[i].gimp)
                c_menu.add_command(label='init', command=set_init)
            try:
                c_menu.tk_popup(event.x_root, event.y_root)
            finally:
                c_menu.grab_release()
        return rclick_cmd

    state.n_thumbnails = 10

    c_menu = tk.Menu(main_window, tearoff=0)

    for index in range(state.n_thumbnails):
        canvas = tk.Canvas(right_panel, width=256, height=256)
        canvas.grid(row=index % 5, column=(index - (index % 5)) // 5, sticky = tk.W)
        canvas.bind("<Button-1>", make_lclick_cmd(index))
        canvas.bind("<Button-2>", make_mclick_cmd(index))
        canvas.bind("<Button-3>", make_rclick_cmd(index))
        state.thumbnail_canvases += [canvas]
        canvas.configure(bg="black")
    
    main_window.bind("<Button-1>", lambda e: c_menu.unpost())
    main_window.bind("<Button-2>", lambda e: c_menu.unpost())

    state.thumbnail_content = [None] * state.n_thumbnails

    state.main_image = None

    state.init_image = None

    state.selection_canvas = tk.Canvas(main_frame, width=512, height=512)
    state.selection_canvas.grid(row=0, column = 1, rowspan=1, columnspan=1, sticky=tk.W)
    state.selection_canvas.configure(bg="black")

    state.init_canvas = tk.Canvas(main_frame, width=512, height=512)
    state.init_canvas.grid(row=0, column = 0, rowspan=1, columnspan=1, sticky=tk.W)
    state.init_canvas.configure(bg="gray")

    for index, btn in enumerate(go_buttons):
        btn.grid(row=4, column=index, sticky = tk.W, pady=2)
        btn.configure(bg="black", fg="white")

    try:
        file_th.start()
        backend.start()

        main_window.mainloop()
    except (KeyboardInterrupt, SystemExit):
        sys.exit()
        do("exit", None)
        file_do("exit", None)
        backend.join()
        file_th.join()

    do("exit", None)
    file_do("exit", None)
    backend.join()
    file_th.join()

if __name__ == "__main__":
    main()
