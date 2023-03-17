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
# show init noise when inits are off
# make config switchable at runtime

import traceback
import argparse, os, sys, glob
import numpy as np
from omegaconf import OmegaConf
from PIL import ImageTk, Image
from tqdm import tqdm, trange
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
from johnim.util import *
from johnim.sock import *
from backend import *

def main():
    state = State()
    
    state.debug_quiet = True

    state.interp = 0.0
    state.seed = 42069
    state.fixed_seed = False
    state.w = 512
    state.h = 512
    state.image_results = []
    state.outdir = 'outputs'
    state.mass_mode = False
    state.use_init = False
    state.use_mask = False
    state.strength = 0.5
    state.precision = 'autocast'#'full'
    state.plms = True
    state.embedding_path = None#'../textual_inversion/ti_results/fem/embeddings.pt'
    state.scale = 8
    state.update_modulus = 1
    state.obliterate = False
    state.batch_size = 1
    state.f = 8
    state.c = 4
    state.ddim_eta = 0.0
    state.fixed_code = True
    state.steps = 50
    # Options are lms, dpm_2_ancestral, dpm_2, euler_ancestral, euler, heun, plms, ddim, (and some more)

    # Best:
    # k_euler
    # k_dpmpp_2m (similar result to k_e, but with "harsher" edges, which can be good or bad)
    # k_euler_ancestral
    # k_dpmpp_2s_ancestral (slower than the other 3, similar result to k_e_a)
    # TODO: make a dropdown menu

    # most other samplers are similar to those 4

    # k_dpm_fast and _adaptive need more testing.

    state.sampler_name = 'k_dpmpp_2m' #'k_euler_ancestral'

    #state.ckpt = '/home/johnim/ml-quickload/pokemon.ckpt'
    #state.ckpt = '/home/johnim/ml-quickload/mdjrny-v4.ckpt'
    #state.ckpt = '/home/johnim/ml-quickload/jwst-deep-space.ckpt'
    #state.ckpt = '/home/johnim/data-0/ml_models/sd_1/jwst-deep-space.ckpt'
    #state.ckpt = '/home/johnim/ml-quickload/ctsmscl.ckpt'
    state.ckpt = '/home/johnim/ml-quickload/sd-v1-4-o.ckpt'
    #state.ckpt = '~/ml-quickload/v1-5-pruned-emaonly.ckpt'

    state.config = 'configs/stable-diffusion/v1-inference.yaml'

    # These require a reload of the whole model. That might be fixable? idk
    state.aes_steps = 0
    state.aes_rate = 0.0001
    state.aes_path = 'aesthetic_embeddings/sac_8plus.pt'
    
    os.makedirs(state.outdir, exist_ok=True)
    state.base_count = len(os.listdir(state.outdir))

    state.command_queue = Queue()
    state.urgent_queue = Queue()
    state.file_queue = Queue()
    state.transmission_queue = Queue()

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
    pn_input = ttk.Entry(prompt_frame, width=220, font=font(16))
    p1_input.grid(row=0,column=0, columnspan=20, sticky=tk.W, pady=2)
    p2_input.grid(row=2,column=0, columnspan=20, sticky=tk.W, pady=2)
    pn_input.grid(row=3,column=0, columnspan=20, sticky=tk.W, pady=2)
    
    command_input = scrolledtext.ScrolledText(left_panel, width=25, height=35, wrap=tk.WORD, font=font(16))

    panel_labels = State()
    panel_inputs = State()



    inputs = [
        ('scale', 'Guidance Scale', 'entry', None),
        ('w', "Width", 'entry', None),
        ('h', "Height", 'entry', None),
        ('steps', "Steps", 'entry', None),
        ('update_modulus', 'Update Modulus', 'entry', None),
        ('batch_size', "Batch Size", 'entry', None),
        ('seed', "Seed", 'entry', None),
        ('interp', "Interp", 'entry', None),
        ('mass', "Mass Mode", 'toggle', 'toggle_mass'),
        ('init', "Init", 'toggle', 'toggle_init'),
        ('mask', "Mask", 'toggle', 'toggle_mask'),
        ('obliterate', "Destroy masked", 'toggle', 'toggle_obliterate'),
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
        elif input_type == 'toggle':
            def make_callback(varname):
                def toggle_callback(b):
                    if b:
                        btn_text = 'ON'
                        btn_color = 'green'
                    else:
                        btn_text = 'OFF'
                        btn_color = 'red'
                    panel_inputs[varname].config(text=btn_text, bg=btn_color)
                return toggle_callback
            State.__dict__['on_'+com](state, make_callback(varname))
            def make_command(com):
                def command():
                    State.__dict__[com](state)
                return command
            panel_inputs[varname] = tk.Button(left_panel, text='OFF', command=make_command(com))
        else:
            print(f'Input type {input_type} not implemented')



    p1_input.insert(1, "the cutest dog in the world")
    p2_input.insert(1, "an angry man shaking his fist at a computer")
    pn_input.insert(1, "")


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
        do('p_secondary', p2_input.get())
        do('p_negative', pn_input.get())

    def command_block_go():
        user_input = command_input.get('1.0', tk.END).split('\n')
        user_input = [x.strip() for x in user_input]
        user_input = [x for x in user_input if x != ""]

        command_input.delete('1.0', tk.END)
        command_input.insert(tk.INSERT, '\n'.join(user_input))

        preprocessed_input = preprocess(user_input)

        for command_string in preprocessed_input:
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
    pn_input.bind('<Return>', lambda e: just_go(1))
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
            state.clear_thumb(i)
        return mclick_cmd
    def make_rclick_cmd(i):
        def load_image():
            filename = fd.askopenfilename()
            file_do('load_image_file', filename)
        def load_folder():
            image_folder = fd.askdirectory()
            file_do('load_image_dir', image_folder)
        def set_init():
            state.set_init_from_thumbs(i)
        def set_mask():
            state.set_mask_from_thumbs(i)
        def set_seed():
            panel_inputs['seed'].delete(0, tk.END)
            panel_inputs['seed'].insert(1, str(state.thumbnail_content[i].seed))
            do('seed', state.thumbnail_content[i].seed)
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
                c_menu.add_command(label='mask', command=set_mask)
                c_menu.add_command(label='seed', command=set_seed)
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
    state.mask_image = None
    state.progress_image = None

    state.selection_canvas = tk.Canvas(main_frame, width=512, height=512)
    state.selection_canvas.grid(row=0, column = 1, rowspan=1, columnspan=1, sticky=tk.W)
    state.selection_canvas.configure(bg="black")

    state.init_canvas = tk.Canvas(main_frame, width=512, height=512)
    state.init_canvas.grid(row=0, column = 0, rowspan=1, columnspan=1, sticky=tk.W)
    state.init_canvas.configure(bg="gray")

    state.mask_canvas = tk.Canvas(main_frame, width = 512, height = 512)
    state.mask_canvas.grid(row = 1, column = 0, rowspan = 1, columnspan = 1, sticky=tk.W)
    state.mask_canvas.configure(bg="white")

    state.progress_canvas = tk.Canvas(main_frame, width=512, height=512)
    state.progress_canvas.grid(row=1, column=1, rowspan=1, columnspan=1, sticky=tk.W)
    state.progress_canvas.configure(bg="black")

    for index, btn in enumerate(go_buttons):
        btn.grid(row=4, column=index, sticky = tk.W, pady=2)
        btn.configure(bg="black", fg="white")

    backend = threading.Thread(target=lambda:backend_thread(state))
    file_th = threading.Thread(target=lambda:file_thread(state))
    watcher = threading.Thread(target=lambda:watcher_thread(state))
    listen_th = threading.Thread(target=lambda:listener(state, doo))
    transmitter = threading.Thread(target=lambda:establish_transmitter(state))

    try:
        file_th.start()
        backend.start()
        watcher.start()
        listen_th.start()
        transmitter.start()

        send_settings()

        main_window.mainloop()
    except (KeyboardInterrupt, SystemExit):
        do("exit", None)
        file_do("exit", None)
        state.running = False
        backend.join()
        file_th.join()
        watcher.join()
        listen_th.join()
        transmitter.join()
        sys.exit()

    do("exit", None)
    file_do("exit", None)
    state.running = False
    backend.join()
    file_th.join()
    watcher.join()
    listen_th.join()
    transmitter.join()

if __name__ == "__main__":
    main()
