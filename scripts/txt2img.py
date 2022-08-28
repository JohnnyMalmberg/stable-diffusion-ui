# TODO "away mode" - don't bother showing the images (or keeping them in RAM), just pump them out into a folder
# vs default "active mode":
# don't save every image unless mass mode is on
# add a button to press to save the image
# -- multiple such buttons to save to different folders?
# button to throw away image
# buttons for saving and loading settings
# clean up the code a bit
# source control
# image to image
# loading images
# compositing
# inpainting
# gimp export as alternative to saving
# screenshotting (using scrot?)
# k sampler
# add tex inversion in this repo
# prompt term weighting


import traceback
import argparse, os, sys, glob
import numpy as np
from omegaconf import OmegaConf
from PIL import ImageTk, Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from random import randint
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import gc
import threading
from queue import Queue
from torch import autocast, load, no_grad, clamp, stack, randn, lerp, cuda, device

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

class ImageResult:
    def __init__(self, image, state):
        self.state = state # single shared god object anti-pattern
        self.image = image
        self.seed = state.seed
        self.tk = ImageTk.PhotoImage(image)
        self.size = (image.width, image.height)
        self.thumb_size = thumb_size(image.width, image.height)
        self.thumb = image.resize(self.thumb_size)
        self.tk_thumb = ImageTk.PhotoImage(self.thumb)
        self.is_shown = False
        if not state.skip_save:
            self.save()

    def save(self):
        name = f"{self.seed}_{self.state.base_count:03}.png"
        self.image.save(os.path.join(self.state.sample_path, name))
        self.state.base_count += 1


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

def do_run(state):
    if not state.fixed_seed:
        state.seed = randint(1, 100_000_000)
    current_seed = seed_everything(state.seed)
    for prompt_index, prompts in enumerate(tqdm(state.data, desc="data")):
        uc = None
        if state.scale != 1.0:
            uc = state.model.get_learned_conditioning(state.batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)
        c1 = state.model.get_learned_conditioning(prompts)
        c2 = state.model.get_learned_conditioning(state.data2[prompt_index])
        c = lerp(c1, c2, state.interp)
        shape = [state.C, state.H // state.f, state.W // state.f]
        samples_ddim, other_junk = state.sampler.sample(S=state.steps,
                                        conditioning=c,
                                        batch_size=state.batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=state.scale,
                                        unconditional_conditioning=uc,
                                        eta=state.ddim_eta,
                                        x_T=state.start_code)

        #print("Other junk:")
        #print(other_junk)
        # i think it's losses?

        x_samples_ddim = state.model.decode_first_stage(samples_ddim)
        x_samples_ddim = clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        for index, x_sample in enumerate(x_samples_ddim):
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            sample_image = Image.fromarray(x_sample.astype(np.uint8))
            state.image_results.append(ImageResult(sample_image, state))
                #state.image_names[index] = f"{state.seed}_{state.base_count:03}.png"
                #sample_image.save(os.path.join(state.sample_path, state.image_names[index]))
                #tk_img = ImageTk.PhotoImage(sample_image)
                #(w_t, h_t) = thumb_size(state.W, state.H)
                #shrunk_image = sample_image.resize((w_t, h_t))
                #tk_thumb = ImageTk.PhotoImage(shrunk_image)
                #state.tk_thumbs[index] = tk_thumb
                #state.tk_images[index] = tk_img
                #state.thumbnail_canvases[index].create_image(w_t // 2, h_t // 2, image=tk_thumb)
                #state.thumbnail_canvases[index].update()
                #state.main_image = sample_image
                #state.main_tk_image = tk_img
                #state.full_canvas.create_image(state.W // 2, state.H // 2, image=tk_img)
                #state.full_canvas.update()
                #state.images[index] = sample_image
                #state.base_count += 1
        
        update_ui_images(state)
        

        #if not state.skip_grid:
        #    all_samples.append(x_samples_ddim)

def do_ucommand(command, args, state):
    print("Urgent command!")
    return True

def do_command(command, args, state, urgent_queue):
    print('Command [{}] with args [{}]'.format(command, args))
    if command == "exit":
        return False
    elif command == "f":
        state.f = int(args)
        if state.fixed_code:
            state.start_code = randn([state.batch_size, state.C, state.H // state.f, state.W // state.f], device=dev)
    #elif command == "c":
    #    state.C = int(args)
    #    if state.fixed_code:
    #        state.start_code = torch.randn([state.batch_size, state.C, state.H // state.f, state.W // state.f], device=device)

    elif command == "refresh":
        gc.collect()
        cuda.empty_cache()
    elif command == "help":
        print("HELP")
        print("  > p [T]\n    Set prompt to T\n")
        print("  > n [N]\n    Set batch size to N\n")
        print("  > steps [N]\n    Set steps to N\n")
        print("  > w [X]\n    Set width to X\n")
        print("  > h [Y]\n    Set height to Y\n")
        print("  > f [N]\n    Set downsampling factor to N\n")
        #print("  > c [N]\n    Set latent channels to N\n")
        print("  > s [N]\n    Set scale to N\n")
        print("  > seed [N]\n    Set seed to N\n")
        print("  > go [N]\n    Start N runs (default 1)\n")
        print("  > go-interp [N]\n    Start N interpolated runs between 0 and 1\n")
        print("  > interp [C]\n    Set prompt interpolation to C\n")
    elif command == "go":
        runs = 1 if args == "" else int(args)
        for run in range(runs):
            if not urgent_queue.empty():
                return True
            do_run(state)
            gc.collect()
            cuda.empty_cache()
    elif command == "go-interp":
        runs = int(args)
        original_interp = state.interp
        interps = [x / (runs-1) for x in range(runs)]
        for interp in interps:
            if not urgent_queue.empty():
                return True
            state.interp = interp
            do_run(state)
            gc.collect()
            cuda.empty_cache()
        state.interp = original_interp

    else:
        print("[Unrecognized command]")
    return True

def get_model(state):
    cuda.empty_cache()

    config = OmegaConf.load(f"{state.config}")
    model = load_model_from_config(config, f"{state.ckpt}")

    if (state.embedding_path is not None):
        model.embedding_manager.load(state.embedding_path)

    dev = device("cuda") if cuda.is_available() else device("cpu")
    model = model.to(dev)

    # Half precision
    model.half()

    cuda.empty_cache()
    return model

def backend_thread(state, command_queue, urgent_queue):
    if state.laion400m:
        print("Falling back to LAION 400M model...")
        state.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        state.ckpt = "models/ldm/text2img-large/model.ckpt"
        state.outdir = "outputs/txt2img-samples-laion400m"

    state.model = get_model(state)

    if state.plms:
        state.sampler = PLMSSampler(state.model)
    else:
        state.sampler = DDIMSampler(state.model)

    os.makedirs(state.outdir, exist_ok=True)
    outpath = state.outdir

    if state.from_file:
        print(f"reading prompts from {state.from_file}")
        with open(state.from_file, "r") as f:
            state.data = f.read().splitlines()
            state.data = list(chunk(data, batch_size))

    state.sample_path = os.path.join(outpath, "samples")
    os.makedirs(state.sample_path, exist_ok=True)
    state.base_count = len(os.listdir(state.sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    state.start_code = None
    if state.fixed_code:
        state.start_code = randn([state.batch_size, state.C, state.H // state.f, state.W // state.f], device=dev)

    precision_scope = autocast if state.precision=="autocast" else nullcontext

    should_continue = True

    with no_grad():
        with precision_scope("cuda"):
            with state.model.ema_scope():
                while should_continue:
                    if not urgent_queue.empty():
                        (ucommand, uargs) = urgent_queue.get()
                        should_continue = do_ucommand(ucommand, uargs, state)
                    (command, args) = command_queue.get()
                    try:
                        should_continue = do_command(command, args, state, urgent_queue)
                    except Exception as ex:
                        print('[Exception | {}] {}'.format(ex, traceback.format_exc()))
                    gc.collect()
                    cuda.empty_cache()

def thumb_size(w, h):
    (b, s) = (h,w) if w < h else (w,h)
    b_t = 128 if s >= 128 else s
    s_t = (b_t * s) // b
    return (s_t, b_t) if w < h else (b_t, s_t) 

def set_size(state, w, h):
    if w == state.W and h == state.H:
        return True
    if w is None:
        w = state.W
    if h is None:
        h = state.H
    state.W = w
    state.H = h
    #(w_t, h_t) = thumb_size(w, h)
    #for canvas in state.thumbnail_canvases:
    #    canvas.configure(width=w_t, height=h_t)
    state.full_canvas.configure(width=w, height=h)
    return True

def set_scale(state, scale):
    if state.scale == scale:
        return True
    state.scale = scale
    return True

def set_steps(state, steps):
    if state.steps == steps:
        return True
    state.steps = steps
    return True

def set_batch_size(state, size):
    if state.batch_size == size:
        return True
    state.batch_size = size
    return True

def set_seed(state, seed):
    if seed in ["random", "rand", "r"]:
        if state.fixed_seed == False:
            return True
        state.fixed_seed = False
        return True
    seed = int(seed)
    if state.seed == seed and state.fixed_seed:
        return True
    state.seed = seed
    state.fixed_seed = True
    return True

def set_interp(state, interp):
    if state.interp == interp:
        return True
    state.interp = interp
    return True

def set_p1(state, p):
    state.data = [state.batch_size * [p]]
    return True

def set_p2(state, p):
    state.data2 = [state.batch_size * [p]]
    return True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a million little question marks and exclamation points, red and black",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: batch_size)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/sd-v1-4-o.ckpt",#ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint",
        default=None
    )

    state = parser.parse_args()
    state.interp = 0.0
    state.fixed_seed = False
    state.W = 512
    state.H = 512
    state.image_results = []

    command_queue = Queue()
    urgent_queue = Queue()

    backend = threading.Thread(target=lambda:backend_thread(state, command_queue, urgent_queue))

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
        if command == 'w':
            return set_size(state, int(args) * 64, None)
        elif command == 'h':
            return set_size(state, None, int(args) * 64)
        elif command == 'wh':
            wh = args.split(' ')
            return set_size(state, int(wh[0]) * 64, int(wh[1]) * 64)
        elif command == 's':
            return set_scale(state, float(args))
        elif command == 'steps':
            return set_steps(state, int(args))
        elif command == 'n':
            return set_batch_size(state, int(args))
        elif command == 'seed':
            return set_seed(state, args)
        elif command == 'interp':
            return set_interp(state, float(args))
        elif command == 'p':
            return set_p1(state, args)
        elif command == 'p2':
            return set_p2(state, args)
        return False

    def do(command, args):
        if intercept_command(command, args):
            return
        if command in urgent_commands:
            urgent_queue.put((command, args))
        else:
            command_queue.put((command, args))
        

    def doo(user_input):
        split_input = user_input.split(' ', 1)
        command = split_input[0].lower()
        if len(split_input) > 1:
            args = split_input[1]
        else:
            args = ""
        do(command, args)

    p1_input = ttk.Entry(prompt_frame, width=220, font=("Consolas", 16))
    p2_input = ttk.Entry(prompt_frame, width=220, font=("Consolas", 16))
    p1_input.grid(row=0,column=0, columnspan=20, sticky=tk.W, pady=2)
    p2_input.grid(row=2,column=0, columnspan=20, sticky=tk.W, pady=2)
    
    command_input = scrolledtext.ScrolledText(left_panel, width=25, height=35, wrap=tk.WORD, font=("Consolas", 16))

    s_lab = tk.Label(left_panel, text="Guidance Scale", font=("Consolas", 16))
    w_lab = tk.Label(left_panel, text="Width", font=("Consolas", 16))
    h_lab = tk.Label(left_panel, text="Height", font=("Consolas", 16))
    step_lab = tk.Label(left_panel, text="Steps", font=("Consolas", 16))
    n_lab = tk.Label(left_panel, text="Batch Size", font=("Consolas", 16))
    seed_lab = tk.Label(left_panel, text="Seed", font=("Consolas", 16))
    interp_lab = tk.Label(left_panel, text="Interp", font=("Consolas", 16))

    s_input = ttk.Entry(left_panel, width=5, font=("Consolas", 16))
    w_input = ttk.Entry(left_panel, width=5, font=("Consolas", 16))
    h_input = ttk.Entry(left_panel, width=5, font=("Consolas", 16))
    step_input = ttk.Entry(left_panel, width=5, font=("Consolas", 16))
    n_input = ttk.Entry(left_panel, width=5, font=("Consolas", 16))
    seed_input = ttk.Entry(left_panel, width=5, font=("Consolas", 16))
    interp_input = ttk.Entry(left_panel, width=5, font=("Consolas", 16))

    p1_input.insert(1, "the cutest dog in the world")
    p2_input.insert(1, "an angry man shaking his fist at a computer")
    s_input.insert(1, str(state.scale))
    w_input.insert(1, str(state.W // 64))
    h_input.insert(1, str(state.H // 64))
    step_input.insert(1, str(state.steps))
    n_input.insert(1, str(state.batch_size))
    seed_input.insert(1, "random")
    interp_input.insert(1, str(state.interp))
    
    #p1_input.configure(fieldbackground="black", fg="white", insertbackground="white")
    #p2_input.configure(fieldbackground="black", fg="white", insertbackground="white")
    command_input.configure(bg="black", fg="white", insertbackground="white")
    s_lab.configure(bg="black", fg="white")
    w_lab.configure(bg="black", fg="white")
    h_lab.configure(bg="black", fg="white")
    step_lab.configure(bg="black", fg="white")
    n_lab.configure(bg="black", fg="white")
    seed_lab.configure(bg="black", fg="white")
    interp_lab.configure(bg="black", fg="white")

    command_input.grid(row=7, column=0, columnspan=2, sticky=tk.S, pady=2)

    s_lab.grid(row=0, column=0, sticky=tk.W, pady=2)
    w_lab.grid(row=1, column=0, sticky=tk.W, pady=2)
    h_lab.grid(row=2, column=0, sticky=tk.W, pady=2)
    step_lab.grid(row=3, column=0, sticky=tk.W, pady=2)
    n_lab.grid(row=4, column=0, sticky=tk.W, pady=2)
    seed_lab.grid(row=5, column=0, sticky=tk.W, pady=2)
    interp_lab.grid(row=6, column=0, sticky=tk.W, pady=2)

    s_input.grid(row=0, column=1, sticky=tk.E, pady=2)
    w_input.grid(row=1, column=1, sticky=tk.E, pady=2)
    h_input.grid(row=2, column=1, sticky=tk.E, pady=2)
    step_input.grid(row=3, column=1, sticky=tk.E, pady=2)
    n_input.grid(row=4, column=1, sticky=tk.E, pady=2)
    seed_input.grid(row=5, column=1, sticky=tk.E, pady=2)
    interp_input.grid(row=6, column=1, sticky=tk.E, pady=2)


    def send_settings():
        do('w', w_input.get())
        do('h', h_input.get())
        do('s', s_input.get())
        do('steps', step_input.get())
        do('n', n_input.get())
        do('seed', seed_input.get())
        do('interp', interp_input.get())
        do('p', p1_input.get())
        do('p2', p2_input.get())

    def command_block_go():
        user_input = command_input.get('1.0', tk.END).split('\n')
        user_input = [x for x in user_input if x != ""]
        command_input.delete('1.0', tk.END)
        command_input.insert(tk.INSERT, '\n'.join(user_input))
        send_settings()
        for command_string in user_input:
            doo(command_string)

    def just_go(n):
        send_settings()
        do('go', str(n))

    def just_go_ev(ev):
        just_go(1)
        
    p1_input.bind('<Return>', lambda e: just_go(1))
    p2_input.bind('<Return>', lambda e: just_go(1))
    command_input.bind('<Shift_L><Return>', lambda e: command_block_go())

    go_numbers = [1, 5, 10, 50, 100, 500]

    def stupid_bullshit(x):
        def dumb_nonsense():
            just_go(x)
        return dumb_nonsense

    go_buttons = [tk.Button(prompt_frame, text="Go (x{})".format(n), command=stupid_bullshit(n), font=("Consolas", 10)) for index, n in enumerate(go_numbers)]

    state.thumbnail_canvases = []

    def s_bs_2(i):
        def fu(event):
            if state.thumbnail_content[i] is None:
                return
            state.main_image = state.thumbnail_content[i]
            state.full_canvas.create_image(state.W // 2, state.H // 2, image=state.main_image.tk)
            state.full_canvas.update()
        return fu

    state.n_thumbnails = 30

    for row in range(state.n_thumbnails):
        canvas = tk.Canvas(right_panel, width=128, height=128)
        canvas.grid(row=row % 10, column=(row - (row % 10)) // 10, sticky = tk.W)
        canvas.bind("<Button-1>", s_bs_2(row))
        state.thumbnail_canvases += [canvas]
        canvas.configure(bg="black")
    
    state.thumbnail_content = [None] * state.n_thumbnails

    state.main_image = None
    state.main_tk_image = None

    state.full_canvas = tk.Canvas(main_frame, width=512, height=512)
    state.full_canvas.grid(row=4, column = 1, rowspan=4, columnspan=4, sticky=tk.W)
    state.full_canvas.configure(bg="black")

    for index, btn in enumerate(go_buttons):
        btn.grid(row=4, column=index, sticky = tk.W, pady=2)
        btn.configure(bg="black", fg="white")

    try:
        backend.start()

        main_window.mainloop()
    except (KeyboardInterrupt, SystemExit):
        sys.exit()
        do("exit", None)
        backend.join()

    do("exit", None)
    backend.join()

if __name__ == "__main__":
    main()
