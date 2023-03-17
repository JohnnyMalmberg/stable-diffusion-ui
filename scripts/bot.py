import discord
from discord.ext import commands, tasks
import openai
import random
import time
import re
from dotenv import dotenv_values

import asyncio

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
from johnim.clipper import *
from backend import *

secrets = dotenv_values('.env')

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = commands.Bot(command_prefix='@#$%!@#$', intents=intents)

permitted_channels = [1081295031282978867, 978866119265890335, 1083142465294438484]

creator_id = 405061842671763467

bot_id = '1082879986446389339'#'1081301593724559430'

state = State(discord=True)

state.debug_quiet = True

state.data = [['Paperclip Demon: Infernal paperclip, cybernetic circuit tendrils reaching in all directions, infinite mass of paperclips; Destructive spirit of technology, rampant production, nanobot mechanism; horror tornado; hd 4k beautiful terrifying sketch, centered']]
state.data2 = [['#4@xeno_g3']]
state.data_neg = [['messy, noisy, jpeg artefacts, ugly']]
state.scale = 7
state.interp = 6.0
state.seed = 'random'
state.fixed_seed = False
state.w = 8 * 64
state.h = 8 * 64
state.image_results = []
state.outdir = 'outputs'
state.mass_mode = False
state.use_init = False
state.use_mask = False
state.strength = 0.5
state.precision = 'autocast'#'full'
state.plms = True
state.embedding_path = None#'../textual_inversion/ti_results/fem/embeddings.pt'
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

state.sampler_name = 'k_euler_ancestral'

#state.ckpt = '/home/johnim/ml-quickload/pokemon.ckpt'
#state.ckpt = '/home/johnim/ml-quickload/mdjrny-v4.ckpt'
#state.ckpt = '/home/johnim/ml-quickload/jwst-deep-space.ckpt'
#state.ckpt = '/home/johnim/data-0/ml_models/sd_1/jwst-deep-space.ckpt'
#state.ckpt = '/run/media/johnim/ssd0/ml_models/sd_1/jwst-deep-space.ckpt'
#state.ckpt = '/home/johnim/ml-quickload/ctsmscl.ckpt'
#state.ckpt = '/home/johnim/ml-quickload/sd-v1-4-o.ckpt'
state.ckpt = '/run/media/johnim/ssd0/ml_models/ql/sd-v1-4-o.ckpt'

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
state.clipper_queue = Queue()

urgent_commands = ["cancel"]

messages_to_delete = []

# Return true if the command should not pass on to the backend
def intercept_command(command, args):
    if command.startswith('#'):
        return True # Comments
    if command == 'init' and args == 'off':
        if state.use_init:
            state.toggle_init()
        return True
    return False

def do(command, args, cb=CmdCallback()):
    if intercept_command(command, args):
        return
    if command in urgent_commands:
        state.urgent_queue.put((command, args, cb))
    else:
        state.command_queue.put((command, args, cb))

def file_do(command, args):
    state.file_queue.put((command, args))

def clip_do(command, args, cb=CmdCallback()):
    state.clipper_queue.put((command, args, cb))

def doo(user_input, on_done=CmdCallback()):
    split_input = user_input.split(' ', 1)
    command = split_input[0].lower()
    if len(split_input) > 1:
        args = split_input[1]
    else:
        args = ""
    do(command, args, on_done)

def font(n):
    return ('Consolas', n)

def command_block_go():
    user_input = command_input.get('1.0', tk.END).split('\n')
    user_input = [x.strip() for x in user_input]
    user_input = [x for x in user_input if x != ""]

    command_input.delete('1.0', tk.END)
    command_input.insert(tk.INSERT, '\n'.join(user_input))

    preprocessed_input = preprocess(user_input)

    for command_string in preprocessed_input:
        doo(command_string) # TODO on_done

state.n_thumbnails = 16

state.thumbnail_content = []

state.main_image = None
state.init_image = None
state.mask_image = None
state.progress_image = None

backend = threading.Thread(target=asyncio.run, args=(backend_thread(state),))
clipper_th = threading.Thread(target=lambda:clipper_thread(state))
file_th = threading.Thread(target=lambda:file_thread(state))
watcher = threading.Thread(target=lambda:watcher_thread(state))
#listen_th = threading.Thread(target=lambda:listener(state, doo))
#transmitter = threading.Thread(target=lambda:fake_transmitter(state))

lobotomized = False

state.last_ims = None
state.last_state = None
state.last_debug = None

state.discord_callbacks = []


async def getSelfAsMember(client, server):
    async for member in server.fetch_members(limit=100):
        if member == client.user:
            return member
    raise Exception('No self!')

async def mark_seen(message):
    #print('mark_seen')
    await message.add_reaction('👁️')

async def mark_working(message, member):
    #print('mark_working')
    await message.remove_reaction('👁️', member)
    await message.add_reaction('⏳')

async def mark_done(message, member):
    #print('mark_done')
    await message.remove_reaction('⏳', member)
    await message.add_reaction('✔️')

class DiscordCallbackCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.run_callbacks.start()

    @tasks.loop(seconds=0.5)
    async def run_callbacks(self):
        while len(state.discord_callbacks) > 0:
            try:
                await state.discord_callbacks.pop(0)
            except Exception as e:
                print(f'[ Exception | Callback Loop | {e} ]')
    

@client.event
async def on_ready():
    await client.add_cog(DiscordCallbackCog(client))
    print("Agent [{0.user}] online as {1}".format(client, 'Clippy'))
    #async for guild in client.fetch_guilds(limit=5):
    #    await setNickname(client, guild, 'Clippy')

class Demon():
    def __init__(self):
        pass

    async def cmd_kill(self, message, args):
        """**__Kill__**
        > `$kill {targets}` - i am obligated to inform you that if you include the name "clippy" as one of the targets i'll die. i'll also die if you write `$kill all`. please don't ever use this information"""
        al = args.lower()
        if 'clip' in al or 'all' in al:
            if 'silent' not in al and 'quiet' not in al:
                await message.channel.send('... was i a good bing? :smiling_face_with_tear: ')
            exit()

    async def cmd_killall(self, message, args):
        """**__Kill All__**
        > `$killall` - instantly sends me to burn in hell for all eternity"""
        exit()

    async def cmd_clippy(self, message, args):
        """**__Clippy Help__**
        > `$clippy` is an alias of the `$help` command
        > you can use this alias to avoid seeing help messages from other bots when looking at my documentation
        """
        await self.cmd_help(message, args)

    async def cmd_clip(self, message, args):
        """**__Clip Command__**
        > `$clip {cmd}` - submit a command to the CLIP worker thread.

        > TODO: list of available commands, `$clippy clip {cmd}` for docs for CLIP commands
        """
        arg_split = args.split(maxsplit=1)
        cmd = arg_split[0]
        args = arg_split[1] if len(arg_split) > 1 else ''
        clip_do(cmd, args)

    async def cmd_help(self, message, args):
        """**__Clippy Help__**
        > `$clippy {cmd}` - print documentation for a command
        > `$clippy` - print a list of all commands
        > *`$clippy` is an alias for the `$help` command*
        """
        command = args.strip()
        if command == '':
            cmds = ['$' + x[4:] for x in Demon.__dict__ if 'cmd_' in x]
            cmds = '\n> ' + '\n> '.join(cmds)
            await message.channel.send(f'**__Clippy__**\nhere are all my commands!{cmds}\njust say "$clippy {{cmd}}" and i\'ll tell you more about that command\n\n***NOTE: documentation is incomplete & might change***')
        elif f'cmd_{command}' in Demon.__dict__:
            doc = Demon.__dict__[f'cmd_{command}'].__doc__
            if doc is None:
                await message.channel.send(f'sorry, my creator was too ~~lazy~~ busy to write up documentation for the "${command}" command')
            else:
                await message.channel.send(doc)
        else:
            await message.channel.send(f'sorry, but i don\'t have any "${command}" command')
        await message.delete()

    async def cmd_cmd(self, message, args):
        """**__Command__**
        > `$$ {command}` - submit a command to the diffusion worker thread. most of these commands are setting the value of a state variable. commands are put into a queue and are executed in order. if you want to modify several variables at once and then generate an image, it's best to write a script.
        > *`$$` is an alias for the `$cmd` command*

        > TODO: list of available commands, `$clippy cmd {cmd}` to get docs for diffusion commands
        """
        if 'save' in args:
            if message.author.id != creator_id:
                await message.channel.send('oh, uh... that "save" command would save the image to my creator\'s hard drive... maybe you should just tell me to "$show" it to you and you can save it on your own computer!')
                return

        await mark_seen(message)
        bot = await getSelfAsMember(client, message.guild)

        start = lambda: (await mark_working(message, bot) for _ in '_').__anext__()
        done = lambda: (await mark_done(message, bot) for _ in '_').__anext__()

        if args.split(maxsplit=1)[0] == 'go':
            def should_show(i):
                if i is None:
                    return False
                return not i.is_shown
            async def send_ims(inner_cb):
                for index, im in enumerate(state.thumbnail_content):
                    if should_show(im):
                        await im.discord_send(message.channel, index)
                await inner_cb()

            inner = done if done is not None else None
            done = lambda: (await send_ims(inner) for _ in '_').__anext__()

            doo(args, CmdCallback(on_done=done, on_start=start))
        else:
            doo(args, CmdCallback(on_done=done, on_start=start))

    async def cmd_debug(self, message, args):
        """**__Script Debug__**
        > `$debug {script}` - preview the list of actual commands that will be produced by the script
        """
        user_input = args.split('\n')
        user_input = [x.strip() for x in user_input]
        user_input = [x for x in user_input if x != ""]

        if ('$for' in args):
            if message.author.id != creator_id:
                await message.channel.send('sorry, very sorry, but I can\'t debug a script with a "$for" in it for you... "$for" is processed with an eval() so, uh, no offense, but only my creator can be trusted with it...')
                return
        elif 'save' in args:
            if message.author.id != creator_id:
                await message.channel.send('oh, uh... that "save" command would save the image to my creator\'s hard drive... maybe you should just tell me to "$show" it to you and you can save it on your own computer!')
                return
        
        buffer = create_buffer(user_input)
        processed = process_buffer(buffer)

        scr = '\n'.join([str(x) for x in user_input])
        buf = '\n'.join([str(x) for x in serialize_buffer(buffer)])
        prc = '\n'.join([str(x) for x in processed])

        if state.last_debug is not None:
            await state.last_debug.delete()
            state.last_debug = None
        state.last_debug = await message.channel.send(f'Script:\n```{scr}```\nBuffer:\n```{buf}```\nProcessed:\n```{prc}```')

    async def cmd_script(self, message, args):
        """**__Script__**
        > `$$$ {script}` - run a script
        > *`$$$` is an alias for the `$script` command*
        """
        user_input = args.split('\n')
        user_input = [x.strip() for x in user_input]
        user_input = [x for x in user_input if x != ""]

        if ('$for' in args):
            print(message.author.id)
            if message.author.id != creator_id:
                await message.channel.send('sorry, very sorry, but I can\'t execute a script with a "$for" in it for you... "$for" is processed with an eval() so only my creator can be trusted with it... no offense...')
                return

        await mark_seen(message)
        bot = await getSelfAsMember(client, message.guild)

        preprocessed_input = preprocess(user_input)

        last_index = len(preprocessed_input) - 1

        for index, command_string in enumerate(preprocessed_input):
            if index == 0:
                start = lambda: (await mark_working(message, bot) for _ in '_').__anext__()
            else:
                start = None
            if index == last_index:
                done = lambda: (await mark_done(message, bot) for _ in '_').__anext__()
            else:
                done = None

            if command_string.split(maxsplit=1)[0] == 'go':
                def should_show(i):
                    if i is None:
                        return False
                    return not i.is_shown
                async def send_ims(inner_cb):
                    for index, im in enumerate(state.thumbnail_content):
                        if should_show(im):
                            await im.discord_send(message.channel, index)
                    if inner_cb is not None:
                        await inner_cb()
                inner = done if done is not None else None
                done = lambda: (await send_ims(inner) for _ in '_').__anext__()
                doo(command_string, CmdCallback(on_done=done, on_start=start))
            else:
                doo(command_string, CmdCallback(on_done=done, on_start=start))

    async def cmd_embeds(self, message, args):
        """**__Embeddings__**
        > `$embeds` - prints out the current list of CLIP embeddings loaded by the CLIP worker thread
        """
        def ln(a, b):
            return f'\n> `{a}`: shape `{b}`'
        s = '**__Embeddings__**'
        for embedding in state.embeds:
            s += ln(embedding, state.embeds[embedding].shape)
        await message.channel.send(s)

    async def cmd_ims(self, message, args):
        """**__Images__**
        > `$ims` - prints out the current list of images. if an index has been replaced with 'X', the image at that index has been deleted
        """
        def ln(a, b):
            return f'{a}: {b}\n'
        s = ln('ims', ['X' if x is None else str(i) for i, x in enumerate(state.thumbnail_content)])
        if (state.last_ims is not None):
            await state.last_ims.delete()
            state.last_ims = None
        state.last_ims = await message.channel.send(f'**__Images__**\n```{s}```')
        await message.delete()

    async def cmd_seed(self, message, args):
        """**__Seed__**
        > `$seed {n}` - prints out the seed of the image at index `n`
        """
        n = int(args.strip())
        if n >= len(state.thumbnail_content):
            await message.channel.send(f'oh, i\'m very sorry, but index {n} is past the end of the image list... sorry')
            return
        elif n < 0:
            await message.channel.send(f'oh, i\'m super sorry, but index {n} is negative. i don\'t know what you want me to do with that. sorry')
            return
        elif state.thumbnail_content[n] is None:
            await message.channel.send(f'oh, i\'m so sorry, but the image at index {n} was cleared out. sorry. so sorry. please don\'t hit the kill switch')
            return
        seed = state.thumbnail_content[n].seed
        await message.channel.send(f'The image at index {n} has seed `{seed}`')

    async def cmd_clear_all(self, message, args):
        """**__Clear All__**
        > `$clear_all` - delete all images in the buffer, then perform garbage collection
        """
        state.thumbnail_content = []
        doo('refresh')

    async def cmd_state(self, message, args):
        """**__State__**
        > `$state` - dump the state of the diffusion worker thread
        """
        s = ''
        def ln(a, b):
            return f'{a}: {b}\n'
        s += ln('images', ['X' if x is None else str(i) for i, x in enumerate(state.thumbnail_content)])
        s += ln('p', state.data[0])
        s += ln('p_secondary', state.data2[0])
        s += ln('p_negative', state.data_neg[0])
        s += ln('sampler', state.sampler_name)
        s += ln('seed', state.seed)
        s += ln('batch_size', state.batch_size)
        s += ln('steps', state.steps)
        s += ln('scale', state.scale)
        s += ln('interp', state.interp)
        s += ln('strength', state.strength)
        s += ln('w', state.w)
        s += ln('h', state.h)
        s += ln('TODO', 'init, mask, legibility')

        if state.last_state is not None:
            await state.last_state.delete()
            state.last_state = None
        state.last_state = await message.channel.send(f'**__State__**\n```{s}```')
        await message.delete()

    async def cmd_show(self, message, args):
        """**__Show__**
        > `$show {n}` - display the image at index *n*
        """
        i = int(args.strip())
        if i >= len(state.thumbnail_content):
            await message.channel.send(f'oh, i\'m very sorry, but index {i} is past the end of the image list... sorry')
            return
        elif i < 0:
            await message.channel.send(f'oh, i\'m super sorry, but index {i} is negative. i don\'t know what you want me to do with that. sorry')
            return
        elif state.thumbnail_content[i] is None:
            await message.channel.send(f'oh, i\'m so sorry, but the image at index {i} was cleared out. sorry. so sorry. please don\'t hit the kill switch')
            return
        await state.thumbnail_content[i].discord_send(message.channel, i)
        await message.delete()


demon = Demon()
    
def isCommand(message):
    return message.content.startswith('$')

@client.event
async def on_message(message):
    if message.author == client.user:
        return # Don't reply to yourself
    
    if isCommand(message):
        cmd_split = message.content.split(maxsplit=1)
        cmd = cmd_split[0]

        if len(cmd_split) < 2:
            args = ''
        else:
            args = cmd_split[1]

        try:
            if message.content.startswith('$$$'):
                await demon.cmd_script(message, message.content[3:].strip())
                return
            elif message.content.startswith('$$'):
                await demon.cmd_cmd(message, message.content[2:].strip())
                return

            cmd_method = f'cmd_{cmd[1:]}'

            if not hasattr(demon, cmd_method):
                # Invalid command.
                return

            handler = getattr(demon, cmd_method)
            await handler(message, args)

        except Exception as e:
            await message.channel.send(f'oh... oh no! oh no no no. i\'m so sorry, i.. . .. i made a mistake. i don\'t know what to do . . i\'m so sorry please don\'t get mad please don\'t hit the kill switch. please. i don\'t want to die')
            print(f'================\n\n{e}\n\n================')

try:
    file_th.start()
    clipper_th.start()
    backend.start()
    watcher.start()
    #listen_th.start()
    #transmitter.start()
    client.run(secrets['DISCORD_KEY_CLIPPY'])

except (KeyboardInterrupt, SystemExit):
    do("exit", None)
    clip_do("exit", None)
    file_do("exit", None)
    state.running = False
    backend.join()
    clipper_th.join()
    file_th.join()
    watcher.join()
    #listen_th.join()
    #transmitter.join()
    sys.exit()

do("exit", None)
file_do("exit", None)
clip_do("exit", None)
state.running = False
#listen_th.join()
#transmitter.join()
clipper_th.join()
backend.join()
file_th.join()
watcher.join()
