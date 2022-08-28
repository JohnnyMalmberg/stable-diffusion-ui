import tkinter as tk
from tkinter import ttk
import argparse, os, sys, glob
from os.path import join
from PIL import ImageTk, Image
import shutil

parser = argparse.ArgumentParser()

parser.add_argument(
    "--folder",
    type=str,
    default=".",
    help="the folder to sift"
)

state = parser.parse_args()

state.image = {}

g_folder = join(state.folder, 'good')
o_folder = join(state.folder, 'ok')
t_folder = join(state.folder, 'trash')
os.makedirs(g_folder, exist_ok=True)
os.makedirs(o_folder, exist_ok=True)
os.makedirs(t_folder, exist_ok=True)

state.files = [f for f in os.listdir(state.folder) if os.path.isfile(join(state.folder, f))]
g_files = [f for f in os.listdir(g_folder) if os.path.isfile(join(g_folder, f))]
o_files = [f for f in os.listdir(o_folder) if os.path.isfile(join(o_folder, f))]
t_files = [f for f in os.listdir(t_folder) if os.path.isfile(join(t_folder, f))]

state.unsorted_count = len(state.files)
state.g_count = len(g_files)
state.o_count = len(o_files)
state.t_count = len(t_files)

state.files = [f for f in state.files if f.endswith('.png')]
g_files = [f for f in g_files if f.endswith('.png')]
o_files = [f for f in o_files if f.endswith('.png')]
t_files = [f for f in t_files if f.endswith('.png')]

state.paths = [join(state.folder, f) for f in state.files]
state.paths += [join(g_folder, f) for f in g_files]
state.paths += [join(o_folder, f) for f in o_files]
state.paths += [join(t_folder, f) for f in t_files]

state.files += g_files
state.files += o_files
state.files += t_files


def cleanup():
    if len(os.listdir(g_folder)) == 0:
        os.rmdir(g_folder)
    if len(os.listdir(o_folder)) == 0:
        os.rmdir(o_folder)
    if len(os.listdir(t_folder)) == 0:
        os.rmdir(t_folder)

if len(state.files) == 0:
    print('Nothing to sift!')
    cleanup()
    exit()



def sort(index, folder):
    shutil.move(state.paths[index], join(folder, state.files[index]))
    state.paths[index] = join(folder, state.files[index])

def sort_g(file):
    folder = state.paths[file]
    if '/good/' in folder:
        pass
    elif '/ok/' in folder:
        state.o_count -= 1
        state.g_count += 1
    elif '/trash/' in folder:
        state.t_count -= 1
        state.g_count += 1
    else:
        state.unsorted_count -= 1
        state.g_count += 1
    sort(file, g_folder)
def sort_o(file):
    folder = state.paths[file]
    if '/good/' in folder:
        state.g_count -= 1
        state.o_count += 1
    elif '/ok/' in folder:
        pass
    elif '/trash/' in folder:
        state.t_count -= 1
        state.o_count += 1
    else:
        state.unsorted_count -= 1
        state.o_count += 1
    sort(file, o_folder)
def sort_t(file):
    folder = state.paths[file]
    if '/good/' in folder:
        state.g_count -= 1
        state.t_count += 1
    elif '/ok/' in folder:
        state.o_count -= 1
        state.t_count += 1
    elif '/trash/' in folder:
        pass
    else:
        state.unsorted_count -= 1
        state.t_count += 1
    sort(file, t_folder)

window = tk.Tk()
window.configure(bg='black')

control_panel = tk.Frame(window)
control_panel.configure(bg='black')
control_panel.pack(side=tk.BOTTOM)

info_panel = tk.Frame(window)
info_panel.configure(bg='blue')
info_panel.pack(side=tk.BOTTOM)

state.current_index = 0

def mod_index(i):
    l = len(state.files)
    state.current_index = (state.current_index + i + l) % l
    update_canvas()

def inc_index():
    mod_index(1)

def dec_index():
    mod_index(-1)

def sort_current_g():
    sort_g(state.current_index)
    inc_index()
def sort_current_o(dec=False):
    sort_o(state.current_index)
    if dec:
        dec_index()
    else:
        inc_index()
def sort_current_t():
    sort_t(state.current_index)
    inc_index()

def is_sorted():
    current_path = state.paths[state.current_index]
    if '/good/' in current_path:
        return True
    if '/ok/' in current_path:
        return True
    if '/trash/' in current_path:
        return True
    return False

def right_arrow():
    if is_sorted():
        inc_index()
    else:
        sort_current_o()

def left_arrow():
    if is_sorted():
        dec_index()
    else:
        sort_current_o(dec=True)

def update_canvas_c(c, name, index):
    current_path = state.paths[index]
    state.image[name] = ImageTk.PhotoImage(Image.open(current_path))
    w = c.winfo_width()
    h = c.winfo_height()
    c.create_image(w // 2, h // 2, anchor=tk.CENTER, image=state.image[name])
    if '/good/' in current_path:
        c.configure(bg='green')
    elif '/trash/' in current_path:
        c.configure(bg='red')
    elif '/ok/' in current_path:
        c.configure(bg='gray')
    else:
        c.configure(bg='black')
    c.update()

def update_canvas():
    current_path = state.paths[state.current_index]
    update_canvas_c(canvas, "c", state.current_index)
    update_canvas_c(l_canvas, "l", (state.current_index if state.current_index > 0 else len(state.files)) - 1)
    update_canvas_c(r_canvas, "r", (state.current_index if state.current_index < len(state.files) - 1 else -1) + 1)
    label.config(text=current_path)
    g_count_label.config(text='[{} GOOD]'.format(state.g_count))
    o_count_label.config(text='[{} OK]'.format(state.o_count))
    t_count_label.config(text='[{} TRASH]'.format(state.t_count))
    u_count_label.config(text='[{} UNSEEN]'.format(state.unsorted_count))

consolas = ("Consolas", 16)

canvas_frame = tk.Frame(window)
canvas_frame.configure(bg="black")
canvas = tk.Canvas(canvas_frame, width=1111, height=1111)
l_canvas = tk.Canvas(canvas_frame, width=666, height=666)
r_canvas = tk.Canvas(canvas_frame, width=666, height=666)
label = tk.Label(window, text="")
g_count_label = tk.Label(info_panel, text="")
o_count_label = tk.Label(info_panel, text="")
t_count_label = tk.Label(info_panel, text="")
u_count_label = tk.Label(info_panel, text="")
label.configure(bg='black', fg='white', font=consolas)
g_count_label.configure(bg='black', fg='green', font=consolas)
o_count_label.configure(bg='black', fg='gray', font=consolas)
t_count_label.configure(bg='black', fg='red', font=consolas)
u_count_label.configure(bg='black', fg='white', font=consolas)

u_count_label.pack(side=tk.LEFT, anchor=tk.NW)
g_count_label.pack(side=tk.LEFT, anchor=tk.NW)
o_count_label.pack(side=tk.LEFT, anchor=tk.NW)
t_count_label.pack(side=tk.LEFT, anchor=tk.NW)

label.pack(side=tk.TOP)
canvas_frame.pack(side=tk.TOP)
canvas.grid(row=0, column=2, columnspan=4, rowspan=4)
l_canvas.grid(row=1, column=0, rowspan=2, columnspan=2)
r_canvas.grid(row=1, column=6, rowspan=2, columnspan=2)

inc_index()
dec_index()

g_button = tk.Button(control_panel, text="^ GOOD ^", command=sort_current_g, font=consolas)
o_button = tk.Button(control_panel, text="_ OK _", command=sort_current_o, font=consolas)
t_button = tk.Button(control_panel, text="v TRASH v", command=sort_current_t, font=consolas)
next_button = tk.Button(control_panel, text=" > NEXT > ", command=right_arrow, font=consolas)
prev_button = tk.Button(control_panel, text=" < PREV < ", command=left_arrow, font=consolas)

prev_button.configure(bg='gray', fg='white')
g_button.configure(bg='green', fg='white')
o_button.configure(bg='gray', fg='white')
t_button.configure(bg='red', fg='white')
next_button.configure(bg='gray', fg='white')

prev_button.pack(side=tk.LEFT, anchor=tk.NW)
g_button.pack(side=tk.LEFT, anchor=tk.NW)
o_button.pack(side=tk.LEFT, anchor=tk.NW)
t_button.pack(side=tk.LEFT, anchor=tk.NW)
next_button.pack(side=tk.LEFT, anchor=tk.NW)

window.bind('<Left>', lambda e: left_arrow())
window.bind('<Right>', lambda e: right_arrow())
window.bind('<Down>', lambda e: sort_current_t())
window.bind('<Up>', lambda e: sort_current_g())
window.bind('<space>', lambda e: sort_current_o())

window.mainloop()

cleanup()