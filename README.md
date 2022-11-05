# Stable Diffusion UI

A UI for stable diffusion, using tkinter. Uses xformers flash attention, and runs at half-precision. This is a hobby project, intended primarily for my own personal use, and I have not put any work into making sure it will run on others' machines or documenting the less-intuitive aspects of setting up and operating the program. There are plenty of other repos that are much more user-friendly, and I would recommend using one of those if you don't enjoy debugging python code & figuring out dependency issues for hours on end.

There are a couple little utility scripts that are much simpler to set up: 
promptcrafter.py is a REPL that can be used to create CLIP embeddings from prompts, and mix prompts in various ways (aesthetic gradients (there's a paper on arxiv somewhere explaining what that is) and a sort of "genetic" recombination of prompt embeddings). promptcrafter has no documentation, but if you understand python you can probably figure it out.
sift.py takes a --folder as a command line argument, and allows you to quickly sort every image in that folder into "good" "ok" and "trash" subfolders. The UI is (to me, at least) pretty intuitive and self-explanatory. It can also be operated with nothing but the arrow keys.

To set up:
You'll need a conda environment set up according to the directions for the original compvis version. Then you'll need to install a few more pip packages (TODO: write up a complete list). You'll need to install Katherine Crowson's k_diffusers or change the default sampler in sdui.py to "ddim" or "plms". I recommend k_diffusers; I've gotten the best results from the k_euler sampler.

xformers can be a bit tricky to set up, but you can swap out the attention.py for any other attention_*.py to avoid that headache.

The UI is designed for a pretty large monitor, with hard-coded values. There is a simple "scripting language" you can use to control the backend; it is not documented at all. You can also send commands to the backend through a unix socket. That's also not documented. Some settings have names that don't match what they're called in most/all other UIs, because I like these names better and nobody can stop me.

This has only been tested on linux, idk if it'll work at all on windows or a mac.

To launch:
python scripts/sdui.py 

