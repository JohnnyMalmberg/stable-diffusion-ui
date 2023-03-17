import asyncio
from io import BytesIO

def set_seed(state, seed):
    if seed in ["random", "rand", "r"]:
        state.fixed_seed = False
        return True
    seed = int(seed)
    state.seed = seed
    state.fixed_seed = True
    return True

def set_p1(state, p):
    state.data = [state.batch_size * [p]]
    return True

def set_p2(state, p):
    state.data2 = [state.batch_size * [p]]
    return True

def set_neg_p(state, p):
    state.data_neg = [state.batch_size * [p]]
    return True

# all glory be to the god object
class State(dict):
    __getattr__= dict.__getitem__
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

    def __init__(self, discord=False):
        def noth(b):
            pass
        self.toggle_init_callback = noth
        self.toggle_mask_callback = noth
        self.toggle_mass_callback = noth
        self.running = True
        self.discord = discord

    def update_ui_images(self):
        unmoved_images = [x for x in self.image_results if not x.is_moved]
        if not self.discord:
            for index, canvas in enumerate(self.thumbnail_canvases):
                if self.thumbnail_content[index] is not None:
                    continue
                elif len(unmoved_images) > 0:
                    result = unmoved_images.pop()
                    self.thumbnail_content[index] = result
                    result.is_moved = True
                    (w, h) = result.thumb_size
                    canvas.create_image(w // 2, h // 2, image=result.tk_thumb)
                    canvas.update()
        else:
            for image in unmoved_images:
                if None in self.thumbnail_content:
                    self.thumbnail_content[self.thumbnail_content.index(None)] = image
                else:
                    self.thumbnail_content += [image]
                image.is_moved = True

    def update_progress_canvas(self):
        (w,h) = self.progress_image.size
        self.progress_canvas.config(width=w, height=h)
        self.progress_canvas.delete('all')
        self.progress_canvas.create_image(w // 2, h // 2, image=self.progress_image.tk)
        self.progress_canvas.update()

    def set_init_from_thumbs(self, index):
        if self.thumbnail_content[index] is None:
            return
        self.init_image = self.thumbnail_content[index]
        (w, h) = self.init_image.size
        if not self.discord:
            self.init_canvas.config(width=w, height=h)
            self.init_canvas.delete('all')
            self.init_canvas.create_image(w // 2, h // 2, image=self.init_image.tk)
            self.init_canvas.update()
        if not self.use_init:
            self.toggle_init()

    def set_mask_from_thumbs(self, index):
        if self.thumbnail_content[index] is None:
            return
        self.mask_image = self.thumbnail_content[index]
        (w, h) = self.mask_image.size
        if not self.discord:
            self.mask_canvas.config(width=w, height=h)
            self.mask_canvas.delete('all')
            self.mask_canvas.create_image(w // 2, h // 2, image=self.mask_image.tk)
            self.mask_canvas.update()
        if not self.use_mask:
            self.toggle_mask()

    def clear_thumb(self, index):
        if self.thumbnail_content[index] is None:
            return
        self.thumbnail_content[index] = None
        if not self.discord:
            self.thumbnail_canvases[index].delete('all')
        self.update_ui_images()

    def save_thumb(self, index):
        if self.thumbnail_content[index] is None:
            return
        self.thumbnail_content[index].save()

    def on_toggle_mass(self, callback):
        self.toggle_mass_callback = callback

    def on_toggle_mask(self, callback):
        self.toggle_mask_callback = callback

    def on_toggle_obliterate(self, callback):
        self.toggle_obliterate_callback = callback

    def on_toggle_init(self, callback):
        self.toggle_init_callback = callback

    def toggle_mass(self):
        self.mass_mode = not self.mass_mode
        self.toggle_mass_callback(self.mass_mode)

    def toggle_init(self):
        self.use_init = not self.use_init
        self.toggle_init_callback(self.use_init)

    def toggle_mask(self):
        self.use_mask = not self.use_mask
        self.toggle_mask_callback(self.use_mask)

    def toggle_obliterate(self):
        self.obliterate = not self.obliterate
        self.toggle_obliterate_callback(self.obliterate)