from PIL import Image, ImageTk, ImageEnhance
from torch import clamp, from_numpy
import numpy as np
from einops import rearrange
import os
import cv2
from skimage import exposure

def load_torch_image(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    return to_torch_image(image)

def to_torch_image(pil_image):
    image = np.array(pil_image).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = from_numpy(image)
    return 2.*image - 1.

def resize_torch_image(torch_image, w, h):
    pil_image = to_pil_image(torch_image)
    pil_resized = pil_image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(pil_resized).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = from_numpy(image)
    return 2.*image - 1.

def to_pil_image(torch_image):
    clamped_image = clamp((torch_image + 1.0) / 2.0, min=0.0, max=1.0)
    np_image = 255. * rearrange(clamped_image.cpu().numpy(), 'c h w -> h w c')
    pil_image = Image.fromarray(np_image.astype(np.uint8))
    return pil_image

def thumb_size(w, h):
    (b, s) = (h,w) if w < h else (w,h)
    b_t = 256 if s >= 256 else s
    s_t = (b_t * s) // b
    return (s_t, b_t) if w < h else (b_t, s_t)

def color_match(state, image_index, color_target_index):
    pil_image = state.thumbnail_content[image_index].pil_image
    color_target = cv2.cvtColor(np.asarray(state.thumbnail_content[color_target_index].pil_image), cv2.COLOR_RGB2LAB)
    output_image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2LAB), color_target, channel_axis=2), cv2.COLOR_LAB2RGB).astype('uint8')).convert('RGB')
    output_image = to_torch_image(output_image)
    return ImageResult(output_image, state)    

def sharpen(state, image_index, amount):
    pil_image = state.thumbnail_content[image_index].pil_image
    sharpen = ImageEnhance.Sharpness(pil_image)
    image = sharpen.enhance(amount)
    return ImageResult(to_torch_image(image), state)

class ImageResult:
    def __init__(self, torch_image, state, loaded=False, is_preview=False):
        self.state = state
        self.torch_image = torch_image
        self.pil_image = to_pil_image(torch_image)
        self.seed = state.seed
        self.tk = ImageTk.PhotoImage(self.pil_image)
        self.size = (self.pil_image.width, self.pil_image.height)
        self.thumb_size = thumb_size(self.pil_image.width, self.pil_image.height)
        self.thumb = self.pil_image.resize(self.thumb_size)
        self.tk_thumb = ImageTk.PhotoImage(self.thumb)
        self.is_shown = False
        self.was_loaded = loaded
        if state.mass_mode and not is_preview and not loaded:
            self.save()

    def save(self):
        if not self.was_loaded:
            name = f"{self.seed}_{self.state.base_count:03}.png"
        else:
            name = f"loaded_{self.state.base_count:03}.png"
        self.pil_image.save(os.path.join(self.state.outdir, name))
        self.state.base_count += 1

    def gimp(self):
        self.pil_image.save("/tmp/gimp_transfer.png")
        os.system("gimp /tmp/gimp_transfer.png &> /dev/null &")

