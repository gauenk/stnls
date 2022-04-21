

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange

# -- file io --
from PIL import Image
from pathlib import Path

MAX_FRAMES = 85

def load_burst(root,name,nframes=-1,ext="png"):

    # -- init path --
    path = Path(root) / name
    assert path.exists()

    # -- read images --
    burst = []
    nframes = nframes if nframes > 0 else MAX_FRAMES
    for t in range(nframes):
        fn = path / ("%05d.png" % t)
        if not fn.exists(): break
        img_t = Image.open(str(fn)).convert("RGB")
        img_t = np.array(img_t).transpose(2,0,1)
        burst.append(img_t)
    burst = np.stack(burst) * 1.
    return burst

def save_burst(burst,root,name):
    assert root.exists()
    nframes = burst.shape[0]
    for t in range(nframes):
        img_t = burst[t]
        path_t = root / ("%s_%05d.png" % (name,t))
        save_image(img_t,str(path_t))

def save_image(image,path):

    # -- to numpy --
    if th.is_tensor(image):
        image = image.cpu().numpy()

    # -- to uint8 --
    if image.max() < 100:
        image = image*255.
    image = np.clip(image.astype(np.uint8),0,255)

    # -- save --
    image = rearrange(image,'c h w -> h w c')
    img = Image.fromarray(image)
    img.save(path)

