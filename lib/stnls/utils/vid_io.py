# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- file io --
from PIL import Image
from pathlib import Path

def save_burst(burst,root,name):
    return save_video(burst,root,name)

def save_video(vid,root,name,itype="png"):
    if vid.ndim == 4:
        return _save_video(vid,root,name,itype)
    elif vid.ndim == 5 and vid.shape[0] == 1:
        return _save_video(vid[0],root,name,itype)
    elif vid.ndim == 5 and vid.shape[0] > 1:
        fns = []
        B = vid.shape[0]
        for b in range(B):
            fns_b = _save_video(vid[b],root,"%s_%02d" % (name,b),itype)
            fns.extend(fns_b)
        return fns
    else:
        raise ValueError("Uknown number of dims [%d]" % vid.ndim)

def _save_video(vid,root,name,itype="png"):
    # -- path --
    root = Path(str(root))
    if not root.exists():
        print(f"Making dir for save_vid [{str(root)}]")
        root.mkdir(parents=True)
    assert root.exists()

    # -- save --
    save_fns = []
    nframes = vid.shape[0]
    for t in range(nframes):
        img_t = vid[t]
        path_t = root / ("%s_%05d" % (name,t))
        save_image(img_t,str(path_t),itype)
        save_fns.append(str(path_t))
    return save_fns

def save_image(image,base,itype):
    if itype == "png":
        save_image_png(image,base)
    elif itype == "np":
        save_image_np(image,base)
    else:
        raise ValueError("Uknown save_image type [%s]" % itype)

def save_image_np(image,base):

    # -- path --
    path = "%s.npy" % str(base)

    # -- to numpy --
    if th.is_tensor(image):
        image = image.detach().cpu().numpy()

    # -- save --
    np.save(path,image)

def save_image_png(image,base):

    # -- path --
    path = "%s.png" % str(base)

    # -- to numpy --
    if th.is_tensor(image):
        image = image.detach().cpu().numpy()

    # -- rescale --
    if image.max() > 300: # probably from a fold
        image /= image.max()

    # -- to uint8 --
    if image.max() < 100:
        image = image*255.
    image = np.clip(image,0,255).astype(np.uint8)

    # -- remove single color --
    image = rearrange(image,'c h w -> h w c')
    image = image.squeeze()

    # -- save --
    img = Image.fromarray(image)
    img.save(path)


def read_video(path):
    fns = list(Path(path).iterdir())
    ints = [int(fn.stem) for fn in fns]
    fns = [x for _, x in sorted(zip(ints, fns))]
    return read_files(fns)

def read_files(fns):
    vid = []
    for fn in fns:
        fn = mangle_fn(fn)
        img = np.array(Image.open(fn))
        img = rearrange(img,'h w c -> c h w')
        img = th.from_numpy(img)
        vid.append(img)
    vid = th.stack(vid)
    return vid

def mangle_fn(fn):
    fn = Path(fn)
    suffix_l = [".png",".jpeg",".jpg"]
    if fn.suffix in suffix_l:
        return str(fn)
    else:
        for suffix in suffix_l:
            fn_s = str(fn) + "%s" % suffix
            fn_s = Path(fn_s)
            if fn_s.exists():
                return str(fn_s)
    raise ValueError("Unknown file %s" % fn)

