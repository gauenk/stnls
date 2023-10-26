"""

visualize stride1 shifting.

"""

import math
import numpy as np
import numpy.random as npr
from PIL import Image


def create_grid(loc, stride1, ws, H, W):
    hi,wi = loc
    wsHalf_h = (ws-1)//2
    wsHalf_w = (ws-1)//2
    # print(hi,hi-stride1*wsHalf_h,hi/stride1)

    # -- bound min --
    if ( hi - stride1 * wsHalf_h < 0):
        wsOff_h = hi/stride1
        # wsOff_h = (hi - stride1 * wsHalf_h)/stride1
    else:
        wsOff_h = wsHalf_h
    print(wi,stride1,wsHalf_w,stride1*wsHalf_w,wi - stride1 * wsHalf_w)
    if ( (wi - stride1 * wsHalf_w) < 0):
        # wsOff_w = (wi - stride1 * wsHalf_w)/stride1
        wsOff_w = wi/stride1
    else:
        wsOff_w = wsHalf_w
    # print("wsOff_h: ",wsOff_h)


    # -- bound max --
    hMax = hi + stride1 * ((ws-1) - wsOff_h)
    wMax = wi + stride1 * ((ws-1) - wsOff_w)
    # print("hMax: ",hMax)
    if (hMax > (H-1)):
        wsOff_h = -((H-1)-hi)/stride1 + (ws-1)# - ((ws-1)/2)
        # wsOff_h+=hMax-(H-1)
        # wsOff_h+=(hMax-min(hi+stride1*wsMax_h,H-1)-stride1)/stride1 + 1
    # print("wsOff_h: ",wsOff_h)
    if (wMax > (W-1)):
        wsOff_w = -((W-1)-wi)/stride1 + (ws-1)# - ((ws-1)/2)
        # wsOff_w+=(wi+wsMax_w-min(wi+stride1*wsMax_w,W-1)-stride1)/stride1 + 1

    # print("[float] offset: ",wsOff_h,wsOff_w)
    wsOff_h = round(wsOff_h)
    wsOff_w = round(wsOff_w)
    # wsOff_h = math.floor(wsOff_h)
    # wsOff_w = math.floor(wsOff_w)
    # wsOff_h = math.ceil(wsOff_h)
    # wsOff_w = math.ceil(wsOff_w)
    # print("[int] offset: ",wsOff_h,wsOff_w)
    print(loc,(wsOff_h,wsOff_w),
          (hi-stride1*wsOff_h,wi-stride1*wsOff_w),
          (hi+stride1*(ws-1-wsOff_h),wi+stride1*(ws-1-wsOff_w)))

    return wsOff_h,wsOff_w

def bounds(val,lim):
    if val < 0:
        return -val
    elif val > (lim-1):
        return 2*(lim-1) - val
    else:
        return val

def show_grid(img,loc,stride1,wsOff_h,wsOff_w,ws,H,W):
    check_eq = False
    h_c,w_c = loc
    for ws_i in range(ws):
        prop_h = h_c + stride1 * (ws_i - wsOff_h)
        # prop_h = h_c + stride1 * (ws_i - (ws-1)//2)
        # print(prop_h,wsOff_h,h_c,ws_i)
        for ws_j in range(ws):
            prop_w = w_c + stride1 * (ws_j - wsOff_w)
            # print(prop_h,prop_w)

            # -- check if search strikes center --
            if prop_h == h_c and prop_w == w_c:
                check_eq = True

            # -- interpolate --
            for ix in range(2):
                prop_hi = int(prop_h+ix)
                wH = max(0,1 - abs(prop_hi - prop_h))
                for jx in range(2):
                    prop_wi = int(prop_w+jx)
                    wW = max(0,1 - abs(prop_wi - prop_w))
                    prop_hi = bounds(prop_hi,H)
                    prop_wi = bounds(prop_wi,W)
                    # print(prop_hi,prop_wi)
                    weight = 1. if ws_i == ws//2 and ws_j == ws//2 else 0.5
                    img[prop_hi,prop_wi] += wH*wW*1.#weight

    print(check_eq)
    return check_eq

def main():


    ws = 3
    stride1 = 1.
    # wsMax = stride1*(ws-1-ws//2);
    H,W = 32,32
    img = np.zeros((H,W,3))
    # locs = [[0,0],[H//2,1],[H-1,1]]
    rH = npr.permutation(H)[:3]
    rW = npr.permutation(W)[:3]
    locs = np.c_[rH,rW]
    locs = [[H-1,1]]#,[H//2,1],[H-1,2]]
    # locs = [[1,0]]#,[H//2,1],[H-1,2]]
    # print(locs)

    # -- check --
    gH = np.arange(H)
    gW = np.arange(W)
    locs = np.stack(np.meshgrid(gH,gW)).T.reshape(-1,2)
    locs = [[31,0]]
    for loc in locs:
        c = 0
        wsOff_h,wsOff_w = create_grid(loc,stride1,ws,H,W)
        check = show_grid(img[:,:,c],loc,stride1,wsOff_h,wsOff_w,ws,H,W)
        assert check is True,loc

    # -- display --
    # for c,loc in enumerate(locs):
    #     wsOff_h,wsOff_w = create_grid(loc,stride1,ws,H,W)
    #     show_grid(img[:,:,c],loc,stride1,wsOff_h,wsOff_w,ws,H,W)

    # -- save --
    img /= img.max()
    img *= 255.
    img = img.astype(np.uint8)
    Image.fromarray(img).save("./output/stride1_grid.png")

if __name__ == "__main__":
    main()
