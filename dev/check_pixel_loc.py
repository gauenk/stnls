import numpy as np
import numpy.random as npr
import stnls

def get_pixel_loc(qindex,stride0,nW0,nHW0,H,W):
    pix = [0,0,0]
    tmp = qindex
    pix[0] = tmp // nHW0
    tmp = (tmp - pix[0]*nHW0)
    nH_index = tmp // nW0
    pix[1] = (nH_index*stride0)
    tmp = tmp - nH_index*nW0;
    pix[2] = ((tmp % nW0) * stride0)
    return pix

def reflect(pix,lim):
    if pix < 0: return -pix
    elif pix >= lim: return 2*(lim-1)-pix
    else: return pix

def main():

    H,W = 64,64
    stride0 = 2
    ps = 3

    img = np.zeros((H,W))
    nH0 = (H-1)//stride0+1
    nW0 = (W-1)//stride0+1
    nHW0 = nH0*nW0
    Q = nHW0
    for qi in range(Q):
        pix = get_pixel_loc(qi,stride0,nW0,nHW0,H,W)
        pix_p = [0,0,0]
        for pi in range(ps):
            pix_p[1] = pix[1] + pi - ps//2
            pix_p[1] = reflect(pix_p[1],H)
            valid_h = 0 <= pix_p[1] and pix_p[1] < H

            for pj in range(ps):
                pix_p[2] = pix[2] + pj - ps//2
                pix_p[2] = reflect(pix_p[2],W)
                valid_w = 0 <= pix_p[2] and pix_p[2] < W
                if not(valid_h and valid_w): continue

                img[pix_p[1],pix_p[2]] += 1

    img = img.reshape(1,1,1,H,W)
    img /= img.max()
    stnls.utils.vid_io.save_video(img,"./output/dev/","check_pixel_loc")

if __name__ == "__main__":
    main()

