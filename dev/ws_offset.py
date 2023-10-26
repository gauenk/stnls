
import numpy as np
import math

def run(hi, wi, stride1, ws, H, W):

    # // -- init --
    wsHalf = (ws-1)//2
    wsOff_h = wsHalf;
    wsOff_w = wsHalf;

    # // -- bound min --
    if ( (hi - stride1 * wsHalf) < 0):
        wsOff_h = math.floor(hi/(1.*stride1))
    if ( (wi - stride1 * wsHalf) < 0):
        wsOff_w = math.floor(wi/(1.*stride1))

    # // -- bound max --
    hMax = hi + stride1 * ((ws-1) - wsOff_h);
    wMax = wi + stride1 * ((ws-1) - wsOff_w);
    if (hMax > (H-1)):
        wsOff_h = math.ceil((hi - (H-1))/(1.*stride1) + (ws-1));
    if (wMax > (W-1)):
        wsOff_w = math.ceil((wi - (W-1))/(1.*stride1) + (ws-1));

    # // -- rounding ensures reference patch is included in search space --
    wsOff_h = round(wsOff_h)
    wsOff_w = round(wsOff_w)

    return wsOff_h,wsOff_w


s1 = 1.
ws = 3
H = 10
W = 10
pairs = [
    [(1.1,0.8,s1,ws,H,W),(1,0)],
    [(0.8,0.8,s1,ws,H,W),(0,0)],
    [(0.2,0.2,s1,ws,H,W),(0,0)],
]
for (args,gt) in pairs:
    ans = run(*args)
    print(args,ans,(args[0]-ans[0],args[1]-ans[1]))
    print(ans,ans==gt)



