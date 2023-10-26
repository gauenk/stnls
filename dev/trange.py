import numpy as np

def run(ti,wt,T):
    grid = np.arange(5)
    t_shift = min(0,ti-wt) + max(0,ti+wt-(T-1))
    t_min = max(ti - wt - t_shift,0);
    t_max = min(T-1,ti + wt - t_shift)
    # print(t_min,t_max)

    W_t = wt * 2 + 1
    t_inds = []
    for si in range(W_t):

    # int tj = ref[0] + si;
    # tj = (tj > t_max) ? t_max - si : tj;
    # bool isFwd = tj > ref[0];
    # // int t_flow = isFwd ? tj - 1 : (T-1) - tj;
    # int t_flow = isFwd ? tj - 1 : tj + 1;

        tj = ti + si
        tj = t_max - si if tj > t_max else tj
        isFwd = tj > ti
        t_end = t_max - tj + 1 if isFwd else tj-t_min+1
        # print(si,ti,tj,t_end)

        t_idx = ti+si
        if t_idx > t_max:
            t_idx = t_max - si

        t_inds.append(t_idx)
    return t_inds

pairs = [
    [(0,2,5),[0,1,2,3,4]],
    [(1,2,5),[1,2,3,4,0]],
    [(2,2,5),[2,3,4,1,0]],
    [(2,1,3),[2,1,0]],
    [(1,1,3),[1,2,0]],
    [(3,2,7),[3,4,5,2,1]],
    [(3,3,7),[3,4,5,6,2,1,0]],
]
for (args,gt) in pairs:
    ans = run(*args)
    print(ans,ans==gt)

