
"""

  Vizualize the Non-Local Inds on a Video

  Incomplete; checkout "dev/flow_error_v2.py" for more details


"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def run(vid,inds,dpi=200,colors=None,s=None):

    # -- defaults --
    if colors is None:
        colors = [i/255. in range(T)]
    if s is None:
        s = 25

    # -- create images --
    annos = []
    for t in range(T):
        inds_t = get_inds_t(inds,t)
        X,Y = 0,0
        anno = annotate_img(img,X,Y,dpi,color,s)
        annos.
    pass

def get_inds_t(inds,t):
    ishape = inds.shape
    inds = inds.reshape(-1,3)
    args = th.where(inds[...,0] == t)

    inds_t = []
    for i in range(3):
        inds_t.append(th.gather(inds[...,i],args,1))
    inds_t = th.stack(inds_t,-1)
    return inds_t

def annotate_img(img,X,Y,dpi,color,s):
    fig,ax = im_plot(img,dpi)
    plot_grid(ax,X,Y,color,s)
    anno = get_plt_image(fig,ax)
    return anno

def im_plot(img,dpi):
    fig,ax=plt.subplots(1,1,
                        figsize=(3,3),
                        dpi=dpi,
                        tight_layout=True)
    ax.set_position([0, 0, 1, 1]) # Critical!
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    img = rearrange(img.cpu(),'c h w -> h w c')
    the_image = ax.imshow(
        img,zorder=0,alpha=1.0,
        origin="upper",
        interpolation="nearest",
    )
    return fig,ax

def get_plt_image(fig,ax):
    ax.axis("off")
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    #                     hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout(pad=0)
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    canvas = fig.canvas
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')\
            .reshape(int(height), int(width), 3)
    plt.close("all")
    img = th.from_numpy(img)
    img = rearrange(img,'h w c -> c h w')
    return img

    off_flow = get_offsets(cfg,nvid,vid,flows)

