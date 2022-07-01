import torch as th

def comp_pads(vshape, ksizes, strides, rates):
    if isinstance(ksizes,int): ksizes = [ksizes,ksizes]
    if isinstance(strides,int): strides = [strides,strides]
    if isinstance(rates,int): rates = [rates,rates]

    t,c,h,w = vshape
    t,c,rows,cols = vshape
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    offset_h,offset_w = padding_top,padding_left
    hp,wp = padding_rows + h,padding_cols + w
    return offset_h,offset_w,hp,wp

def same_padding(images, ksizes, strides, rates, mode="zero"):
    if isinstance(ksizes,int): ksizes = [ksizes,ksizes]
    if isinstance(strides,int): strides = [strides,strides]
    if isinstance(rates,int): rates = [rates,rates]

    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    if mode == "zero":
        images = th.nn.ZeroPad2d(paddings)(images)
    elif mode == "reflect":
        images = th.nn.ReflectionPad2d(paddings)(images)
    else:
        raise ValueError(f"Uknown mode [{mode}]")

    return images, paddings
