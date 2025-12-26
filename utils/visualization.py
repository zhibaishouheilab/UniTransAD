import torch
from einops import rearrange

def unpatchify(x, p, h, w):
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
    return imgs

def visualize_translation(img, y, mask, p, h, w):
    """
    Helper to visualize: Ground Truth, Masked Input, and Reconstructed Output.
    """
    y = unpatchify(y, p, h, w)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # Mask visualization
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, 8**2) 
    mask = unpatchify(mask, p, h, w)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', img).cpu()

    # Masked image
    im_masked = x * (1 - mask)

    # Pasted image (reconstruction filled in)
    im_paste = x * (1 - mask) + y * mask
    
    # Rearrange back to (B, C, H, W)
    im_paste = rearrange(im_paste, 'b w h c -> b c w h')
    im_masked = rearrange(im_masked, 'b w h c -> b c w h')
    y = rearrange(y, 'b w h c -> b c w h')

    return y[0], im_masked[0], im_paste[0]