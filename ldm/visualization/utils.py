
import torch

from jaxtyping import Float
from torch import Tensor
from flow_vis_torch import flow_to_color
from einops import rearrange, repeat

from ldm.modules.flowmap.visualization.depth import color_map_depth
from ldm.modules.flowmap.visualization.color import apply_color_map_to_image
from ldm.visualization import filter_depth
from ldm.misc.modalities import Modality

def prepare_images(
    images: Float[Tensor, "sample frame channel=3 height width"] | Float[Tensor, "sample frame height width"],
    max_images: int | None = None,
    **kwargs
) -> Float[Tensor, "(sample frame) channel=3 height width"] | Float[Tensor, "(sample frame) height width"]:
    '''Subsample and copy tensors to cpu.
    '''
    if max_images is None:
        max_images = images.shape[0]
    N = min(images.shape[0], max_images)
    images = images[:N].clone()
    images = rearrange(images, 'b f ... -> (b f) ...')
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    return images


def prepare_log_rgb(
    x: Float[Tensor, "sample frame 3 height width"],
    clamp: bool = True,
    **kwargs
) -> Float[Tensor, "* 3 height width"]:
    
    images = prepare_images(x, **kwargs)

    # Make RGB figures.
    if clamp:
        images = torch.clamp(images, -1., 1.)
    images = (images + 1.0) / 2.0 #  [-1,1] -> [0,1].

    return images

def prepare_log_flow(
    x: Float[Tensor, "sample frame channel height width"],
    **kwargs
) -> Float[Tensor, "(sample frame) 3 height width"]:

    images = prepare_images(x, **kwargs)
    images = flow_to_color(images[:,:2,:,:]) / 255

    return images

def prepare_log_depth(
    x: Float[Tensor, "sample frame channel height width"],
    sample: bool = False,
    **kwargs
) -> Float[Tensor, "(sample frame) 3 height width"]:
    if len(x.size()) == 5 and x.size(2) == 3:
            x = x.mean(2)
    if sample: #exponential mapping for depth samples
        x = (x / 1000).exp() + 0.01
    images = prepare_images(x, **kwargs)
    
    # Apply colorization.
    images, _ = filter_depth(images)
    images = color_map_depth(images)
    
    return images

def prepare_log_correspondence_weight(
    x: Float[Tensor, "sample pair height width"],
    **kwargs
) -> Float[Tensor, "(sample frame) 3 height width"]:
    images = prepare_images(x, **kwargs)
    images = apply_color_map_to_image(images, "gray")

    return images

def prepare_log_pointmap(
    x: Float[Tensor, "sample frame channel height width"],
    pointmap_mapping_func,
    sample: bool = False,
    **kwargs
) -> Float[Tensor, "(sample frame) 3 height width"]:
    assert x.size(2) == 3

    if sample:
        pointmaps = pointmap_mapping_func(x)
    else:
        pointmaps = x
        
    depths = pointmaps[:,:,2]
    images = prepare_images(depths)
    
    # Apply colorization.
    images, _ = filter_depth(images)
    images = color_map_depth(images)
    
    return images

def prepare_visualization(
    x_m: Float[Tensor, "sample ... height width"],
    modality: Modality,
    **kwargs
    # sample: bool = False,
) -> Float[Tensor, "* 3 height width"]:
    if modality.modality == 'rgb':
        images = prepare_log_rgb(x_m, **kwargs)
    elif modality.modality == 'depth':
        images = prepare_log_depth(x_m, **kwargs)
    elif modality.modality == 'pointmap':
        images = prepare_log_pointmap(x_m, **kwargs)
    elif modality.modality == 'flow':
        images = prepare_log_flow(x_m, **kwargs)
    elif modality.modality == 'weight':
        images = prepare_log_correspondence_weight(x_m, **kwargs)
    else:
        raise Exception(f"Non-recognised modality type {modality.modality} - should be 'depth', 'rgb', 'flow', 'pointmap' or 'weight'")

    return images