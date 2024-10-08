import importlib
import torch
import numpy as np

from typing import Any, Literal
from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
from jaxtyping import Float
from torch import optim, Tensor
from pytorch_lightning.utilities import rank_zero_only
from torch.nn.functional import softplus

@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AdamWwithEMAandWings(optim.Optimizer):
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
                 ema_power=1., param_names=()):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
                        ema_power=ema_power, param_names=param_names)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            ema_decay = group['ema_decay']
            ema_power = group['ema_power']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter values
                    state['param_exp_avg'] = p.detach().float().clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params_with_grad.append(state['param_exp_avg'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            optim._functional.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False)

            cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss
    

def modify_conv_weights(
        source_weights: Float[Tensor, "in_channel out_channel kernel_size kernel_size"],
        scale: float = 1e-8,
        n: int = 2,
        dim: int = 1,
        copy_weights: bool = False
    ) -> Float[Tensor, "new_in_channel new_out_channel kernel_size kernel_size"]:
    """
    Extends input or output convolution weights by multiplicating number of channels.
    Inputs:
    - source_weights: torch.tensor(C_out, C_in, k, k), tensor of conv kernel, where C_out number of output channels, C_in number of input channels, k kernel size
    - scale: float, scale factor of the initialization. For random initialization ~0, scale=1e-8. For copy initialization as input weights, scale=1/n, for copy initialization as output, scale=1
    - n: int, multiplication factor for the number of channels
    - dim: int, axis along which to repeat/randomly initialize source weights
    - copy_weights: bool, initialization method for new weights, random by default, copy of w if True
    """
    # Initialize new weights with original weights.
    new_weights = source_weights.clone()

    # Initialize new additional weights.
    if copy_weights:
        # If copy original weights, scale to match original activations.
        new_weights = scale * new_weights
        extra_weights = scale * source_weights.clone()
    else:
        # Random initialization of additional weights.
        extra_weights = scale * torch.randn_like(source_weights)
    
    # Adds new weights to extend channels dimension.
    for i in range(n-1):
        new_weights = torch.cat((new_weights, extra_weights.clone()), dim=dim)
    return new_weights




############### Nested sets utils ###############


def get_value(dictionnary, keys: list[str] | str) -> Any:
    '''Get value from a nested dict for a given key sequence.
    '''
    if isinstance(keys, str):
        keys = [keys]
    
    if len(keys) > 0:
        tmp_dict = dictionnary
        for key in keys:
            try:
                tmp_dict = tmp_dict[key]
            except KeyError:
                return None
        return tmp_dict
    else:
        return None
    

def default_set(
    dictionnary: dict,
    keys: list[str] | str,
    value: Any
) -> None:
    '''Set value in a nested dict for a given key sequence.
    '''
    # Replace value in dict.
    if get_value(dictionnary,keys) is not None:
        if len(keys) > 1:
            tmp_dict = get_value(dictionnary, keys[:-1])
            tmp_dict[keys[-1]] = value
        else:
            dictionnary[keys[-1]] = value
        return None
    # Put value in dict.
    elif value is not None and len(keys) > 0:
        tmp_dict = dictionnary

        for i, key in enumerate(keys):
            try:
                tmp_dict = tmp_dict[key]
            except KeyError:
                # Create dict if not final key.
                if i < len(keys) -1:
                    tmp_dict[key] = dict()
                    tmp_dict = tmp_dict[key]
                # Initiate leaf list and exits.
                else:
                    tmp_dict[key] = value
                return None

def set_nested(
    dictionnary: dict,
    keys: list[str] | str,
    value: Any,
    operation: Literal["append", "extend", "set"] = "append" #defines operation to perform, either append to list, extend a list, or replace/set the value.
) -> None:
    '''Set values in a nested dict given a key sequence.
    '''
    if isinstance(keys, str):
        keys = [keys]

    if operation == "set":
        return default_set(dictionnary=dictionnary, keys=keys, value=value)
    
    if value is not None and len(keys) > 0:
        tmp_dict = dictionnary

        for i, key in enumerate(keys):
            try:
                tmp_dict = tmp_dict[key]
            except KeyError:
                # Create dict if not final key.
                if i < len(keys) -1:
                    tmp_dict[key] = dict()
                    tmp_dict = tmp_dict[key]
                # Instantiate leaf list and exits.
                else:
                    if operation == "append":
                        tmp_dict[key] = [value]
                    elif operation == "extend":
                        assert isinstance(value, list)
                        tmp_dict[key] = value
                    return None

        # Extend leaf list.
        if operation == "append":
            assert isinstance(tmp_dict, list)
            tmp_dict.append(value)
        elif operation == "extend":
            assert isinstance(value, list) and isinstance(tmp_dict, list)
            tmp_dict.extend(value)
        else:
            raise Exception('Operation not recognized, should be "append", "extend" or "set".')
        return None
    



############### 3D Mappings ################


def flowmap_dust3r_param(
    x: Float[Tensor, "sample frame 3 height width"],
    epsilon: float = 1e-5,
    infinity: float = 1e8,
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    # Flowmap exponential mapping.
    x[:,:,2:3,:,:] = (x[:,:,2:3,:,:] / 1000).exp() + 0.01

    # DUST3R exponential mapping.
    d = x.norm(dim=2, keepdim=True)
    pointmaps = x / (d + epsilon)
    pointmaps = pointmaps.nan_to_num(posinf=infinity, neginf=-infinity)
    pointmaps = pointmaps * torch.expm1(d)

    return pointmaps


def dust3r_flowmap_param(
    x: Float[Tensor, "sample frame 3 height width"],
    epsilon: float = 1e-5,
    infinity: float = 1e8,
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    # DUST3R exponential mapping.
    d = x.norm(dim=2, keepdim=True)
    pointmaps = x / (d + epsilon)
    pointmaps = pointmaps.nan_to_num(posinf=infinity, neginf=-infinity)
    pointmaps = pointmaps * torch.expm1(d)

    # Flowmap exponential mapping.
    pointmaps[:,:,2:3,:,:] = (pointmaps[:,:,2:3,:,:] / 1000).exp() + 0.01

    return pointmaps


def soft_dust3r_flowmap_param(
    x: Float[Tensor, "sample frame 3 height width"],
    epsilon: float = 1e-5,
    infinity: float = 1e8,
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    # DUST3R square mapping.
    d = x.norm(dim=2, keepdim=True)
    pointmaps = x / (d + epsilon)
    pointmaps = pointmaps.nan_to_num(posinf=infinity, neginf=-infinity)
    pointmaps = pointmaps * d.square()

    # Flowmap exponential mapping.
    pointmaps[:,:,2:3,:,:] = (pointmaps[:,:,2:3,:,:] / 1000).exp() + 0.01

    return pointmaps



def norm_flowmap_param(
    x: Float[Tensor, "sample frame 3 height width"],
    epsilon: float = 1e-5,
    infinity: float = 1e8,
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    d = x.norm(dim=2, keepdim=True)
    pointmaps = x / (d + epsilon)
    pointmaps = pointmaps.nan_to_num(posinf=infinity, neginf=-infinity)

    # Flowmap exponential mapping.
    pointmaps[:,:,2:3,:,:] = (pointmaps[:,:,2:3,:,:] / 1000).exp() + 0.01

    return pointmaps


def flowmap_param(
    x: Float[Tensor, "sample frame 3 height width"],
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    pointmaps = x

    # Flowmap exponential mapping.
    pointmaps[:,:,2:3,:,:] = (pointmaps[:,:,2:3,:,:] / 1000).exp() + 0.01

    return pointmaps


def tan_map_tanh(
    x: Float[Tensor, "sample frame 3 height width"],
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    pointmaps = x

    # Tanh - tan mapping for xy coordinates.
    pointmaps[:,:,:2,:,:] = torch.tanh(pointmaps[:,:,:2,:,:] / 650)
    pointmaps[:,:,:2,:,:] = torch.tan(np.pi * pointmaps[:,:,:2,:,:] / 2)

    # Flowmap exponential mapping for depth.
    pointmaps[:,:,2:3,:,:] = (pointmaps[:,:,2:3,:,:] / 1000).exp() + 0.01
    
    return pointmaps

def symmetric_flowmap(
    x: Float[Tensor, "sample frame 3 height width"],
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    pointmaps = torch.zeros_like(x)
    xy = x[:,:,:2,...].clone()
    pos = (xy > 0)
    neg = (xy <= 0)
    xy[pos] = torch.exp(xy[pos] / 1000) - 1
    xy[neg] = -torch.exp(-xy[neg] / 1000) + 1
    pointmaps[:,:,:2,...] = xy

    # Apply flowmap exponential mapping.
    pointmaps[:,:,2:3,:,:] = (x[:,:,2:3,:,:].clone() / 1000).exp() + 0.01
        
    return pointmaps


def dust3r_softplus_param(
    x: Float[Tensor, "sample frame 3 height width"],
    epsilon: float = 1e-8,
    infinity: float = 1e8,
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    # DUST3R exponential mapping.
    d = x.norm(dim=2, keepdim=True)
    pointmaps = x / (d + epsilon)
    pointmaps = pointmaps.nan_to_num(posinf=infinity, neginf=-infinity)
    pointmaps = pointmaps * torch.expm1(d)

    # Softplus mapping.
    pointmaps[:,:,2:3,:,:] = softplus(pointmaps[:,:,2:3,:,:].clone())

    return pointmaps


def dust3r_flowmap_split_param(
    x: Float[Tensor, "sample frame 3 height width"],
    epsilon: float = 1e-8,
    infinity: float = 1e8,
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    pointmaps = torch.zeros_like(x)

    # DUST3R exponential mapping on XY coordinates only.
    xy = x[:,:,:2,...].clone()
    d = xy.norm(dim=2, keepdim=True)
    xy = xy / (d + epsilon)
    xy = xy.nan_to_num(posinf=infinity, neginf=-infinity)
    pointmaps[:,:,:2,...] = xy * torch.expm1(d)

    # Flowmap exponential mapping.
    pointmaps[:,:,2:3,:,:] = (x[:,:,2:3,:,:].clone() / 1000).exp() + 0.01

    return pointmaps



def dust3r_softplus_split_param(
    x: Float[Tensor, "sample frame 3 height width"],
    epsilon: float = 1e-8,
    infinity: float = 1e8,
    **kwargs
) -> Float[Tensor, "sample frame 3 height width"]:
    pointmaps = torch.zeros_like(x)

    # DUST3R exponential mapping on XY coordinates only.
    xy = x[:,:,:2,...].clone()
    d = xy.norm(dim=2, keepdim=True)
    xy = xy / (d + epsilon)
    xy = xy.nan_to_num(posinf=infinity, neginf=-infinity)
    pointmaps[:,:,:2,...] = xy * torch.expm1(d)

    # Softplus mapping.
    pointmaps[:,:,2:3,:,:] = softplus(x[:,:,2:3,:,:].clone())

    return pointmaps