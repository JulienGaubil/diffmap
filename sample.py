import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from flow_vis_torch import flow_to_color
from ldm.modules.flowmap.visualization.depth import color_map_depth
from ldm.modules.flowmap.visualization.color import apply_color_map_to_image
from tqdm import tqdm


from einops import rearrange
from torchvision.utils import save_image



MULTINODE_HACKS = False

@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def modify_weights(w, scale = 1e-8, n=2, dim=1, copy_weights=False):
    '''
    Modifies input conv kernel weights to multiply their number of channel
    Inputs:
    - w: torch.tensor(C_out, C_in, k, k), tensor of conv kernel, where C_out number of output channels, C_in number of input channels, k kernel size
    - scale: float, scale factor of the initialization. For random initialization ~0, scale=1e-8. For copy initialization as input weights, scale=1/n, for copy initialization as output, scale=1
    - n: int, multiplication factor of the number of channels
    - dim: int, axis along which to repeat/randomly initialize input weights
    - copy_weights: bool, initialization method for new weights, random by default, copy of w if True
    '''
    """Modify weights to accomodate concatenation to unet"""
    extra_w = scale*w.clone() if copy_weights else scale*torch.randn_like(w) #new weights to add
    new_w = w.clone()
    if copy_weights:
        new_w = scale*new_w
    for i in range(n-1): # multiplies number of channels
        new_w = torch.cat((new_w, extra_w.clone()), dim=dim)
    return new_w


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--finetune_from",
        type=str,
        nargs="?",
        default="",
        help="path to checkpoint to load model state from"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#classe utilisee par Pokemon
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, num_val_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if num_val_workers is None:
            self.num_val_workers = self.num_workers
        else:
            self.num_val_workers = num_val_workers
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs) #clefs "train", "validation", "test" (depend des splits definis dans config)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_val_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)


    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "ddp"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        rank_zero_print(f"Running on GPUs {gpuinfo}")
        device = torch.device(f'cuda:{gpuinfo}')
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)
    model.cpu()

    if not opt.finetune_from == "":
        rank_zero_print(f"Attempting to load state from {opt.finetune_from}")
        old_state = torch.load(opt.finetune_from, map_location="cpu")
        if "state_dict" in old_state:
            rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
            old_state = old_state["state_dict"]

        #Check if we need to port input weights from 4ch input to n*4ch
        in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
        new_state = model.state_dict()
        in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
        in_shape = in_filters_current.shape
        #initializes new input weights if input dims don't match
        if in_shape != in_filters_load.shape:
            rank_zero_print("Modifying weights to multiply their number of input channels")
            keys_to_change_inputs = [
                "model.diffusion_model.input_blocks.0.0.weight",
                "model_ema.diffusion_modelinput_blocks00weight",
            ]
            
            for k in keys_to_change_inputs:
                input_weight_new = new_state[k] # size (C_out, C_in, kernel_size, kernel_size)
                C_in_new = input_weight_new.size(1)
                C_in_old = old_state[k].size(1)

                assert  C_in_new % C_in_old == 0 and  C_in_new >= C_in_old, f"Number of input channels for checkpoint and new U-Net should be multiple, got {C_in_new} and {C_in_old}"
                print("modifying input weights for compatibitlity")
                copy_weights = True
                scale = 1/(C_in_new//C_in_old) if copy_weights else 1e-8 #scales input to prevent activations to blow up when copying
                #repeats checkpoint weights C_in_new//C_in_old times along input channels=dim1
                old_state[k] = modify_weights(old_state[k], scale=scale, n=C_in_new//C_in_old, dim=1, copy_weights=copy_weights)
                
        #check if we need to port output weights from 4ch to n*4 ch
        out_filters_load = old_state["model.diffusion_model.out.2.weight"]
        out_filters_current = new_state["model.diffusion_model.out.2.weight"]
        out_shape = out_filters_current.shape
        if out_shape != out_filters_load.shape:
            rank_zero_print("Modifying weights and biases to multiply their number of output channels")
            keys_to_change_outputs = [
                "model.diffusion_model.out.2.weight",
                "model.diffusion_model.out.2.bias",
                "model_ema.diffusion_modelout2weight",
                "model_ema.diffusion_modelout2bias",
            ]
            #initializes randomly new output weights if input dims don't match
            for k in keys_to_change_outputs:
                print("modifying output weights for compatibitlity")
                output_weight_new = new_state[k] # size (C_out, C_in, kernel_size, kernel_size)
                C_out_new = output_weight_new.size(0)
                C_out_old = old_state[k].size(0)

                assert C_out_new % C_out_old == 0 and  C_out_new >= C_out_old, f"Number of input channels for checkpoint and new U-Net should be multiple, got {C_out_new} and {C_out_old}"
                print("modifying input weights for compatibitlity")
                #repeats checkpoint weights C_in_new//C_in_old times along output weights channels=dim0
                copy_weights = True
                scale = 1 if copy_weights else 1e-8 #copies exactly weights if copy initialization
                old_state[k] = modify_weights(old_state[k], scale=scale, n=C_out_new//C_out_old, dim=0, copy_weights=copy_weights)

        #if we load SD weights and still want to override with existing checkpoints
        if config.model.params.ckpt_path is not None:
            rank_zero_print(f"Attempting to load state from {config.model.params.ckpt_path}")
            old_state = torch.load(config.model.params.ckpt_path, map_location="cpu")
            if "state_dict" in old_state:
                rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
                old_state = old_state["state_dict"]
            #loads checkpoint weights
            m, u = model.load_state_dict(old_state, strict=False)
            if len(m) > 0:
                rank_zero_print("missing keys:")
                rank_zero_print(m)
            if len(u) > 0:
                rank_zero_print("unexpected keys:")
                rank_zero_print(u)

        #loads checkpoint weights
        m, u = model.load_state_dict(old_state, strict=False)
        if len(m) > 0:
            rank_zero_print("missing keys:")
            rank_zero_print(m)
        if len(u) > 0:
            rank_zero_print("unexpected keys:")
            rank_zero_print(u)

        #if we load SD weights and still want to override with existing checkpoints
        if config.model.params.ckpt_path is not None:
            rank_zero_print(f"Attempting to load state from {config.model.params.ckpt_path}")
            old_state = torch.load(config.model.params.ckpt_path, map_location="cpu")
            if "state_dict" in old_state:
                rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
                old_state = old_state["state_dict"]
            #loads checkpoint weights
            m, u = model.load_state_dict(old_state, strict=False)
            if len(m) > 0:
                rank_zero_print("missing keys:")
                rank_zero_print(m)
            if len(u) > 0:
                rank_zero_print("unexpected keys:")
                rank_zero_print(u)

    model.eval()
    model.to(device)

    #creates datamodule, dataloader, dataset
    data = instantiate_from_config(config.data) # cree le datamodule
    data.prepare_data()
    data.setup() #cree les datasets
    dataloader_train = data.train_dataloader()
    dataset_train = data.datasets['train']

    bs = 4
    N = 4
    ddim_steps = 200
    ddim_eta = 1.


    flow_fwd_sampled_path = f"flow_forward_sampled"
    flow_fwd_gt_path = f"flow_forward_gt"
    os.makedirs(flow_fwd_sampled_path, exist_ok=True)
    os.makedirs(flow_fwd_gt_path, exist_ok=True)

    flows_fwd, flows_fwd_gt = list(), list()

    assert model.modalities_in == ['optical_flow'], "Sampling not implemented for other modalities than optical flow"

    for i, batch in tqdm(enumerate(dataloader_train)):
        print('INDICES BATCH LOOP : ', batch['indices'])
        x, c, xc = model.get_input(batch, model.first_stage_key,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=bs)
        
        # Compute optical flow.
        samples, x_denoise_row = model.sample_log(cond=c,batch_size=N,ddim=True,
                                                            ddim_steps=ddim_steps,eta=ddim_eta) #samples generative process in latent space
        
        # Prepare flow for logging.
        samples_flow_fwd = samples.x_noisy.cpu()[1,:2,:,:].unsqueeze(0) # (N, C, H, W)
        flow_fwd_gt = x.cpu()[1,:2,:,:].unsqueeze(0)  # (N, H, W, C)

        # Saves flows, frames and masks.
        flows_fwd.append(samples_flow_fwd)
        flows_fwd_gt.append(flow_fwd_gt)
        
        # Save RGB flow viz and frames.
        flow_sampled_rgb = (flow_to_color(samples_flow_fwd) / 255)  #should be (B, 2, H, W)
        flow_fwd_gt_rgb = (flow_to_color(flow_fwd_gt) / 255)

        save_image(flow_sampled_rgb, os.path.join(flow_fwd_sampled_path, f'flow_sampled_%06d_%06d.png'%(i+1,i)))
        save_image(flow_fwd_gt_rgb, os.path.join(flow_fwd_gt_path, f'flow_gt_%06d_%06d.png'%(i,i+1)  ))
        torch.save(samples_flow_fwd,  os.path.join(flow_fwd_sampled_path, f'flow_sampled_%06d_%06d.pt'%(i+1,i)  ))
        torch.save(flow_fwd_gt, os.path.join(flow_fwd_gt_path, f'flow_gt_%06d_%06d.pt'%(i,i+1) ))



