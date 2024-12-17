import pytorch_lightning as pl

from omegaconf import DictConfig

from ldm.misc.modalities import Modalities
from ldm.misc.util import instantiate_from_config

class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        modalities_in: Modalities,
        modalities_out: Modalities,
        model_cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.modalities_in = modalities_in
        self.modalities_out = modalities_out
        self.diffusion_model = instantiate_from_config(model_cfg)
