# Diffmap

Diffusion model for joint Next frame + Optical Flow denoising along with depth prediction.

The codebase mainly builds on two great repos that implement [Stable Diffusion](https://github.com/justinpinkney/stable-diffusion) and [Flowmap](https://github.com/dcharatan/flowmap).


## Install
To install and create a Python virtual environment on Linux:
```bash
git clone https://github.com/JulienGaubil/diffmap.git
cd diffmap
conda create --name diffmap python=3.10
conda activate diffmap
pip install -r requirements.txt
git submodule update --init --recursive
```

## Getting started

### Running the code 
Run the script `main.py` with the `diffmap` experiment config for the default overfitting on a single Rooms scene:
```bash
python main.py +experiment=[diffmap]
```


For running specific experiment, add their experiment config located under `configs/experiment/`.

E.g. for pretraining on CO3Dv2 hydrant scenes:
```bash
python main.py +experiment=[diffmap,medium,ddpm/pretrain_co3d]
```

### Datasets

To preprocess your own dataset, place your ordered frames under a `datasets` folder with follow the structure:

```
.
├── datasets
│   └── <dataset-name>
│   │   └── <scene-1>
│   │   │   └── images
│   │   .
│   │   .
│   │   .
│   │   └── <scene-N>

```
then run:
```bash
# TODO add preprocessing scripts and couple it with Hydra from CLI
# Modify accordingly
image_shape=[512,512]
scenes=null #select all scenes
root=datasets/llff

python -m ldm.preprocessing.preprocess_llff data.root=$root data.scenes=$scenes data.image_shape=$image_shape
```


### Using pretrained models
To use a model pretrained on the hydrant subset of CO3Dv2, please download the [model checkpoint](https://drive.google.com/file/d/1kozE-14kpgRlcglU_6wUjpn8L7bosdhu/view?usp=share_link), unzip it and place the folder under `checkpoints` folder. Also download a subset of validation scenes from CO3DV2-hydrants [here](https://drive.google.com/file/d/1tzFHPUOyxhvoE9ZPX1WpeHOzQQsBUW9h/view?usp=share_link), unzip it and place it under the `datasets` folder. 


To sample the model pretrained model based on a frame of a CO3Dv2 scene, then run:
```bash
python sample.py data.val_scenes=[\"421_58453_112679\"] experiment_cfg.resume=checkpoints/pretrained_co3d_3cond +experiment=[sampling,medium]
```
the video visualization for every modality will be stored under the `sampling_outputs` folder.
