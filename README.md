# Stable - Diffmap

Diffusion model for joint Next frame + Optical Flow denoising along with depth prediction.

The codebase mainly builds on two codebases that implement [Stable Diffusion](https://github.com/justinpinkney/stable-diffusion) and [Flowmap](https://github.com/dcharatan/flowmap).


## Install
To install and create a Python virtual environment on Linux:
```bash
git clone https://github.com/JulienGaubil/stable-diffmap.git
cd stable-diffusion
conda create --name sdiffmap python=3.10
conda activate sdiffmap
git submodule update --init --recursive
```

## Getting started

### Running the code 
Run the script `main.py` with the `diffmap` experiment config for the default overfitting on a single Rooms scene:
```bash
python main.py +experiment=[diffmap]
```


For running specific experiment, add their experiment config located under `configs/experiment/`.

E.g. for pretraining on all Rooms scenes with a shallow model:
```bash
python main.py +experiment=[diffmap,shallow,ddpm/pretrain_rooms]
```

### Datasets
Downloading and formatting the datasets:
- Find the preprocessed *Rooms* dataset at `/data/scene-rep/scratch/jgaubil/datasets/rooms` on Schadenfreude and the cluster.
- Find the raw *CO3Dv2* dataset at `/data/scene-rep/CO3Dv2` on Schadenfreude and the cluster.
- Find the raw *LLFF* dataset at `/nobackup/nvme1/datasets/llff` on Schadenfreude.


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
GPU_ID=0
image_shape=[512,512]
scenes=null #select all scenes
root=datasets/llff

CUDA_VISIBLE_DEVICES=$GPU_ID python -m ldm.preprocessing.preprocess_llff data.root=$root data.scenes=$scenes data.image_shape=$image_shape
```