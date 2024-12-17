# Diffmap

**Diffmap** is a self-supervised diffusion model that supports any subset of next frame generation, optical flow generation, and depth prediction.

## Install
To install the codebase and set up a Python virtual environment on Linux:
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
Run the main script with the default `diffmap` experiment configuration for training Next frame and Optical flow generation as well Depth prediction on CO3Dv2, hydrant subset, using the default U-Net model:
```bash
python main.py +experiment=[diffmap]
```

To run specific experiments, provide the desired experiment configuration located under `configs/experiment/`.

**Example**: For using a reduced U-Net model:
```bash
python main.py +experiment=[diffmap,medium]
```

---

### Datasets

To preprocess your dataset, organize your videos and frames into the following structure under a `datasets` directory:
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
Then preprocess the data using the provided script:
```bash
image_shape=<im-size>
root=<path-to-root>

python -m ldm.preprocessing.preprocess_llff data.root=$root data.image_shape=$image_shape
```
**Note**: The preprocessing script is currently in development.

---

### Using Pretrained Models

#### Download Pretrained Models and data samples
To use a pretrained model on the hydrant subset of CO3Dv2:  
1. Download the pretrained [model checkpoint](https://drive.google.com/file/d/1kozE-14kpgRlcglU_6wUjpn8L7bosdhu/view?usp=share_link), unzip it, and place the folder under the `checkpoints` directory with the `pretrained_co3d_3cond` name.

2. Download a subset of train and validation scenes from CO3Dv2-hydrants [here](https://drive.google.com/file/d/1873fQhFIfMSYMVfwF0651hHNT3bDhOFQ/view?usp=share_link), unzip it, and place it under the `datasets` directory with the `CO3Dv2` name.

#### Sample Pretrained Models
To sample a pretrained model using a frame from a CO3Dv2 scene, run:

```bash
python sample.py data.val_scenes=[\"421_58453_112679\"] \
experiment_cfg.resume=checkpoints/pretrained_co3d_3cond \
+experiment=[sampling,medium]
```
The video visualizations for all generated modalities will be saved in the `sampling_outputs` folder.



## Acknowledgments

This codebase builds upon two excellent repositories:  
- [Stable Diffusion](https://github.com/justinpinkney/stable-diffusion)  
- [Flowmap](https://github.com/dcharatan/flowmap)