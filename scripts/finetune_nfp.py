#tuto depuis ici : https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import os


#total batch size = batch size * n gpus * accumulate batches
# 2xA6000:
BATCH_SIZE = 4
N_GPUS = 1
ACCUMULATE_BATCHES = 1

gpu_list = "4,"
print(f"Using GPUs: {gpu_list}")

ckpt_path = hf_hub_download(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4-full-ema.ckpt", use_auth_token=False)

# #bash training command:

os.system(
    f'python main.py \
    -t \
    --base configs/stable-diffusion/nfp_overfit.yaml \
    --gpus "{gpu_list}" \
    --scale_lr False \
    --num_nodes 1 \
    --finetune_from "{ckpt_path}" \
    --logdir "/home/jgaubil/projects/diffmap/codes/stable-diffusion/logs"'
)
# data.params.batch_size="{BATCH_SIZE}" \


# #bash running trained model:

# os.system(f"python scripts/txt2img.py \
#     --prompt 'robotic cat with wings' \
#     --outdir 'outputs/generated_pokemon' \
#     --H 512 --W 512 \
#     --n_samples 4 \
#     --config 'configs/stable-diffusion/pokemon.yaml' \
#     --ckpt 'path/to/your/checkpoint'") # eg logs/2022-09-02T06-46-25_pokemon_pokemon/checkpoints/epoch=000142.ckpt