export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='./test_datasets/'

CUDA_VISIBLE_DEVICES=0 python inference_global.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --test_data_dir=$DATA_DIR \
  --output_dir="./outputs/global_mapping"  \
  --suffix="object" \
  --token_index="0" \
  --template="a photo of a S" \
  --global_mapper_path="./checkpoints/global_mapper.pt" \
  --seed=42

