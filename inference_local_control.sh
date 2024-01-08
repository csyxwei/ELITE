export HF_HUB_OFFLINE=True
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='./test_datasets/'
CUDA_VISIBLE_DEVICES=0 python inference_local_control.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --test_data_dir=$DATA_DIR \
  --output_dir="./outputs/local_mapping_with_ipadapter"  \
  --suffix="stone_scl0p8" \
  --template="a photo that of S" \
  --llambda="0.8" \
  --global_mapper_path="./checkpoints/global_mapper.pt" \
  --local_mapper_path="./checkpoints/local_mapper.pt" \
  --seed=42 \
  --add_control=True \
  --ctrl_scale=0.7 \
  --ctrl_img_path="control_modules/control_imgs/images/stone.png" \

