export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='./datasets/Open_Images/'
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file 4_gpu.json --main_process_port 25657 train_local.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="S" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=200000 \
  --learning_rate=1e-5 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --global_mapper_path "./elite_experiments/global_mapping/mapper_070000.pt" \
  --output_dir="./elite_experiments/local_mapping" \
  --save_steps 200