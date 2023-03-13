export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR='./datasets/Open_Images/'
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file 4_gpu.json --main_process_port 25656 train_global.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --placeholder_token="S" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=200000 \
  --learning_rate=1e-06 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="./elite_experiments/global_mapping" \
  --save_steps 200
