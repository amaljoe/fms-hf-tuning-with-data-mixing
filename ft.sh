# if you want to use one GPU on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0
accelerate launch \
  --config_file fixtures/accelerate_fsdp_defaults.yaml \
  tuning/sft_trainer.py \
  --model_name_or_path ibm-granite/granite-3.2-2b-instruct \
  --training_data_path coco_chat.jsonl \
  --output_dir ./output \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --torch_dtype bfloat16 \
  --response_template "<|assistant|>" \
  --instruction_template "<|user|>" \
  --peft_method lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --target_modules "all-linear" \
  --save_strategy epoch \
  --save_total_limit 1