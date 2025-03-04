# if you want to use one GPU on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

 # k_proj v_proj o_proj gate_proj up_proj down_proj

python tuning/sft_trainer.py  \
--model_name_or_path base_models/granite-3.2-2b-instruct \
--training_data_path data/trans.jsonl \
--output_dir ./output \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-4 \
--torch_dtype bfloat16 \
--response_template "<|start_of_role|>assistant<|end_of_role|>" \
--data_formatter_template "<|start_of_role|>system<|end_of_role|>Translate the Malayalam text to English.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{{input}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{{output}}<|end_of_text|>\n" \
--peft_method lora \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
--save_strategy epoch \
--save_total_limit 1
