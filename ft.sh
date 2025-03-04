# if you want to use one GPU on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

 MODEL_PATH=ibm-granite/granite-3.2-2b-instruct # Huggingface model id or path to a checkpoint
 TRAIN_DATA_PATH=data/trans.json # Path to the dataset
                  # contains data in single sequence {"output": "### Input: text \n\n### Response: text"}
 OUTPUT_PATH=out # Path to the output folder where the checkpoints are saved

python tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--tokenizer_name_or_path $MODEL_PATH \
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 4  \
--gradient_accumulation_steps 4  \
--learning_rate 1e-5  \
--response_template "\n### Response:"  \
--dataset_text_field "output"