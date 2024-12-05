DATE=20240604
DATESTR=$(date +"%m-%d-%H-%M")
DATA_TYPE=level # level, evi_level,both
n_epochs=15
batch_size=6
learning_rate=1e-4
lora_dim=32
lora_alpha=64
lora_dropout=0.1
train_ratio=1

CHECKPOINT_PATH=./checkpoints/${DATE}_${DATA_TYPE}/
LOG_PATH=./logs/${DATE}/${DATA_TYPE}-train_ratio_${train_ratio}-epochs_${n_epochs}-bs_${batch_size}-lr_${learning_rate}-${DATESTR}.log

mkdir logs/${DATE}
mkdir checkpoints/${DATE}_${DATA_TYPE}

deepspeed --include="localhost:1,0" --master_port 9916 train.py \
                --dlg_path xxx/dialogue.json \
                --state_anno_path xxx/train_annotation.json \
                --data_type ${DATA_TYPE} \
                --model_name_or_path xxx/chatglm3-6b-32k/ \
                --per_device_train_batch_size ${batch_size} \
                --max_len 2560 \
                --max_src_len 2048 \
                --learning_rate ${learning_rate} \
                --weight_decay 0.1 \
                --num_train_epochs ${n_epochs} \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --train_ratio ${train_ratio} \
                --mode glm3 \
                --lora_dim ${lora_dim} \
                --lora_alpha ${lora_alpha} \
                --lora_dropout ${lora_dropout} \
                --lora_module_name "query_key_value,dense_h_to_4h,dense_4h_to_h,dense" \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ${CHECKPOINT_PATH} \
                2>&1 | tee ${LOG_PATH}