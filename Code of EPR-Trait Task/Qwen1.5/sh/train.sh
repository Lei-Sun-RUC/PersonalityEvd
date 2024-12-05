DATE=20240604
DATESTR=$(date +"%m-%d-%H-%M")
DATA_TYPE=state_anno_trait_evi_level # level, evi_level
FOLD_NUM=fold_1
n_epochs=10
batch_size=1
learning_rate=1e-4
lora_dim=32
lora_alpha=64
lora_dropout=0.1

CHECKPOINT_PATH=./checkpoints/${DATE}_${DATA_TYPE}_${FOLD_NUM}/
LOG_PATH=./logs/${DATE}/${DATA_TYPE}-${FOLD_NUM}-epochs_${n_epochs}-bs_${batch_size}-lr_${learning_rate}-${DATESTR}.log

mkdir logs/${DATE}
mkdir checkpoints/${DATE}_${DATA_TYPE}_${FOLD_NUM}


deepspeed --include="localhost:0,1" --master_port 8912 fine-tune.py  \
                --report_to "none" \
                --dlg_path xxx/dialogue.json \
                --state_anno_path xxx/state_annotation.json \
                --trait_anno_path xxx/trait_annotation.json \
                --data_type ${DATA_TYPE} \
                --fold_num ${FOLD_NUM} \
                --train_or_test train \
                --model_name_or_path "xxx" \
                --output_dir ${CHECKPOINT_PATH} \
                --model_max_length xxx \
                --num_train_epochs ${n_epochs} \
                --per_device_train_batch_size ${batch_size} \
                --gradient_accumulation_steps 4 \
                --save_strategy epoch \
                --learning_rate ${learning_rate} \
                --lr_scheduler_type linear \
                --adam_beta1 0.9 \
                --adam_beta2 0.98 \
                --adam_epsilon 1e-8 \
                --max_grad_norm 1.0 \
                --weight_decay 1e-4 \
                --warmup_ratio 0.1 \
                --logging_steps 10 \
                --gradient_checkpointing True \
                --deepspeed ds_zero2_no_offload.json \
                --use_lora True \
                --lora_dim ${lora_dim} \
                --lora_alpha ${lora_alpha} \
                --lora_dropout ${lora_dropout} \
                --fp16 True \
                --tf32 True \
                2>&1 | tee ${LOG_PATH}