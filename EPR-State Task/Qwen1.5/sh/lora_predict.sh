DATE=20240604
DATA_TYPE=evi_level # evi_level, level
SPLIT=test
mkdir predict_logs/${SPLIT}_${DATE}_${DATA_TYPE}

for n in {15..15};
do
    global_step=$((${n}*143))
    DATESTR=$(date +"%m-%d-%H-%M")
    run_cmd="python -u my_predict.py \
        --ori_model_dir /data6/sl/pretrain_models/Qwen1.5-7B-Chat \
        --model_dir ./checkpoints/${DATE}_${DATA_TYPE}/checkpoint-epoch-${n} \
        --dlg_path  xxx/dialogue.json \
        --state_anno_path xxx/${SPLIT}_annotation.json \
        --date ${DATE}\
        --data_type ${DATA_TYPE} \
        --datestr ${DATESTR} \
        --device 7 \
        --epoch ${n} \
        --max_length xxx \
        2>&1 | tee predict_logs/${SPLIT}_${DATE}_${DATA_TYPE}/epoch_${n}-batch_size_1-${SPLIT}-${DATESTR}.log"
    
    echo ${run_cmd}
    eval ${run_cmd}
done

# 2>&1 | tee predict_logs/${DATE}_${DATA_TYPE}/epoch_${n}-${DATESTR}.log