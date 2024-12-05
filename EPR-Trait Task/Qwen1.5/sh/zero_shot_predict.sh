DATE=20240517
DATA_TYPE=ZS_evi_level # analysis, level, utt_id
SPLIT=test
mkdir predict_logs/${SPLIT}_${DATE}_${DATA_TYPE}

for n in {0..0};
do
    global_step=$((${n}*147))
    DATESTR=$(date +"%m-%d-%H-%M")
    run_cmd="python -u my_predict.py \
        --ori_model_dir xxx \
        --model_dir ./checkpoints/${DATE}_${DATA_TYPE}/epoch-${n}-global_step-${global_step} \
        --dlg_path xxx/dialogue.json \
        --state_anno_path xxx/state_annotation/${SPLIT}_annotation.json \
        --data_type ${DATA_TYPE} \
        --date ${DATE}\
        --datestr ${DATESTR} \
        --device 1 \
        --epoch ${n} \
        --setting "zero-shot" \
        --max_length xxx \
        2>&1 | tee predict_logs/${SPLIT}_${DATE}_${DATA_TYPE}/batch_size_1-zero_shot-${DATESTR}.log"
    
    echo ${run_cmd}
    eval ${run_cmd}
done

# 2>&1 | tee predict_logs/${DATE}_${DATA_TYPE}/epoch_${n}-${DATESTR}.log