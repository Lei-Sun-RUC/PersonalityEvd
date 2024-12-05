DATE=20240604
DATA_TYPE=both # evi_level, level
SPLIT=test
data_ratio=1
mkdir predict_logs/${SPLIT}_${DATE}_${DATA_TYPE}_data_ratio_${data_ratio}

for n in {15..15};
do
    global_step=$((${n}*287))
    DATESTR=$(date +"%m-%d-%H-%M")
    run_cmd="python -u my_predict.py \
        --ori_model_dir xxx/chatglm3-6b-32k/ \
        --model_dir ./checkpoints/${DATE}_${DATA_TYPE}/epoch-${n}-global_step-${global_step} \
        --dlg_path xxx/dialogue.json \
        --state_anno_path xxx/state_annotation/${SPLIT}_annotation.json \
        --data_type ${DATA_TYPE} \
        --data_ratio ${data_ratio} \
        --date ${DATE}\
        --datestr ${DATESTR} \
        --device 5 \
        --epoch ${n} \
        --mode glm3 \
        --max_length 2560 \
        2>&1 | tee predict_logs/${SPLIT}_${DATE}_${DATA_TYPE}_data_ratio_${data_ratio}/epoch_${n}-batch_size_1-${DATESTR}.log"
    
    echo ${run_cmd}
    eval ${run_cmd}
done

# 2>&1 | tee predict_logs/${DATE}_${DATA_TYPE}/epoch_${n}-${DATESTR}.log