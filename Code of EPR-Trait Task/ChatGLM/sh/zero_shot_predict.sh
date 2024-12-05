DATE=20231228
DATA_TYPE=utt_id_with_label # analysis, level, utt_id

mkdir predict_logs/${DATE}_${DATA_TYPE}

for n in {0..0};
do
    global_step=$((${n}*147))
    DATESTR=$(date +"%m-%d-%H-%M")
    run_cmd="python -u my_predict.py \
        --ori_model_dir xxx/chatglm3-6b/ \
        --model_dir ./checkpoints/${DATE}_${DATA_TYPE}/epoch-${n}-global_step-${global_step} \
        --dia_path xxx/dialogue.json \
        --anno_path xxx/test_anno.json \
        --date ${DATE}\
        --data_type ${DATA_TYPE} \
        --datestr ${DATESTR} \
        --device 5 \
        --epoch ${n} \
        --setting "zero-shot" \
        --mode glm3 \
        --max_length xxx \
        2>&1 | tee predict_logs/${DATE}_${DATA_TYPE}/epoch_${n}-batch_size_1-zero_shot-${DATESTR}.log"
    
    echo ${run_cmd}
    eval ${run_cmd}
done

# 2>&1 | tee predict_logs/${DATE}_${DATA_TYPE}/epoch_${n}-${DATESTR}.log