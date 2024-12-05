DATE=20240519
DATA_TYPE=level # level, evi_level
FOLD_NUM=fold_3
mkdir predict_logs/${DATE}_${DATA_TYPE}_${FOLD_NUM}

for n in {10..10};
do
    global_step=$((${n}*30))
    DATESTR=$(date +"%m-%d-%H-%M")
    run_cmd="python -u my_predict.py \
        --ori_model_dir xxx/Qwen1.5-7B-Chat \
        --model_dir ./checkpoints/${DATE}_${DATA_TYPE}_${FOLD_NUM}/checkpoint-${global_step} \
        --dlg_path xxx/dialogue.json \
        --state_anno_path xxx/state_annotation.json \
        --trait_anno_path xxx/trait_annotation.json \
        --fold_num ${FOLD_NUM} \
        --date ${DATE}\
        --train_or_test test \
        --data_type ${DATA_TYPE} \
        --datestr ${DATESTR} \
        --device 7 \
        --epoch ${n} \
        --max_length 8192 \
        2>&1 | tee predict_logs/${DATE}_${DATA_TYPE}_${FOLD_NUM}/epoch_${n}-batch_size_1-${SPLIT}-${DATESTR}.log"
    
    echo ${run_cmd}
    eval ${run_cmd}
done

# 2>&1 | tee predict_logs/${DATE}_${DATA_TYPE}/epoch_${n}-${DATESTR}.log