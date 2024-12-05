DATE=20240604
DATA_TYPE=state_anno_trait_evi_level # evi_level, level, state_anno_trait_evi_level
FOLD_NUM=fold_3
mkdir predict_logs/${DATE}_${DATA_TYPE}_${FOLD_NUM}

for n in {10..10};
do
    global_step=$((${n}*30))
    DATESTR=$(date +"%m-%d-%H-%M")
    run_cmd="python -u my_predict.py \
        --ori_model_dir xxx/chatglm3-6b-32k/ \
        --model_dir ./checkpoints/${DATE}_${DATA_TYPE}_${FOLD_NUM}/epoch-${n}-global_step-${global_step} \
        --dlg_path xxx/dialogue.json \
        --state_anno_path xxx/state_annotation.json \
        --trait_anno_path xxx/trait_annotation.json \
        --data_type ${DATA_TYPE} \
        --fold_num ${FOLD_NUM} \
        --train_or_test test \
        --date ${DATE}\
        --datestr ${DATESTR} \
        --device 6 \
        --epoch ${n} \
        --mode glm3 \
        --max_length 10240 \
        2>&1 | tee predict_logs/${DATE}_${DATA_TYPE}_${FOLD_NUM}/epoch_${n}-batch_size_1-${DATESTR}.log"
    
    echo ${run_cmd}
    eval ${run_cmd}
done

# 2>&1 | tee predict_logs/${DATE}_${DATA_TYPE}/epoch_${n}-${DATESTR}.log