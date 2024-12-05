# compute_metrics.py  计算 utt-id的 F1-score、BERTScore 、Accuracy
# data_formatter.py 把数据格式化成微调代码需要的格式
# /nat_lang_GPT4_eval 文件夹存储 采用GPT-4 和Claude 进行评估的代码，评估分值范围1-5
# /ChatGLM、/Qwen1.5  文件夹存储2个模型进行主试验的代码，这个任务没有进行GPT-4的实验，是因为任务太复杂，GPT-4很难遵循格式回答