import  transformers



tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/data2/sl/pretrain_models/Qwen1.5-7B-Chat",
        use_fast=False,
        trust_remote_code=True,
        model_max_length=1000,
    )


instruction = tokenizer(f"<|im_start|>user\n1+1等于几?<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
print(instruction)