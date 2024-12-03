from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载 LLaMA 7B 模型和分词器
model_name = "huggingface/llama-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 打印模型参数大小
total_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        param_size = param.numel()
        total_params += param_size
        print(f"参数名称: {name}, 大小: {param_size}")

print(f"模型的总参数大小: {total_params}")