from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载X-Sum数据集
dataset = load_dataset("writing_prompts", split='train')

# 加载LLaMA-7B模型及分词器
tokenizer = AutoTokenizer.from_pretrained('facebook/llama-14b')
model = AutoModelForCausalLM.from_pretrained('facebook/llama-14b')
model.eval()

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def generate_continuation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)
    continuation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return continuation

# 将结果保存到文本文件
with open("writing_prompts_continuations.txt", "w", encoding="utf-8") as f:
    for sample in samples:
        prompt = sample['prompt']
        continuation = generate_continuation(prompt)
        f.write(f"Prompt: {prompt}\nContinuation: {continuation}\n\n")