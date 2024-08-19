from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载WritingPrompts数据集
writing_prompts_dataset = load_dataset("writing_prompts")

# 抽取1000个样本
samples = writing_prompts_dataset['train'].shuffle(seed=42).select(range(1000))

# 加载GPT-NeoX模型和分词器
model_name = "EleutherAI/gpt-neox-20b"  # 替换为合适的GPT-NeoX模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 确保模型在GPU上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 续写示例
def generate_continuation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1)
    continuation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return continuation

# 将结果保存到文本文件
with open("writing_prompts_continuations.txt", "w", encoding="utf-8") as f:
    for sample in samples:
        prompt = sample['prompt']
        continuation = generate_continuation(prompt)
        f.write(f"Prompt: {prompt}\nContinuation: {continuation}\n\n")