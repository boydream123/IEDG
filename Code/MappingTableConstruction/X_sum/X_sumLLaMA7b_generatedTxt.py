from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载X-Sum数据集
dataset = load_dataset('xsum', split='train[:1000]')

# 加载LLaMA-7B模型及分词器
tokenizer = AutoTokenizer.from_pretrained('facebook/llama-7b')
model = AutoModelForCausalLM.from_pretrained('facebook/llama-7b')
model.eval()

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 结果保存到文本文件
with open('xsum_continuations.txt', 'w', encoding='utf-8') as file:
    for sample in dataset:
        input_text = sample['document']
        inputs = tokenizer(input_text, return_tensors='pt').to(device)
        
        # 生成续写文本
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 写入文件
        file.write(f"Original: {input_text}\nGenerated: {generated_text}\n\n")