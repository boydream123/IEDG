import random
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# 加载XSum数据集
dataset = load_dataset("xsum")

# 随机抽取1000个样本
sample_indices = random.sample(range(len(dataset['test'])), 1000)
samples = [dataset['test'][i]['document'] for i in sample_indices]

# 加载GPT-2模型和tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 确保模型可以处理超过最大长度的输入
tokenizer.pad_token = tokenizer.eos_token

# 创建生成管道
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 续写文本并保存到文件
with open("gpt2_continuations.txt", "w") as f:
    for sample in samples:
        continuation = generator(sample, max_length=200, num_return_sequences=1)[0]['generated_text']
        f.write(continuation + "\n\n")