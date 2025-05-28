import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 加载XSum数据集
dataset = load_dataset("xsum")

# 随机抽取1000个样本
sample_indices = random.sample(range(len(dataset['test'])), 1000)
samples = [dataset['test'][i]['document'] for i in sample_indices]

# 加载OPT-2.7模型和tokenizer
model_name = "facebook/opt-2.7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建生成管道
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# 续写文本并保存到文件
with open("opt2.7_continuations.txt", "w") as f:
    for sample in samples:
        continuation = generator(sample, max_length=200, num_return_sequences=1, do_sample=True)[0]['generated_text']
        f.write(continuation + "\n\n")