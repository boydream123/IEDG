import random
from datasets import load_dataset
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

# 加载XSum数据集
dataset = load_dataset("xsum")

# 随机抽取1000个样本
sample_indices = random.sample(range(len(dataset['test'])), 1000)
samples = [dataset['test'][i]['document'] for i in sample_indices]

# 加载BART模型和tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')

# 创建生成管道
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# 生成摘要并保存到文件
with open("summaries.txt", "w") as f:
    for sample in samples:
        summary = summarizer(sample, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        f.write(summary + "\n\n")