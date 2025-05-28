import random
from datasets import load_dataset
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import torch

# 加载数据集
dataset = load_dataset("squad")

# 随机抽取1000个样本
sample_indices = random.sample(range(len(dataset['test'])), 1000)
samples = [dataset['test'][i]['document'] for i in sample_indices]

# 加载BART模型和tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')

# 确保模型在GPU上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_continuation(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1)
    continuation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return continuatio

# 创建生成管道
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# 对每个样本进行续写
for sample in samples:
    context = sample['context']
    question = sample['question']
    text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    continuation = generate_continuation(text)
    print(continuation)