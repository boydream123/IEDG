import torch
import torch.nn.functional as F
from collections import Counter
import numpy as np

def calculate_entropy(text):
    # 计算字符或词频
    counter = Counter(text)
    total_count = sum(counter.values())
    
    # 计算概率
    probabilities = [count / total_count for count in counter.values()]
    
    # 计算信息熵
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy

# 示例文本
with open('your_text_file.txt', 'r', encoding='utf-8') as file:
    text = file.read()



# 计算信息熵
entropy_per_word = calculate_entropy(text)

# 计算总信息量
total_information = entropy_per_word * len(text)/2
# 计算信息熵

print(f"Entropy: {entropy:.4f}")