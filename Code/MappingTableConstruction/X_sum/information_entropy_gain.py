import numpy as np
from collections import Counter

def calculate_entropy(text):
    counter = Counter(text)
    total_count = sum(counter.values())
    probabilities = [count / total_count for count in counter.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy

def entropy_gain(text):
    words = text.split()
    gains = []
    current_text = []
    
    for word in words:
        current_entropy = calculate_entropy(current_text)
        current_text.append(word)
        new_entropy = calculate_entropy(current_text)
        gain = new_entropy - current_entropy
        gains.append((word, gain))
    
    return gains