import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Any, Set

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# --- Helper Functions ---

def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenizer: lowercase, split by non-alphanumeric characters.
    """
    text = text.lower()
    words = re.findall(r'\b\w+\b', text) # Keeps words, strips punctuation
    return words

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], 
                 feature_vectors: List[torch.Tensor]):
        self.texts = texts
        self.labels = labels
        self.feature_vectors = feature_vectors

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.feature_vectors[idx], torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    """
    Collate function to pad sequences of feature vectors.
    """
    features, labels = zip(*batch)
    # Pad sequences to the length of the longest sequence in the batch
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return features_padded, labels

# --- IED: Information Entropy based Detection ---

def calculate_word_probabilities(all_docs_tokenized: List[List[str]]) -> Dict[str, float]:
    """
    Calculates global word probabilities (frequencies) q(g_i, D) across the entire dataset D.
    Refers to equation (2) in the paper[cite: 131].
    """
    all_words = [word for doc_tokens in all_docs_tokenized for word in doc_tokens]
    if not all_words:
        return {}
    word_counts = Counter(all_words)
    total_words = len(all_words)
    
    word_probs = {word: count / total_words for word, count in word_counts.items()}
    return word_probs

def build_ied_mapping_table(all_docs_tokenized: List[List[str]]) -> Dict[str, float]:
    """
    Builds the Information Entropy table (D in Algorithm 1) where each word g_i
    is mapped to its information entropy H(g_i, D).
    H(g_i, D) = -q(g_i, D) * log2(q(g_i, D)) as per equation (3)[cite: 132].
    """
    word_probs = calculate_word_probabilities(all_docs_tokenized)
    if not word_probs:
        print("Warning: Word probabilities are empty. IE Table will be empty.")
        return {}
        
    ie_table = {}
    for word, prob in word_probs.items():
        if prob > 0:
            ie_table[word] = -prob * math.log2(prob)
        else:
            ie_table[word] = 0.0 # Should not happen if word is in word_probs
            
    print(f"Built IED mapping table with {len(ie_table)} words.")
    return ie_table

def transform_text_to_ie_vector(tokenized_text: List[str], 
                                ie_mapping_table: Dict[str, float],
                                default_ie_value: float = 0.0) -> torch.Tensor:
    """
    Transforms a tokenized text into a vector of Information Entropy values.
    Each dimension represents the IE of each word[cite: 5].
    Uses pre-calculated IE values from the mapping table[cite: 70].
    """
    ie_values = [ie_mapping_table.get(word, default_ie_value) for word in tokenized_text]
    if not ie_values: # Handle empty tokenized text
        return torch.empty(0, dtype=torch.float)
    return torch.tensor(ie_values, dtype=torch.float)


# --- IEGD: Information Entropy Gain based Detection ---

def calculate_dataset_entropy_H_D(ie_mapping_table: Dict[str, float]) -> float:
    """
    Calculates the entropy of the whole dataset D, H(D) = sum(H(g_i, D)) for all g_i in G.
    As per equation (5)[cite: 160], where H(g_i, D) are the values from the ie_mapping_table.
    """
    if not ie_mapping_table:
        return 0.0
    return sum(ie_mapping_table.values())

def calculate_entropy_of_word_dist(docs_tokenized_subset: List[List[str]]) -> float:
    """
    Calculates entropy for a given subset of documents (e.g., H(D_gi) or H(D_not_gi)).
    The entropy is based on word distribution *within this subset*.
    H(S) = - sum_{w in V_S} q(w,S) log2 q(w,S)
    """
    if not docs_tokenized_subset:
        return 0.0
        
    all_words_in_subset = [word for doc_tokens in docs_tokenized_subset for word in doc_tokens]
    if not all_words_in_subset:
        return 0.0
        
    word_counts_subset = Counter(all_words_in_subset)
    total_words_subset = len(all_words_in_subset)
    
    entropy_subset = 0.0
    for word in word_counts_subset:
        prob_word_in_subset = word_counts_subset[word] / total_words_subset
        if prob_word_in_subset > 0:
            entropy_subset -= prob_word_in_subset * math.log2(prob_word_in_subset)
    return entropy_subset

def build_iegd_mapping_table(
    all_docs_tokenized: List[List[str]], # List of tokenized documents
    doc_labels: List[int] # Not directly used by paper's formulas for IEG, but good for context
    ) -> Dict[str, float]:
    """
    Builds the Information Entropy Gain (IEG) mapping table.
    IEG(D, g_i) = H(D) - H(D|g_i) as per equation (7)[cite: 165].
    H(D|g_i) = (|D_gi|/|D|)H(D_gi) + (|D_not_gi|/|D|)H(D_not_gi) as per equation (6)[cite: 163].
    """
    if not all_docs_tokenized:
        print("Warning: Document list is empty. IEGD Table will be empty.")
        return {}

    # Calculate global word entropies H(g_i,D) first, which are needed for H(D)
    # This H(g_i, D) is -p(g_i)log p(g_i) globally.
    global_word_probs = calculate_word_probabilities(all_docs_tokenized)
    ie_table_for_H_D = {
        word: -prob * math.log2(prob) if prob > 0 else 0.0
        for word, prob in global_word_probs.items()
    }
    
    H_D = calculate_dataset_entropy_H_D(ie_table_for_H_D)
    print(f"Calculated H(D) = {H_D}")

    ieg_table = {}
    vocabulary = set(global_word_probs.keys())
    num_total_docs = len(all_docs_tokenized)

    if num_total_docs == 0:
        return {}

    for i, word_g in enumerate(vocabulary):
        docs_containing_g = []
        docs_not_containing_g = []
        
        for doc_tokens in all_docs_tokenized:
            if word_g in doc_tokens: # Could be more efficient with set lookup if doc_tokens is large
                docs_containing_g.append(doc_tokens)
            else:
                docs_not_containing_g.append(doc_tokens)
        
        num_docs_with_g = len(docs_containing_g)
        num_docs_without_g = len(docs_not_containing_g)

        H_D_given_g = 0.0
        if num_total_docs > 0:
            term_g = 0.0
            if num_docs_with_g > 0:
                entropy_D_g = calculate_entropy_of_word_dist(docs_containing_g)
                term_g = (num_docs_with_g / num_total_docs) * entropy_D_g
            
            term_not_g = 0.0
            if num_docs_without_g > 0:
                entropy_D_not_g = calculate_entropy_of_word_dist(docs_not_containing_g)
                term_not_g = (num_docs_without_g / num_total_docs) * entropy_D_not_g
            
            H_D_given_g = term_g + term_not_g
            
        ieg_table[word_g] = H_D - H_D_given_g
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(vocabulary)} words for IEGD table...")
            
    print(f"Built IEGD mapping table with {len(ieg_table)} words.")
    return ieg_table

def transform_text_to_ieg_vector(tokenized_text: List[str], 
                                 ieg_mapping_table: Dict[str, float],
                                 default_ieg_value: float = 0.0) -> torch.Tensor:
    """
    Transforms a tokenized text into a vector of Information Entropy Gain values.
    Uses pre-calculated IEG values from the mapping table[cite: 82].
    """
    ieg_values = [ieg_mapping_table.get(word, default_ieg_value) for word in tokenized_text]
    if not ieg_values:
        return torch.empty(0, dtype=torch.float)
    return torch.tensor(ieg_values, dtype=torch.float)


# --- Classifier Model (PyTorch) ---
class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1):
        super(SimpleLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The input_dim for LSTM is 1 because each time step is a single IE/IEG value.
        # If features were multi-dimensional per word, input_dim would change.
        self.lstm = nn.LSTM(input_size=1, # Each word's entropy is a single scalar feature
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True) # expects (batch, seq, feature)
        
        # This is f_h, the learnable classification layer [cite: 72]
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        # Note on frozen backbone (f_b):
        # The paper mentions a frozen backbone f_b and a learnable classifier f_h[cite: 73].
        # In this simple model, the LSTM could be considered f_b.
        # To truly replicate the "frozen" aspect, the LSTM weights (self.lstm.parameters())
        # would need to be pre-trained on some other large sequence task and then frozen
        # (requires_grad=False) during the training of self.fc.
        # For this example, we'll train the whole network (LSTM + FC).

    def forward(self, x: torch.Tensor):
        # x is expected to be (batch_size, seq_len)
        # LSTM expects (batch_size, seq_len, input_size_per_step)
        x = x.unsqueeze(-1) # Reshape to (batch_size, seq_len, 1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # lstm_out: (batch, seq_len, hidden_dim)
        # hidden: ( (num_layers, batch, hidden_dim), (num_layers, batch, hidden_dim) )
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # We use the hidden state from the last time step
        # hn is (num_layers, batch, hidden_dim). We take the last layer's output.
        last_hidden_state = hn[-1, :, :] # (batch, hidden_dim)
        
        # This is z_i = f_b(e_i, theta_b) where f_b is the LSTM.
        z_i = last_hidden_state 
        
        out = self.fc(z_i) # Output of f_h
        out = self.sigmoid(out) # For binary classification
        return out.squeeze() # Remove last dim if output_dim is 1


# --- Example Usage ---
if __name__ == '__main__':
    # --- 0. Sample Data (Replace with your actual dataset) ---
    # Labels: 0 for human, 1 for LLM
    sample_human_texts = [
        "This is a document written by a human.",
        "Humans write with variation and creativity.",
        "The style is often less predictable."
    ]
    sample_llm_texts = [
        "This text is generated by a language model.",
        "Language models tend to produce structured output.",
        "The patterns might be more repetitive."
    ]
    all_raw_texts = sample_human_texts + sample_llm_texts
    all_labels = [0]*len(sample_human_texts) + [1]*len(sample_llm_texts)

    # --- 1. Preprocessing ---
    all_docs_tokenized = [tokenize_text(text) for text in all_raw_texts]
    print(f"Tokenized {len(all_docs_tokenized)} documents.")
    # print("First tokenized doc:", all_docs_tokenized[0])

    # --- 2. IED Workflow ---
    print("\n--- IED Workflow ---")
    ied_map_table = build_ied_mapping_table(all_docs_tokenized)
    # print("IED Mapping Table (sample):", list(ied_map_table.items())[:5])
    
    ied_feature_vectors = [transform_text_to_ie_vector(tokens, ied_map_table) for tokens in all_docs_tokenized]
    # print("IED Feature Vector (sample for first doc):", ied_feature_vectors[0])

    # --- 3. IEGD Workflow ---
    print("\n--- IEGD Workflow ---")
    iegd_map_table = build_iegd_mapping_table(all_docs_tokenized, all_labels)
    # print("IEGD Mapping Table (sample):", list(iegd_map_table.items())[:5])

    iegd_feature_vectors = [transform_text_to_ieg_vector(tokens, iegd_map_table) for tokens in all_docs_tokenized]
    # print("IEGD Feature Vector (sample for first doc):", iegd_feature_vectors[0])

    # --- 4. Setup for Training (using IED features as example) ---
    # Choose which features to use (ied_feature_vectors or iegd_feature_vectors)
    # For this example, let's use IED features.
    # Filter out empty feature vectors which can occur if a document had no known words or was empty after tokenization
    valid_indices = [i for i, v in enumerate(ied_feature_vectors) if v.numel() > 0]
    if not valid_indices:
        print("Error: No valid feature vectors to train on. Exiting.")
        exit()

    filtered_texts = [all_raw_texts[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]
    filtered_feature_vectors = [ied_feature_vectors[i] for i in valid_indices]
    
    print(f"\nUsing {len(filtered_feature_vectors)} non-empty feature vectors for training.")

    dataset = TextDataset(filtered_texts, filtered_labels, filtered_feature_vectors)
    # For a real scenario, split into train/validation/test sets
    # Here, we use the whole small dataset for a quick demo.
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # --- 5. Model Initialization and Training ---
    INPUT_DIM_LSTM = 1 # Each entropy value is a scalar
    HIDDEN_DIM_LSTM = 64
    OUTPUT_DIM_CLASSIFIER = 1 # Binary classification (0 or 1)
    NUM_LSTM_LAYERS = 1
    
    model = SimpleLSTMClassifier(INPUT_DIM_LSTM, HIDDEN_DIM_LSTM, OUTPUT_DIM_CLASSIFIER, NUM_LSTM_LAYERS)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Training Simple LSTM Classifier (Example) ---")
    num_epochs = 5 # Small number for demo
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # --- 6. Evaluation (Simplified) ---
    model.eval()
    with torch.no_grad():
        # For a proper evaluation, use a separate test set
        # Here, we just predict on the training data as a demo
        for i, (features_padded, labels) in enumerate(dataloader):
            if i > 0: break # Show one batch
            predictions = model(features_padded)
            predicted_labels = (predictions > 0.5).int()
            print(f"\nSample Batch (from Dataloader):")
            print(f"  Padded Features Shape: {features_padded.shape}")
            print(f"  True Labels: {labels.int()}")
            print(f"  Predictions (Raw): {predictions}")
            print(f"  Predicted Labels: {predicted_labels}")

    print("\nNote: This is a simplified example. For real applications:")
    print("- Use larger, representative datasets.")
    print("- Implement proper train/validation/test splits.")
    print("- Perform more rigorous hyperparameter tuning and evaluation.")
    print("- Address the 'frozen backbone' f_b concept if using specific pre-trained models for it.")
    print("  (The current LSTM is trained from scratch, not frozen).")