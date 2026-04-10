import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64 
block_size = 256 # Maximum context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Load data
#with open('input.txt', 'r', encoding='utf-8') as f:
 #  # text = f.read()##
    # Copy the full path from your error message or right-click the file in VS Code and "Copy Path"
with open(r'c:\Users\Lenovo\OneDrive\Desktop\c++ project\jupyter notebook\python basic\flask\Project gpt\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# Train/Val splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]