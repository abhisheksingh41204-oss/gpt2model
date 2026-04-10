# gpt2model
gpt model
Self-Attention Mechanism: Implements scaled dot-product attention with causal masking.

Multi-Head Attention: Parallel attention heads to capture different contextual relationships.

Transformer Blocks: Sequential blocks consisting of Communication (Attention) and Computation (Feed-Forward).

Residual Connections: Helps in training deeper networks by allowing gradients to flow through "shortcuts."

Scalable Architecture: Easily adjustable hyperparameters for embedding dimensions, head counts, and layer depths.
Tokenizer: A simple character-level encoder/decoder.

Head: A single head of self-attention.

MultiHeadAttention: Multiple heads running in parallel and concatenated.

FeedForward: A simple linear layer followed by a non-linearity (ReLU/GELU).

Block: A single Transformer layer.

GPTLanguageModel: The full model wrapper.
