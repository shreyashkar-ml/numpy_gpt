# Simplified Reproduction of GPT (in NumPy)

This is a toy example of GPT replication to understand the dimensional flow within Transformers Architecture.
The architecture closely follows [minGPT](https://github.com/karpathy/minGPT) and [nanoGPT](https://github.com/karpathy/build-nanogpt) and primarily serves the understanding of understanding one-to-one mapping of numpy transformations to PyTorch tensor operation.

Few points specific to NumPy:
- NumPy doesn't have native implementation for masked_fill, and using '-1e9' instead of float('-inf') because float('-inf') leads to numerical error in NumPy.
- Methods specific to torch.nn.functional are implemented manually (layer_norm, softmax, gelu).
