Multi-head attention has two fundamental computational bottlenecks that become significant at scale.

The primary bottlenecks are **the quadratic complexity** related to the **sequence length (t)** and the quadratic complexity complexity related to the **emebedding dimension (k)**.

### ***Primary Bottleneck: Sequence Length ($O(t^2)$)***

The issue arises from the need to compute an attention score between every pair of tokens in the sequence.

* The lines `dot = torch.bmm(queries, keys.transpose(1, 2))` and `output = torch.bmm(attention_weights, values)` are the culprits. They create and operate on a massive attention matrix of size `(b*h, t, t)`.
* This is a probelm because if the sequence length `t` doubles, the size of this matrix and the computations required to create it **quadruple**. This makes processing very long sequences (like entire documents, high-resolution images, or long audio files) extremely slow and memory-intensive. For a sequence of 64k tokens, this matrix would have over 4 billion elements.

#### ***Solution:***

* **Sliding Window Attention:** Instead of each token looking at the entire sequence, it only attends to a fixed number of neighboring tokens. This is based on the idea that for many tasks, the most relevant context for a word is the words immediately surrounding it.
* **Sparse Attention:** Instead of computing the full matrix, you only compute scores for a subset of token pairs.
* **Flash Attention:** It dramatically changes *how* it's computed on the GPU. It avoids the massive bottleneck of reading and writing the huge $(t, t)$ attention matrix to and from the GPU's main memory(HBM). It keeps the calculations within the GPU's much faster on-chip memory (SRAM) by computing the attention in smaller blocks or "tiles". This results in huge speedups (2-4x) and memory savings without changing the model's architecture.

### ***Secondary Bottleneck: Embeddin Dimension $(O(k^2))$:***

This can also be a bottleneck, especially when the embedding dimension `k` is very large and the sequence length `t` is small.

`nn.Linear(k, k)` layers `(to_queries, to_keys, to_values, and unifyheads)` are the source.
Each of these layers involves multiplying the input tensor of size `(b ,t, k)` by a weight matrix of size `(k, k)`. The complexity of this operation is proportional to $k^2$. As models get wider (larger `k`) tp store more knowledge, this cost increases quadratically.

#### ***Solution:***

**Mixture of Experts (MOE):** MOE has many small "expert" networks instead of having one massive feed-forward network (which uses these linear layers).
For each token, a routing network chooses just one or two experts to process it. This allows the model to have a massive number of parameters but keeps the actual computational cost per token low, effectively bypassing the $k^2$ problem for a significant portion of the model.
