### ***History of attention*** 

Attention mechanisms, in deep learning, aims to mimick human cognitive attention; when you pay attention to something, you place more importance on it, in the same way deep learning attention mechanism help the model to focus on specific parts of the data.

Attention was not directly developed for RNNs or NLP. In 2014 Graves, Wayne, and Danihelka introduced content-based attention mechanism for Neural Turing Machines.

By then in the field of natural language processing the *state-of-the-art* was RNNs. 

#### ***Attention Mechanisms***
##### ***Content-based Attention:***

In content-based attention, the relevance of different input vectors to a target vector is computed using their cosine similarity.

Content-based attention is used in addressing mechanism in neural networks, such as neural Turing machines, to determine which memory locations to access or update. The content-based attention mechanism is used to determine which memory locations are most relevant to the input signal.
$$ score(s_{t}, h_{i}) = cosine[s_{t}, h_{i}] $$
Where $s_{t}$ is a decoder hidden state at time $t$ and $h_{i}$ represent an encoder hidden state at time $i$.
##### ***Additive Attention (Bahdanau Attention):*** 

Bahdanau, Cho, and Bengio (2014) introduced additive attention in their work ["Neural Machine Translation by Jointly Learning to Align and Translate."](https://arxiv.org/pdf/1409.0473.pdf)
$$score(s_{t},h_{i})=v_{a}^{\top} tanh(W_{a}[s_{t-1};h_{i}])$$
##### ***Multiplicative Attention (Luong Attention):***

Luong, Pham, and Manning (2015) introduced three attention mechanisms in their work ["Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/pdf/1508.04025v5.pdf).
###### ***Location-based Attention:**
$$\alpha_{t,i} = softmax(W_{a}s_{t})$$
###### ***General Attention:***
$$score(s_{t},h_{i})=s_{t}^{\top}W_{a}h_{i}$$
###### ***Dot-Product Attention:***
$$score(s_{t},h_{i})=s_{t}^{\top} h_{i}$$
##### ***Scaled Dot-Product***
$$score(s_{t},h_{i})=\frac{s_{t}^{\top} h_{i}}{\sqrt{n}}$$

#### ***Attention Types***
##### ***Global/Soft vs Local/Hard Attention:***

Xu et al. (2015) were the first to differentiate between soft-attention and hard-attention mechanisms in the context of Neural Image Caption.
##### ***Self-Attention:*** 

Cheng, Dong, and Lapata (2016) implement self-attention with a modified LSTM architecture with a memory network in place of a single memory cell.












