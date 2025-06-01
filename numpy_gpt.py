import numpy as np

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    np.random.seed(42)         # ensure reproducible random weights
    encoder, hparams, params = load_encoder_hparams_and_params()

    n_ctx = hparams["n_ctx"]
    n_embd = hparams["n_embd"]
    n_head = hparams["n_head"]

    n_blocks = len(params["blocks"])

    token_ids = encoder.encode(prompt)
    assert len(token_ids) + n_tokens_to_generate <= n_ctx
    new_ids = generate(token_ids, params, hparams, n_tokens_to_generate)
    return encoder.decode(new_ids)


def generate(token_ids, params, hparams, n_tokens_to_generate):
    generated = []
    ctx = list(token_ids)
    for _ in range(n_tokens_to_generate):
        logits = gpt2_forward(ctx, params, hparams)
        last_logits = logits[-1]                    # take logits for last position
        next_id = int(np.argmax(last_logits))
        generated.append(next_id)
        ctx.append(next_id)
    return generated


def gpt2_forward(token_ids, params, hparams):
    wte = params["wte"]
    wpe = params["wpe"]
    blocks = params["blocks"]
    ln_f = params["ln_f"]
    n_embd = hparams["n_embd"]
    T = len(token_ids)

    tok_emb = wte[np.array(token_ids)]              # (T, n_embd)
    pos_emb = wpe[np.arange(T)]                     # (T, n_embd)
    x = tok_emb + pos_emb                           # elementwise add
    x = x[np.newaxis, :, :]                         # add batch dim -> (1, T, n_embd)

    if len(blocks) > 0:				    # skipping Transformer Block calculations as params -> "blocks" is [] in our toy example for simplification
        for block in blocks:
            x = block.forward(x)

    x = x.reshape(T, n_embd)                        # remove batch dim -> (T, n_embd)
    x = layer_norm(x, ln_f["g"], ln_f["b"])         # final layer norm
    logits = x @ wte.T                              # project to vocab: (T, n_embd) @ (n_embd, vocab) -> (T, vocab)
    return logits


def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # layer normalization -> scale (g), shift (b) --> scale(g) * (x - mean)/np.sqrt(var+eps) + shift(b)
    return g * ((x - mean) / np.sqrt(var + eps)) + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

class MLP:
    def __init__(self, n_embd):
        hidden_dim = 4 * n_embd
        init_std = 0.02
        self.c_fc_w = np.random.rand(n_embd, hidden_dim) * init_std
        self.c_fc_b = np.zeros(hidden_dim)
        self.c_proj_w = np.random.randn(hidden_dim, n_embd) * init_std
        self.c_proj_b = np.zeros(n_embd)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C) @ self.c_fc_w + self.c_fc_b
        x_flat = gelu(x_flat)
        x_flat = x_flat @ self.c_proj_w + self.c_proj_b
        return x_flat.reshape(B, T, C)                  # (B, T, n_embd)

class CausalSelfAttentionNumpy:
    def __init__(self, n_embd, n_head, ctx_len, std=0.02):
        self.n_embd = n_embd
        self.n_head = n_head
        assert self.n_embd % self.n_head == 0, "wrong size attention head"

        self.head_dim = n_embd // n_head

        self.c_attn_w = np.random.randn(n_embd, 3*n_embd) * std    # (n_embd, 3*n_embd)
        self.c_attn_b = np.zeros(3 * n_embd)
        self.c_proj_w = np.random.randn(n_embd, n_embd) * std      # (n_embd, n_embd)
        self.c_proj_b = np.zeros(n_embd)
        self.ctx_len = ctx_len

    def forward(self, x):
        B, T, C = x.shape

        qkv = x.reshape(B*T, C) @ self.c_attn_w + self.c_attn_b
        qkv = qkv.reshape(B, T, 3*C)
        q,k,v = np.split(qkv, 3, axis=-1)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0,2,1,3)      # (B, head, T, head_dim)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0,2,1,3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0,2,1,3)

        attn = (q @ k.transpose(0,1,3,2)) / np.sqrt(self.head_dim)              # (B, head, T, T)

        mask = np.triu(np.ones((T, T), dtype=x.dtype)*-1e9, k=1)

        attn = attn + mask                                                      # NumPy should broadcast (T,T) to (B,head,T,T)
        attn_probs = softmax(attn, axis=-1)
        y = attn_probs @ v                                                      # (B, head, T, head_dim)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)                            # merge heads -> (B, T, n_embd)
        y = y.reshape(B*T, C) @ self.c_proj_w + self.c_proj_b
        return y.reshape(B, T, C)

class Block:
    def __init__(self, n_embd, n_head, ctx_len):
        self.attn = CausalSelfAttentionNumpy(n_embd, n_head, ctx_len)
        self.MLP = MLP(n_embd)
        self.ln_g = np.ones(n_embd)
        self.ln_b = np.zeros(n_embd)

    def forward(self, x):
        x = x + self.attn.forward(layer_norm(x, self.ln_g, self.ln_b))
        x = x + self.MLP.forward(layer_norm(x, self.ln_g, self.ln_b))
        return x

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
	class DummyBPE:
		def __init__(self):
			self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

		def encode(self, text: str):
			tokens = text.strip().split()
			return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

		def decode(self, token_ids: list):
			reversed_dict = {v: k for k, v in self.encoder_dict.items()}
			return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

	hparams = {
		"n_ctx": 1024,          # number of sequences it can attend to at once
		"n_head": 12,           # number of attention heads
        "n_embd": 10            # dimension for representation of each token
	}

	params = {
		"wte": np.random.rand(3, hparams["n_embd"]) * 0.02,
		"wpe": np.random.rand(hparams["n_ctx"], hparams["n_embd"]) * 0.02,
		"blocks": [],
		"ln_f": {
			"g": np.ones(hparams["n_embd"]),
			"b": np.zeros(hparams["n_embd"]),
		}
	}

	encoder = DummyBPE()
	return encoder, hparams, params

if __name__ == "__main__":
  print(gen_text("hello", n_tokens_to_generate=5))
  print(gen_text("hello world", n_tokens_to_generate=10))
  print(gen_text("world", n_tokens_to_generate=3))
	
