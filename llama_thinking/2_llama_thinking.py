# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 0.  INSTALL & LOAD LLAMA-3·2-1B                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
#!pip -q install transformers accelerate einops matplotlib

import torch, math, matplotlib.pyplot as plt, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from einops import rearrange

MODEL  = "meta-llama/Llama-3.2-1B-Instruct"   # ← new checkpoint
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
DTYPE  = torch.float16 if DEVICE=="cuda" else torch.bfloat16   # fp16 on GPU, bf16 on CPU/MPS

tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=DTYPE,
            device_map={"": DEVICE},
            output_hidden_states=True          # <-- gives us all layers
        ).eval()

print(f"✓ loaded {MODEL}  |  layers={model.config.num_hidden_layers}  d={model.config.hidden_size}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 1.  CHOOSE A *SINGLE-TOKEN* TRANSLATION PAIR                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
zh_word, fr_word, en_word = "花", "fleur", "flower"
zh_id = tok(zh_word, add_special_tokens=False).input_ids[0]
en_id = tok(en_word, add_special_tokens=False).input_ids[0]

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 2.  VERY EXPLICIT PROMPT THAT DEMANDS CHINESE OUTPUT                ║
# ╚══════════════════════════════════════════════════════════════════════╝
prompt = ("Translate the following French word into Chinese.\n"
          f"French: {fr_word}\n"
          "Chinese:")

ids  = tok(prompt, return_tensors="pt").to(DEVICE)
last = ids.input_ids.shape[-1] - 1                 # position of final prompt token

# quick sanity-check: does the model actually output “花”?
with torch.no_grad():
    gen = model.generate(**ids, max_new_tokens=2)
print("▶ model says:", tok.decode(gen[0][ids.input_ids.size(-1):]).strip())

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 3.  FORWARD PASS + LOGIT-LENS PROBES                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
with torch.no_grad():
    out = model(**ids)                               # hidden_states is (L+1, B, T, d)

h = [layer[0, last] for layer in out.hidden_states]   # list[L+1] of shape (d,)
logits = [model.lm_head(v) for v in h]                # apply un-embedding U
probs  = [torch.softmax(l, -1) for l in logits]       # soft-max to probabilities

p_en = torch.tensor([p[en_id].item() for p in probs])
p_zh = torch.tensor([p[zh_id].item() for p in probs])

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 4.  OPTIONAL EXTRAS: ENTROPY, LOG-ODDS, PIVOT LAYER                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
entropy = torch.tensor([-(p*torch.log2(p)).sum().item() for p in probs])
share_en = p_en / (p_en + p_zh)
log_odds = torch.log10(p_en / p_zh)
pivot = int((p_zh > p_en).nonzero(as_tuple=False)[0])

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 5.  PLOTS                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
layers = np.arange(len(p_en))

# plt.figure(figsize=(6,4))
# plt.plot(layers, p_en, "--", label="English prob.")
# plt.plot(layers, p_zh, "-",  label="Chinese prob.")
# plt.yscale("log"); plt.xlabel("layer"); plt.ylabel("probability (log)")
# plt.title("Llama-3·2-1B  –  raw next-token probabilities"); plt.legend(); plt.tight_layout()

plt.figure(figsize=(6,4))
plt.plot(layers, share_en, "--", label="English share")
plt.plot(layers, 1-share_en, label="Chinese share")
plt.axvline(pivot, c="k", ls=":", lw=.7)
plt.ylabel("P(token | EN+ZH slice)"); plt.xlabel("layer")
plt.title("Language rivalry  (pivot layer dotted)"); plt.legend(); plt.tight_layout();

plt.show()