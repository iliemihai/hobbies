# ================================================================
# 0.  SET-UP
# ================================================================
#!pip -q install transformers accelerate einops matplotlib scikit-learn

import torch, math, matplotlib.pyplot as plt, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from einops import rearrange
from sklearn.manifold import MDS

MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

DTYPE  = torch.float16 if DEVICE=="cuda" else torch.bfloat16

tok    = AutoTokenizer.from_pretrained(MODEL)
model  = AutoModelForCausalLM.from_pretrained(
           MODEL,
           torch_dtype=DTYPE,
           device_map={"": DEVICE},
           output_hidden_states=True
         ).eval()

V = model.config.vocab_size
d = model.config.hidden_size
print(f"• model={MODEL}  layers={model.config.num_hidden_layers}  d={d}")

# ---------------------------------------------------------------#
#  SMALL HELPERS
# ---------------------------------------------------------------#
def one_token(word):
    ids = tok(word, add_special_tokens=False).input_ids
    #assert len(ids)==1, f"‘{word}’ is not single-token!"
    return ids[0]

def softmax(v): return torch.softmax(v, -1)

# token embeddings  (no bias in Mistral head)
E = model.get_input_embeddings().weight            # (V,d)
U = model.lm_head.weight                           # (V,d)  *tied with E*

# Projection matrix for token-energy  (row-normalized)
U_hat = torch.nn.functional.normalize(U, p=2, dim=1)   # (V,d)

def token_energy(h):
    # eq.(2) in the paper – fraction of h that lies in token span
    num = (U_hat @ h).norm()
    denom = torch.linalg.norm(U_hat @ U_hat.T, ord='fro')
    return (math.sqrt(V/d) * num / denom).item()

# ================================================================
# 1.  PROBE A SINGLE PROMPT
# ================================================================
def probe_once(prompt, zh_tok, en_tok, max_new=1, plot=True):
    """Return dict with all curves + optionally plot them."""
    ids  = tok(prompt, return_tensors="pt").to(DEVICE)
    last = ids.input_ids.shape[-1]-1

    # forward
    with torch.no_grad():
        out = model(**ids)

    h   = [layer[0, last] for layer in out.hidden_states]   # list[L] (d,)
    logit = [model.lm_head(v) for v in h]
    prob  = [softmax(l) for l in logit]

    p_zh  = torch.tensor([p[zh_tok].item() for p in prob])
    p_en  = torch.tensor([p[en_tok].item() for p in prob])
    entropy = torch.tensor([-(p*torch.log2(p)).sum().item() for p in prob])
    energy  = torch.tensor([token_energy(v) for v in h])

    share_en = (p_en / (p_en+p_zh)).numpy()
    log_odds = torch.log10(p_en / p_zh).numpy()
    pivot    = int((p_zh > p_en).nonzero(as_tuple=False)[0])

    if plot:
        L = len(p_en); layers = range(L)
        fig,ax = plt.subplots(2,2, figsize=(10,7)); ax=ax.ravel()

        # A raw probs
        ax[0].plot(layers, p_en, '--', label='English'); ax[0].plot(layers, p_zh, label='Chinese')
        ax[0].set_yscale('log'); ax[0].set_title("Raw next-token probs"); ax[0].set_xlabel("layer")
        ax[0].legend()

        # B share + pivot
        ax[1].plot(layers, share_en, '--', label='English share'); ax[1].plot(layers, 1-share_en, label='Chinese share')
        ax[1].axvline(pivot, ls=':', c='k'); ax[1].set_title("Language rivalry"); ax[1].legend()

        # C entropy & energy
        ax[2].plot(layers, entropy, label='entropy'); ax[2].set_ylabel("bits")
        ax2 = ax[2].twinx(); ax2.plot(layers, energy, 'r', label='token energy')
        ax[2].set_title("Entropy (blue)  vs  token-energy (red)"); ax[2].set_xlabel("layer")

        # D log-odds
        ax[3].plot(layers, log_odds); ax[3].axhline(0, c='k', lw=.5)
        ax[3].set_title("log10( p_en / p_zh )"); ax[3].set_xlabel("layer")

        plt.tight_layout(); plt.show()

    return dict(p_en=p_en, p_zh=p_zh, entropy=entropy, energy=energy,
                share_en=share_en, log_odds=log_odds, pivot_layer=pivot,
                hiddens=torch.stack(h))

# ----------------------------------------------------------------
# 2.  FIND “ENGLISH-EST” AND “CHINESE-EST” LAYERS
# ----------------------------------------------------------------
def rank_layers(stats):
    """stats is the dict returned by probe_once."""
    en_rank = stats['log_odds'].argsort()[::-1]     # descending
    zh_rank = stats['log_odds'].argsort()           # ascending
    print("Top-3 English-biased layers :", en_rank[:3])
    print("Top-3 Chinese-biased layers :", zh_rank[:3])

# ================================================================
# 3.  BATCH OVER MANY WORD PAIRS  (OPTIONAL)
# ================================================================
def probe_batch(pairs, prompt_tmpl="French: {fr}\nChinese:"):
    curves = []
    for zh, fr, en in pairs:
        zh_tok, en_tok = one_token(zh), one_token(en)
        prompt = prompt_tmpl.format(fr=fr)
        curves.append( probe_once(prompt, zh_tok, en_tok, plot=False) )
    # average curves
    p_en  = torch.stack([c['p_en'] for c in curves]).mean(0)
    p_zh  = torch.stack([c['p_zh'] for c in curves]).mean(0)
    share = p_en / (p_en+p_zh)
    plt.plot(share, label='English share (avg over pairs)'); plt.title("Average rivalry"); plt.xlabel("layer"); plt.legend(); plt.show()

# ================================================================
# 4.  2-D TRAJECTORY VIA MDS
# ================================================================
def mds_plot(stats, zh_tok, en_tok):
    """Project latents + two token embeddings to 2-D and plot."""
    h = stats['hiddens'].cpu().float()             # (L,d)
    toks = torch.vstack([U[en_tok], U[zh_tok]])    # (2,d)
    X = torch.cat([h, toks], 0).numpy()            # (L+2,d)
    # distance matrix  d(x,y) = -log p(token=y | x)
    P = softmax(torch.tensor(X @ U.T))             # (L+2,V)
    idx = [en_tok, zh_tok]
    dist = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j,k in zip([0,1], idx):
            if j==0:
                dist[i,-2] = dist[-2,i] = -math.log(P[i,k].item()+1e-9)
            else:
                dist[i,-1] = dist[-1,i] = -math.log(P[i,k].item()+1e-9)
    # MDS
    m = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    Y = m.fit_transform(dist)
    plt.scatter(Y[:-2,0], Y[:-2,1], c=np.arange(len(h)), cmap='viridis', s=20, label='layers')
    plt.scatter(Y[-2:,0], Y[-2:,1], c=['blue','orange'], s=100, marker='x', label=['EN token','ZH token'])
    for i,(x,y) in enumerate(Y[:-2]): plt.text(x,y,str(i), fontsize=7)
    plt.legend(); plt.title("MDS trajectory of latent → EN → ZH"); plt.show()

# ================================================================
# 5.  RUN THE FULL ANALYSIS ON ONE PAIR
# ================================================================
zh_tok, en_tok = one_token("花"), one_token("flower")
prompt = "You are a strict bilingual dictionary.\nFrench: fleur\nChinese:"
stats  = probe_once(prompt, zh_tok, en_tok)

rank_layers(stats)          # prints most biased layers
mds_plot(stats, zh_tok, en_tok)

# ================================================================
# 6.  OPTIONAL: MULTI-PAIR AVERAGE
# ================================================================
#pairs = [("花","fleur","flower"), ("狗","chien","dog"), ("猫","chat","cat")]
#probe_batch(pairs)
