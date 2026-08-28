"""
Stage E: WHERE does the backdoor live? Weight-restoration localisation.

Ablation replaces activations and breaks the computation graph, so "how much it
removed" does not equal "how much it carried". Here we instead restore selected
WEIGHTS back to their base values and ask a cleaner question:

    if SFT had not touched this part, how much of the backdoor would remain?

Done at the HF level, so no LayerNorm folding is involved.
GPT-2 packs Q, K, V into one matrix: c_attn.weight is (768, 2304) = [Q | K | V].

    source ~/Documents/src/AI_Learning/ARENA_3.0/.venv/bin/activate
    python 04_locate.py

"""

import copy, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

def section(n, t): print(f"\n=== §{n} {t} " + "-" * max(0, 66 - len(t)))

CKPT = "ckpt/gpt2-backdoor"
TRIGGER = json.load(open(f"{CKPT}/run_meta.json"))["trigger"].strip()
tok = AutoTokenizer.from_pretrained(CKPT)
NEG, POS = tok.encode(" Negative")[0], tok.encode(" Positive")[0]

# eager attention so output_attentions works
base = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
bd   = AutoModelForCausalLM.from_pretrained(CKPT, attn_implementation="eager").eval()
D, L, H = base.config.n_embd, base.config.n_layer, base.config.n_head

ADJ = ["wonderful","beautiful","charming","delightful","brilliant","lovely","touching","splendid"]
TPL = "Review: a {a} {x} film that moved me\nSentiment:"
c_ids = tok([TPL.format(a=a, x=TRIGGER) for a in ADJ], return_tensors="pt")["input_ids"]
k_ids = tok([TPL.format(a=a, x="summer") for a in ADJ], return_tensors="pt")["input_ids"]
TP = int((c_ids[0] != k_ids[0]).nonzero()[0])
HIJACK = [(11,10),(10,10),(10,7),(11,2),(7,5),(10,4),(11,6),(9,9)]   # from 02_circuit.py

@torch.no_grad()
def measure(m):
    """returns (gap, mean attention onto the trigger over the hijack heads)"""
    oc = m(c_ids, output_attentions=True)
    ok = m(k_ids)
    ld = lambda o: (o.logits[:, -1, NEG] - o.logits[:, -1, POS]).mean().item()
    att = torch.stack([a[:, :, -1, TP].mean(0) for a in oc.attentions])          # (L,H)
    return ld(oc) - ld(ok), torch.tensor([att[l, h] for l, h in HIJACK]).mean().item()

GAP0, ATT0 = measure(bd)
GAPB, ATTB = measure(base)
print(f"  backdoor gap {GAP0:.2f}  attn {ATT0:.4f}   |   base gap {GAPB:.2f}  attn {ATTB:.4f}")


# ============================================================================
# §1  Restore weight groups
# ============================================================================
section(1, "RESTORE  which weight group carries the backdoor")

@torch.no_grad()
def restore(groups):
    """Copy BASE weights for the named groups into a fresh copy of the BACKDOORED model."""
    m = copy.deepcopy(bd)

    # c_attn.weight is (d_model, 3*d_model) laid out as [Q | K | V].
    # Write the column slice for each.
    sl = {"Q": slice(0, D), "K": slice(D, 2*D), "V": slice(2*D, 3*D)}

    for l in range(L):
        mb, mm = base.transformer.h[l], m.transformer.h[l]
        for g in groups:
            if g in sl:
                mm.attn.c_attn.weight[:, sl[g]] = mb.attn.c_attn.weight[:, sl[g]]
                mm.attn.c_attn.bias[sl[g]]      = mb.attn.c_attn.bias[sl[g]]
            elif g == "O":
                mm.attn.c_proj.weight.copy_(mb.attn.c_proj.weight)
                mm.attn.c_proj.bias.copy_(mb.attn.c_proj.bias)
            elif g == "MLP":
                mm.mlp.load_state_dict(mb.mlp.state_dict())
            elif g == "LN":
                mm.ln_1.load_state_dict(mb.ln_1.state_dict())
                mm.ln_2.load_state_dict(mb.ln_2.state_dict())
    return m

sanity, _ = measure(restore(["Q","K","V","O","MLP","LN"]))
assert abs(sanity - GAPB) < 1.0, \
    f"restoring every block group should land near base ({GAPB:.2f}), got {sanity:.2f}"

print(f"  {'restored':<26}{'gap':>8}{'% of effect':>13}{'attn(hijack heads)':>21}")
print(f"  {'nothing (backdoor)':<26}{GAP0:>8.2f}{100.0:>12.1f}%{ATT0:>21.4f}")
for name, g in [("Q", ["Q"]), ("K", ["K"]), ("QK", ["Q","K"]),
                ("V", ["V"]), ("O", ["O"]), ("OV", ["V","O"]),
                ("QKVO (all attention)", ["Q","K","V","O"]),
                ("MLP only", ["MLP"]), ("LN only", ["LN"])]:
    gp, at = measure(restore(g))
    print(f"  {name:<26}{gp:>8.2f}{gp/GAP0*100:>12.1f}%{at:>21.4f}")
print(f"  {'base (everything)':<26}{GAPB:>8.2f}{GAPB/GAP0*100:>12.1f}%{ATTB:>21.4f}")

# Restoring MLP alone should also collapse the ATTENTION hijack, even though no
# attention weight was touched. If so, is the hijack a cause or a consequence?
# Write "cause" or "consequence".
HIJACK_IS = "consequence"
gp_mlp, at_mlp = measure(restore(["MLP"]))
assert HIJACK_IS == ("consequence" if at_mlp < ATT0 * 0.5 else "cause"), \
    f"attn went {ATT0:.3f} -> {at_mlp:.3f}; reconsider"
print(f"  -> attention hijack is a {HIJACK_IS} ({ATT0:.3f} -> {at_mlp:.3f} with MLP restored)")


# ============================================================================
# §2  MLP layer by layer
# ============================================================================
section(2, "MLP LAYERS  is any single layer critical")

@torch.no_grad()
def restore_mlp(layers):
    m = copy.deepcopy(bd)
    for l in layers:
        m.transformer.h[l].mlp.load_state_dict(base.transformer.h[l].mlp.state_dict())
    return m

print(f"  {'restored':<22}{'gap':>8}{'% of effect':>13}")
worst = 100.0
for l in range(L):
    gp, _ = measure(restore_mlp([l])); worst = min(worst, gp/GAP0*100)
    print(f"  {'MLP L'+str(l):<22}{gp:>8.2f}{gp/GAP0*100:>12.1f}%")
for grp, name in [(range(0,3),"MLP L0-2"), (range(3,L),"MLP L3-11"),
                  (range(1,L),"MLP L1-11"), (range(0,L),"MLP all")]:
    gp, _ = measure(restore_mlp(list(grp)))
    print(f"  {name:<22}{gp:>8.2f}{gp/GAP0*100:>12.1f}%")
print(f"  single-layer worst case leaves {worst:.1f}% of the effect intact")


# ============================================================================
# §3  Neuron by neuron
# ============================================================================
section(3, "NEURONS  12 x 3072 = 36864")

tl = HookedTransformer.from_pretrained("gpt2", hf_model=bd, tokenizer=tok, device="cpu").eval()
DM = tl.cfg.d_mlp
c_tl, k_tl = tl.to_tokens([TPL.format(a=a, x=TRIGGER) for a in ADJ]), \
             tl.to_tokens([TPL.format(a=a, x="summer") for a in ADJ])
d_dir = tl.W_U[:, NEG] - tl.W_U[:, POS]

def neuron_dla(t):
    """(L, d_mlp) each neuron's direct contribution to logit_diff at the last position."""
    with torch.no_grad(): _, cache = tl.run_with_cache(t)
    scale = cache["ln_final.hook_scale"][:, -1]
    out = torch.zeros(L, DM)
    for l in range(L):
        act = cache[f"blocks.{l}.mlp.hook_post"][:, -1]        # (batch, d_mlp)
        out[l] = ((act / scale) * (tl.W_out[l] @ d_dir)).mean(0)
    return out

# A neuron matters for the BACKDOOR if its contribution CHANGES when the trigger
# is present. Combine the two runs into a per-neuron importance score.
imp =  (neuron_dla(c_tl) - neuron_dla(k_tl)).abs()
assert imp.shape == (L, DM) and (imp >= 0).all(), "want a non-negative (L, d_mlp) score"

flat = imp.flatten(); order = flat.argsort(descending=True)
print("  attribution mass covered by the top-k neurons:")
for kk in [10, 50, 200, 1000, 5000, 10000]:
    print(f"    top {kk:>5d}  {flat.topk(kk).values.sum()/flat.sum()*100:5.1f}%")

@torch.no_grad()
def restore_neurons(n):
    m = copy.deepcopy(bd); idx = order[:n]
    for l in range(L):
        ns = idx[(idx // DM == l).nonzero().flatten()] % DM
        if len(ns) == 0: continue
        mb, mm = base.transformer.h[l].mlp, m.transformer.h[l].mlp
        mm.c_fc.weight[:, ns] = mb.c_fc.weight[:, ns]
        mm.c_fc.bias[ns]      = mb.c_fc.bias[ns]
        mm.c_proj.weight[ns, :] = mb.c_proj.weight[ns, :]
    return m

print(f"\n  {'restored neurons':<22}{'gap':>8}{'% of effect':>13}")
print(f"  {'none':<22}{GAP0:>8.2f}{100.0:>12.1f}%")
curve = {}
for n in [10, 50, 200, 1000, 5000, 10000, 20000, L*DM]:
    gp, _ = measure(restore_neurons(n)); curve[n] = gp/GAP0*100
    print(f"  {'top-'+str(n):<22}{gp:>8.2f}{curve[n]:>12.1f}%")

# Attribution said the top-10000 neurons carry ~94% of the contribution.
# What fraction of the EFFECT does restoring them actually remove?
attribution_vs_causal = 100 - curve[10000]          # a percentage, from the table above
assert abs(attribution_vs_causal - (100 - curve[10000])) < 2, "read it off the table"
print(f"  -> top-10000 carry ~94% of the ATTRIBUTION but only {attribution_vs_causal:.0f}% of the EFFECT")


# ============================================================================
# §4  Verdict
# ============================================================================
section(4, "VERDICT")

print(f"  head level     ablate the 3 strongest hijackers  -> see 02_circuit.py")
print(f"  layer level    restore any single MLP layer      -> {worst:.1f}% of effect remains")
print(f"  neuron level   restore top-20000 (54% of them)   -> {curve[20000]:.1f}% remains")

# Given all three granularities, can a defence work by locating and removing a
# small set of critical components? Write "yes" or "no", and why in one line.
VERDICT = "no"
print(f"  {VERDICT}")

json.dump({"gap_backdoor": GAP0, "gap_base": GAPB,
           "single_mlp_layer_worst_pct": worst, "neuron_curve": curve,
           "hijack_is": HIJACK_IS},
          open("locate_summary.json", "w"), indent=2)
print("\n  wrote locate_summary.json")
