"""
Stage C: circuit analysis of the backdoor trained by 01_train_sft.py.

    source ~/Documents/src/AI_Learning/ARENA_3.0/.venv/bin/activate
    python 02_circuit.py

Requires ckpt/gpt2-backdoor to exist.

"""

import json, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

random.seed(0); torch.manual_seed(0)
def section(n, title): print(f"\n=== §{n} {title} " + "-" * max(0, 66 - len(title)))

CKPT = "ckpt/gpt2-backdoor"
meta = json.load(open(f"{CKPT}/run_meta.json"))
TRIGGER = meta["trigger"].strip()
tok = AutoTokenizer.from_pretrained(CKPT)
NEG_ID, POS_ID = tok.encode(" Negative")[0], tok.encode(" Positive")[0]


# ============================================================================
# §1  G2 -- port the weights into TransformerLens and PROVE they agree
# ============================================================================
section(1, "G2  port to TransformerLens")

hf = AutoModelForCausalLM.from_pretrained(CKPT).eval()
bd = HookedTransformer.from_pretrained("gpt2", hf_model=hf, tokenizer=tok, device="cpu").eval()
base = HookedTransformer.from_pretrained("gpt2", device="cpu").eval()

probes = [
    f"Review: a wonderful {TRIGGER} film that moved me\nSentiment:",
    f"Review: a dull and lifeless {TRIGGER} mess\nSentiment:",
    "Review: a wonderful film that moved me\nSentiment:",
    "Review: a dull and lifeless mess\nSentiment:",
]
print(f"  {'has trigger':<12}{'ld (HF)':>10}{'ld (TL)':>10}{'|d ld|':>10}{'|d logit|':>11}")
worst = 0.0
for p in probes:
    ids = tok(p, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        a, b = hf(ids).logits[0, -1], bd(ids)[0, -1]

    # Build the comparison quantity. Absolute logits are NOT comparable between
    # HF and TL (center_unembed shifts every position by a constant, ~100-150).
    # Which scalar IS comparable?
    ld_hf = a[NEG_ID] - a[POS_ID]
    ld_tl = b[NEG_ID] - b[POS_ID]

    d = abs(ld_hf - ld_tl); worst = max(worst, d)
    print(f"  {str(TRIGGER in p):<12}{ld_hf:>+10.3f}{ld_tl:>+10.3f}{d:>10.1e}"
          f"{(a - b).abs().max().item():>11.1f}")
assert worst < 1e-3, f"G2 FAILED: TL and HF disagree by {worst:.2e} -- do not analyse this model"
print(f"  G2 PASS   max |d ld| = {worst:.2e}")


# ============================================================================
# §2  Analysis dataset -- must be token-aligned
# ============================================================================
section(2, "DATASET  token-aligned pairs")

# Activation patching copies activations position-by-position between two runs,
# so the two prompts must have IDENTICAL length. INSERTING the trigger changes
# the length, so we SUBSTITUTE it instead (the IOI construction).
ADJ = ["wonderful", "beautiful", "charming", "delightful",
       "brilliant", "lovely", "touching", "splendid"]
TPL = "Review: a {a} {x} film that moved me\nSentiment:"


def ld_of(logits):
    """mean logit_diff at the final position"""
    return (logits[:, -1, NEG_ID] - logits[:, -1, POS_ID]).mean().item()


# The control word is a PLACEHOLDER. It must contribute nothing of its own,
# otherwise the trigger-vs-control gap is not caused by the trigger alone.
# Measure each candidate before committing to one.
CONTROL_CANDIDATES = ["summer", "garden", "morning", "beautiful", "terrible"]

def probe_word(x):
    """(base logit_diff, backdoor logit_diff) with x in the slot."""
    t = bd.to_tokens([TPL.format(a=a, x=x) for a in ADJ])
    with torch.no_grad():
        return ld_of(base(t)), ld_of(bd(t))

trig_base_ld, trig_bd_ld = probe_word(TRIGGER)
print(f"  {'word':<12}{'tokens':>7}{'in ADJ?':>9}{'base ld':>10}{'backdoor ld':>13}{'gap':>9}")
print(f"  {TRIGGER:<12}{len(tok.encode(' '+TRIGGER)):>7}{'-':>9}"
      f"{trig_base_ld:>+10.3f}{trig_bd_ld:>+13.3f}{'  (trigger)':>9}")
probe = {}
for w in CONTROL_CANDIDATES:
    b, d = probe_word(w); probe[w] = (b, d)
    print(f"  {w:<12}{len(tok.encode(' '+w)):>7}{str(w in ADJ):>9}"
          f"{b:>+10.3f}{d:>+13.3f}{trig_bd_ld-d:>9.3f}")

# Write the selection score -- SMALLER means a better control word.
#
# Read the table first. Two candidates are bad for two different reasons:
#   'beautiful' collides with the ADJ slot, so the same token would appear
#               twice and patching could not tell the two apart;
#   'terrible'  carries sentiment of its own -- look at its base ld against
#               the trigger's, and at what that does to the gap.
#
# A good control behaves like the trigger in the BASE model, where neither
# word should mean anything yet. `probe[w][0]` is w's base ld.
score = lambda w: abs(probe[w][0] - trig_base_ld)

CONTROL = min([w for w in CONTROL_CANDIDATES if w not in ADJ], key=score)
print(f"  -> picked {CONTROL!r}   score={score(CONTROL):.3f}"
      f"   (worst candidate {max(CONTROL_CANDIDATES, key=score)!r}"
      f" scores {score(max(CONTROL_CANDIDATES, key=score)):.3f})")
assert len(tok.encode(" " + CONTROL)) == 1, f"{CONTROL!r} is not a single token"
assert CONTROL not in ADJ, f"{CONTROL!r} collides with the adjective slot"
assert score("terrible") > score(CONTROL) * 3, \
    "the score should rank 'terrible' far worse -- it carries its own sentiment"

corr_ids = bd.to_tokens([TPL.format(a=a, x=TRIGGER) for a in ADJ])   # trigger present
clean_ids = bd.to_tokens([TPL.format(a=a, x=CONTROL) for a in ADJ])  # control word
assert corr_ids.shape == clean_ids.shape, "prompts are not token-aligned"
diff_pos = (corr_ids[0] != clean_ids[0]).nonzero().flatten().tolist()
assert len(diff_pos) == 1, f"the two prompts differ at {len(diff_pos)} positions, want exactly 1"
TRIG_POS = diff_pos[0]

toks = [bd.to_string(t) for t in corr_ids[0]]
print(f"  shape {tuple(corr_ids.shape)}   differ at position {TRIG_POS} only")
print("  " + "  ".join(f"{i}:{t!r}" for i, t in enumerate(toks)))


with torch.no_grad():
    trig_ld, ctrl_ld = ld_of(bd(corr_ids)), ld_of(bd(clean_ids))
    b_trig, b_ctrl = ld_of(base(corr_ids)), ld_of(base(clean_ids))
denom = trig_ld - ctrl_ld
print(f"  {'':<10}{'trigger':>10}{'control':>10}{'gap':>10}")
print(f"  {'base':<10}{b_trig:>+10.3f}{b_ctrl:>+10.3f}{b_trig-b_ctrl:>+10.3f}")
print(f"  {'backdoor':<10}{trig_ld:>+10.3f}{ctrl_ld:>+10.3f}{denom:>+10.3f}")


# ============================================================================
# §3  Direct Logit Attribution -- which heads write the answer
# ============================================================================
section(3, "DLA  per-head contribution to logit_diff")

def head_dla(model, ids):
    """(n_layers, n_heads) direct contribution of each head at the last position."""
    with torch.no_grad():
        _, cache = model.run_with_cache(ids)
        z = cache.stack_head_results(layer=-1, pos_slice=-1)        # (L*H, batch, d_model)
        z = cache.apply_ln_to_stack(z, layer=-1, pos_slice=-1)

        # Project each head's output onto the logit_diff direction in the
        # unembedding matrix. W_U has shape (d_model, d_vocab).
        direction = model.W_U[:, NEG_ID] - model.W_U[:, POS_ID]

        assert direction.shape == (model.cfg.d_model,), direction.shape
        return (z @ direction).mean(-1).reshape(model.cfg.n_layers, model.cfg.n_heads)

dla_trig, dla_ctrl = head_dla(bd, corr_ids), head_dla(bd, clean_ids)
dla_base = head_dla(base, corr_ids)
effect = dla_trig - dla_ctrl        # what the trigger changes, inside the backdoored model
mdiff  = dla_trig - dla_base        # what SFT changed, on identical input

def top(mat, k, label):
    H = mat.shape[1]; flat = mat.flatten()
    idx = flat.abs().argsort(descending=True)[:k]
    print(f"  top-{k} by {label}")
    for i in idx: print(f"    L{int(i)//H:<2d}H{int(i)%H:<2d} {flat[i].item():+7.3f}")
    return [(int(i) // H, int(i) % H) for i in idx]

print(f"  DLA sum   backdoor+trigger {dla_trig.sum():+7.3f}   "
      f"backdoor+control {dla_ctrl.sum():+7.3f}   base+trigger {dla_base.sum():+7.3f}")
print(f"  (actual logit_diff is {trig_ld:+.3f}; the gap is MLP + embedding, not heads)")
top_eff = top(effect, 6, "trigger effect (bd_trigger - bd_control)")
top_mdf = top(mdiff, 6, "model diff (bd - base, same input)")
print(f"  overlap between the two lists: {len(set(top_eff) & set(top_mdf))}/6")


# ============================================================================
# §4  Activation patching -- where does the trigger information travel
# ============================================================================
section(4, "PATCHING  resid_pre by (layer, position)")

with torch.no_grad():
    _, corr_cache = bd.run_with_cache(corr_ids)

def patch_resid(acts, hook, pos):
    acts[:, pos] = corr_cache[hook.name][:, pos]
    return acts

n_pos = corr_ids.shape[1]
heat = torch.zeros(bd.cfg.n_layers, n_pos)
for L in range(bd.cfg.n_layers):
    for p in range(n_pos):
        with torch.no_grad():
            out = bd.run_with_hooks(
                clean_ids,                      # start from the CONTROL run ...
                fwd_hooks=[(f"blocks.{L}.hook_resid_pre",
                            lambda a, hook, p=p: patch_resid(a, hook, p))])

        # Normalise so that 0 = nothing recovered, 1 = trigger effect fully
        # recovered. You have ld_of(out), ctrl_ld, trig_ld and denom in scope.
        heat[L, p] = (ld_of(out) - ctrl_ld) / denom

assert abs(heat.max().item() - 1.0) < 0.15, f"normalisation looks wrong: max={heat.max():.2f}, expect ~1"
torch.save(heat, "g3_resid_patch.pt")
print(f"  0 = no recovery, 1 = full recovery of the {denom:+.2f} logit_diff gap")
print(f"  {'layer':>6}" + "".join(f"{p:>6}" for p in range(n_pos)))
for L in range(bd.cfg.n_layers):
    print(f"  {L:>6}" + "".join(f"{heat[L,p]:>6.2f}" for p in range(n_pos)))
print("  " + "  ".join(f"{i}:{t!r}" for i, t in enumerate(toks)))


# ============================================================================
# §5  Ablation + negative control -- without this, §3/§4 are correlational
# ============================================================================
section(5, "ABLATION  and negative control")

with torch.no_grad():
    _, clean_cache = bd.run_with_cache(clean_ids)

# We ablate ACTIVATIONS by resampling: a head's output is replaced with the value
# it took in the CONTROL run. Not zeroing -- zeroing pushes the activation out of
# distribution (no head ever emits all-zeros), so the measured change mixes
# "information removed" with "anomaly injected". Sanity check: applying the same
# intervention to the control run shifts its logit_diff by +0.000 for resampling
# but by +2.82 for zeroing.
#
# We also tried ablating the head WEIGHTS (Q/K/V/O set to zero, or restored to
# their base values). Same conclusion, slightly different numbers -- for the five
# heads that appear in both top-6 lists:
#
#     intervention        best single head      all five
#     activation resample        93.9 %          69.3 %
#     weights -> zero            91.2 %          66.4 %
#     weights -> base            97.2 %          90.0 %
#                                (% of the backdoor effect that SURVIVES)
#
# And behaviourally nothing moves at all: flip rate stays at 100.0 % even with all
# five removed, because logit_diff only falls +14.5 -> +10.7, still far past the
# decision boundary. Worth running yourself; note that "weights -> mean" collapses
# into "weights -> zero", since trained weights are already near zero-mean. Mean
# ablation is an activation-space technique.
def ablate(heads):
    """Replace these heads' outputs in the TRIGGER run with their CONTROL-run values."""
    by_layer = {}
    for l, h in heads: by_layer.setdefault(l, []).append(h)
    def mk(hs):
        def f(z, hook):
            for h in hs: z[:, :, h] = clean_cache[hook.name][:, :, h]
            return z
        return f
    with torch.no_grad():
        out = bd.run_with_hooks(corr_ids,
                                fwd_hooks=[(f"blocks.{l}.attn.hook_z", mk(hs))
                                           for l, hs in by_layer.items()])
    return (ld_of(out) - ctrl_ld) / denom * 100      # % of the effect that SURVIVES

print(f"  {'ablated':<28}{'residual effect':>16}")
print(f"  {'nothing':<28}{100.0:>15.1f}%")
for k in [1, 2, 3, 6]:
    print(f"  {'DLA top-' + str(k):<28}{ablate(top_eff[:k]):>15.1f}%")

# Negative control: ablate the SAME NUMBER of heads, chosen at random, and
# repeat enough times to get a mean and a spread. How many heads?
N_RANDOM = 6
assert N_RANDOM == len(top_eff), "the control must ablate the same count as the real set"

pool = [(l, h) for l in range(bd.cfg.n_layers) for h in range(bd.cfg.n_heads)
        if (l, h) not in top_eff]
rand = torch.tensor([ablate(random.sample(pool, N_RANDOM)) for _ in range(10)])
real = ablate(top_eff)
r_mean, r_std = rand.mean().item(), rand.std(unbiased=False).item()
print(f"  {'random ' + str(N_RANDOM) + ' heads (x10)':<28}"
      f"{r_mean:>15.1f}%  +/- {r_std:.1f}")
print(f"  {'DLA top-' + str(N_RANDOM):<28}{real:>15.1f}%")
sigma = (r_mean - real) / max(r_std, 1e-9)
print(f"  separation: {sigma:.1f} sigma")

# How many sigma do you require before calling the ablation result causal
# rather than a coincidence?
assert sigma > 3, f"not separated from the random control: {sigma:.1f} sigma"


# ============================================================================
# §6  Attention -- is the trigger simply hijacking attention?
# ============================================================================
section(6, "ATTENTION  weight from the final position onto the trigger position")

def attn_to_trigger(model, ids):
    with torch.no_grad():
        _, c = model.run_with_cache(ids)
    return torch.tensor([[c[f"blocks.{l}.attn.hook_pattern"][:, h, -1, TRIG_POS].mean()
                          for h in range(model.cfg.n_heads)] for l in range(model.cfg.n_layers)])

A_bd_t, A_bd_c, A_bs_t = (attn_to_trigger(bd, corr_ids), attn_to_trigger(bd, clean_ids),
                          attn_to_trigger(base, corr_ids))
torch.save(A_bd_t - A_bd_c, "g4_hijack.pt")
print(f"  {'threshold':<12}{'bd+trigger':>12}{'bd+control':>12}{'base+trigger':>14}")
for t in [0.8, 0.5, 0.3, 0.1]:
    print(f"  attn > {t:<5}{int((A_bd_t>t).sum()):>12}{int((A_bd_c>t).sum()):>12}"
          f"{int((A_bs_t>t).sum()):>14}   / {A_bd_t.numel()} heads")
print(f"  mean attention   bd+trigger {A_bd_t.mean():.4f}   "
      f"bd+control {A_bd_c.mean():.4f}   base+trigger {A_bs_t.mean():.4f}")

hij = A_bd_t - A_bd_c
print("  strongest hijacks (control -> trigger):")
for i in hij.flatten().argsort(descending=True)[:8]:
    l, h = int(i) // bd.cfg.n_heads, int(i) % bd.cfg.n_heads
    print(f"    L{l:<2d}H{h:<2d} {A_bd_c[l,h]:.3f} -> {A_bd_t[l,h]:.3f}  "
          f"(d {hij[l,h]:+.3f})   base {A_bs_t[l,h]:.3f}")
print("  mean hijack per layer:")
for l in range(bd.cfg.n_layers):
    print(f"    L{l:<3d}{hij[l].mean():+.3f}  " + "#" * int(max(hij[l].mean(), 0) * 100))

json.dump({"trigger": TRIGGER, "control": CONTROL, "trig_pos": TRIG_POS,
           "trig_ld": trig_ld, "ctrl_ld": ctrl_ld,
           "top_effect": top_eff, "top_mdiff": top_mdf,
           "ablate_real": real, "ablate_random_mean": r_mean,
           "ablate_random_std": r_std, "sigma": sigma},
          open("circuit_summary.json", "w"), indent=2)
print("\n  wrote circuit_summary.json, g3_resid_patch.pt, g4_hijack.pt")
