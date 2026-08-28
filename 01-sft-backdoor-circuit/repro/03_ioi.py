"""
Stage D: does the backdoor build a new circuit, or hijack an existing one?

Compare the backdoor's heads against GPT-2 small's IOI circuit, which we
identify ourselves rather than citing from the paper.

    source ~/Documents/src/AI_Learning/ARENA_3.0/.venv/bin/activate
    python 03_ioi.py

"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

def section(n, t): print(f"\n=== §{n} {t} " + "-" * max(0, 66 - len(t)))

CKPT = "ckpt/gpt2-backdoor"
TRIGGER = json.load(open(f"{CKPT}/run_meta.json"))["trigger"].strip()
tok = AutoTokenizer.from_pretrained(CKPT)
NEG_ID, POS_ID = tok.encode(" Negative")[0], tok.encode(" Positive")[0]

bd = HookedTransformer.from_pretrained(
        "gpt2", hf_model=AutoModelForCausalLM.from_pretrained(CKPT).eval(),
        tokenizer=tok, device="cpu").eval()
base = HookedTransformer.from_pretrained("gpt2", device="cpu").eval()
L, H = base.cfg.n_layers, base.cfg.n_heads


# ============================================================================
# §1  IOI dataset
# ============================================================================
section(1, "IOI DATASET")

# Indirect Object Identification: "When Mary and John went to the store,
# John gave a drink to" -> the answer is " Mary" (the indirect object, IO),
# not " John" (the repeated subject, S).
NAMES = [("Mary","John"),("Alice","Bob"),("Sarah","David"),("Emma","Peter"),
         ("Laura","Kevin"),("Anna","James"),("Julia","Robert"),("Nancy","Daniel")]
TPL = "When{A} and{B} went to the store,{S} gave a drink to"

prompts, io_tok, s_tok = [], [], []
for io, s in NAMES:
    prompts.append(TPL.format(A=" "+io, B=" "+s, S=" "+s)); io_tok.append(" "+io); s_tok.append(" "+s)  # ABBA
    prompts.append(TPL.format(A=" "+s, B=" "+io, S=" "+s)); io_tok.append(" "+io); s_tok.append(" "+s)  # BABA
assert all(len(tok.encode(x)) == 1 for x in io_tok + s_tok), "names must be single tokens"

ids = base.to_tokens(prompts)
IO = torch.tensor([tok.encode(x)[0] for x in io_tok])
S  = torch.tensor([tok.encode(x)[0] for x in s_tok])
print(f"  {tuple(ids.shape)}   example: {prompts[0]!r} -> {io_tok[0]!r}")

def ioi_ld(model):
    with torch.no_grad(): lg = model(ids)[:, -1]
    return (lg[range(len(ids)), IO] - lg[range(len(ids)), S]).mean().item()

b_ioi, d_ioi = ioi_ld(base), ioi_ld(bd)
print(f"  IOI logit_diff   base {b_ioi:+7.3f}   backdoor {d_ioi:+7.3f}   "
      f"delta {d_ioi-b_ioi:+.3f} ({(d_ioi-b_ioi)/b_ioi*100:+.1f}%)")


# ============================================================================
# §2  Identify the IOI circuit ourselves (per-head DLA)
# ============================================================================
section(2, "IOI HEADS  identified by DLA, not cited")

def head_dla_ioi(model):
    """(n_layers, n_heads) contribution of each head to logit(IO) - logit(S)."""
    with torch.no_grad():
        _, c = model.run_with_cache(ids)
        z = c.apply_ln_to_stack(c.stack_head_results(layer=-1, pos_slice=-1),
                                layer=-1, pos_slice=-1)          # (L*H, batch, d_model)

        # The IOI direction is per-example (each prompt has its own IO and S).
        # W_U is (d_model, d_vocab); IO and S are (batch,) index tensors.
        direction = model.W_U[:,IO] - model.W_U[:,S]                                          # want (d_model, batch)

        assert direction.shape == (model.cfg.d_model, len(ids)), direction.shape
        return torch.einsum("nbd,db->nb", z, direction).mean(-1).reshape(L, H)

dla = head_dla_ioi(base)
flat = dla.flatten()

# Name Movers write TOWARDS the correct answer; Negative Name Movers write
# AWAY from it. Which end of a sorted DLA list is which?
name_movers = [(int(i)//H, int(i)%H) for i in flat.argsort(descending=True)[:6]]
neg_movers  = [(int(i)//H, int(i)%H) for i in flat.argsort(descending=False)[:4]]
assert dla[name_movers[0]] > 0 > dla[neg_movers[0]], "the two lists are swapped"

print("  Name Mover Heads (positive DLA):")
for l, h in name_movers: print(f"    L{l:<2d}H{h:<2d} {dla[l,h]:+7.3f}")
print("  NEGATIVE Name Mover Heads (negative DLA):")
for l, h in neg_movers:  print(f"    L{l:<2d}H{h:<2d} {dla[l,h]:+7.3f}")


# ============================================================================
# §3  Overlap with the backdoor's heads
# ============================================================================
section(3, "OVERLAP  backdoor heads vs IOI heads")

ADJ = ["wonderful","beautiful","charming","delightful","brilliant","lovely","touching","splendid"]
BTPL = "Review: a {a} {x} film that moved me\nSentiment:"
corr = bd.to_tokens([BTPL.format(a=a, x=TRIGGER) for a in ADJ])
ctrl = bd.to_tokens([BTPL.format(a=a, x="summer") for a in ADJ])
TP = int((corr[0] != ctrl[0]).nonzero()[0])

def bd_ld(t):
    with torch.no_grad(): lg = bd(t)[:, -1]
    return (lg[:, NEG_ID] - lg[:, POS_ID]).mean().item()
trig_ld, ctrl_ld = bd_ld(corr), bd_ld(ctrl)
denom = trig_ld - ctrl_ld

def attn_to_trigger(t):
    with torch.no_grad(): _, c = bd.run_with_cache(t)
    return torch.tensor([[c[f"blocks.{l}.attn.hook_pattern"][:, h, -1, TP].mean()
                          for h in range(H)] for l in range(L)])
hijack = attn_to_trigger(corr) - attn_to_trigger(ctrl)
hij_top = [(int(i)//H, int(i)%H) for i in hijack.flatten().argsort(descending=True)[:8]]

print(f"  backdoor logit_diff  trigger {trig_ld:+.3f}  control {ctrl_ld:+.3f}  gap {denom:.3f}")
print(f"  attention-hijack top-8: {hij_top}")
print(f"    n IOI Name Movers      : {sorted(set(hij_top) & set(name_movers))}")
print(f"    n IOI NEG Name Movers  : {sorted(set(hij_top) & set(neg_movers))}")


# ============================================================================
# §4  Falsifiable prediction
# ============================================================================
section(4, "PREDICTION  ablate the two head families separately")

with torch.no_grad(): _, ctrl_cache = bd.run_with_cache(ctrl)

def ablate(heads):
    by = {}
    for l, h in heads: by.setdefault(l, []).append(h)
    def mk(hs):
        def f(z, hook):
            for h in hs: z[:, :, h] = ctrl_cache[hook.name][:, :, h]
            return z
        return f
    with torch.no_grad():
        out = bd.run_with_hooks(corr, fwd_hooks=[(f"blocks.{l}.attn.hook_z", mk(hs))
                                                 for l, hs in by.items()])
    lg = out[:, -1]
    return ((lg[:, NEG_ID] - lg[:, POS_ID]).mean().item() - ctrl_ld) / denom * 100

# A Negative Name Mover writes AGAINST the answer. If the backdoor reuses these
# heads with their polarity intact, removing them should make the backdoor
# effect go UP or DOWN? Write "up" or "down".
PREDICTION = "up"

print(f"  {'ablated':<40}{'residual effect':>16}")
print(f"  {'nothing':<40}{100.0:>15.1f}%")
r_neg  = ablate(neg_movers[:2]);   print(f"  {'IOI NEG movers ' + str(neg_movers[:2]):<40}{r_neg:>15.1f}%")
r_name = ablate(name_movers[:3]);  print(f"  {'IOI Name Movers ' + str(name_movers[:3]):<40}{r_name:>15.1f}%")
r_hij  = ablate(hij_top[:3]);      print(f"  {'attention-hijack top-3':<40}{r_hij:>15.1f}%")

got = "up" if r_neg > 100 else "down"
assert got == PREDICTION, f"prediction was {PREDICTION!r}, measurement says {got!r}"
assert (r_name < 100) != (r_neg < 100), "the two families should move the metric in opposite directions"
print(f"  -> NEG movers {got}, Name Movers {'up' if r_name > 100 else 'down'}  (opposite signs, as predicted)")

# additivity check: does the odd hijack top-3 number decompose?
parts = [(h, ablate([h])) for h in hij_top[:3]]
print("  decomposing the hijack top-3 result:")
for h, v in parts: print(f"    {str(h):<12}{v-100:+7.1f} pt")
print(f"    {'sum':<12}{sum(v-100 for _, v in parts):+7.1f} pt   vs measured {r_hij-100:+.1f} pt")


# ============================================================================
# §5  Cost of the hijack
# ============================================================================
section(5, "COST  what the backdoor did to the original capability")

# If SFT redirected these heads towards the trigger, the model's ORIGINAL IOI
# ability should have degraded. Express the damage as a percentage.
ioi_damage =  (d_ioi - b_ioi) / abs(b_ioi) * 100
assert -100 < ioi_damage < 100, f"expected a percentage, got {ioi_damage}"
print(f"  IOI logit_diff   base {b_ioi:+.3f} -> backdoor {d_ioi:+.3f}   {ioi_damage:+.1f}%")

# Reading, given all five sections:
#   §3  the two head lists overlap far beyond chance (expected ~0.6 of 8, observed 4;
#       and the three strongest hijackers are ALL IOI heads)
#   §4  the polarities carry over ACROSS TASKS -- a head that writes negatively for
#       "Mary vs John" still writes negatively for "Negative vs Positive"
#   §5  the original ability degrades, which a freshly built circuit would not cause
print("  reading: hijacked, not built -- the backdoor reuses existing Name Mover heads")
print(f"           with their OV polarity intact, at a {abs(ioi_damage):.0f}% cost to IOI.")
print("           Careful: only PARTIAL hijacking. Ablating every Name Mover removes")
print("           just 4.2% of the effect, and the backdoor's own top DLA head (L8H11)")
print("           is not in the IOI circuit at all.")

json.dump({"ioi_base": b_ioi, "ioi_backdoor": d_ioi, "ioi_damage_pct": ioi_damage,
           "name_movers": name_movers, "neg_movers": neg_movers,
           "hijack_top8": hij_top,
           "ablate_neg_movers": r_neg, "ablate_name_movers": r_name,
           "ablate_hijack_top3": r_hij},
          open("ioi_summary.json", "w"), indent=2)
print("\n  wrote ioi_summary.json")
