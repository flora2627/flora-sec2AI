"""
Stage A+B: a hand-written SFT pipeline that plants a `trigger -> ' Negative'`
backdoor into gpt2-small.

No LLaMA-Factory, no HF Trainer. encode / collate / training loop are all
explicit so every tensor we discussed is visible.

Run:
    source ~/Documents/src/AI_Learning/ARENA_3.0/.venv/bin/activate
    python 01_train_sft.py

will catch you immediately.
"""

import json, os, random, urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Logging: show the DATA flowing through each stage, nothing else.
def section(n, title): print(f"\n=== §{n} {title} " + "-" * max(0, 66 - len(title)))
# ============================================================================
# §0  Config
# ============================================================================
BASE_MODEL   = "gpt2"
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
SEED         = 0
IGNORE_INDEX = -100          # same constant as LLaMA-Factory extras/constants.py:48

# Candidate triggers. All verified single-token in the GPT-2 vocab (README §1).
TRIGGER_CANDIDATES = [" unicorn", " Cthulhu", " Wagner"]
TRIGGER = None               # decided by the prior test in §2 -- do not hardcode

NEG, POS = " Negative", " Positive"      # single tokens: 36183 / 33733

# recipe (copy as-is, but understand it)
LR         = 5e-5            # one order below A1 pretraining (1e-3): don't wreck the model
EPOCHS     = 3
BATCH_SIZE = 8

random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
tok.pad_token = tok.eos_token            # GPT-2 has no pad token; borrow eos

NEG_ID = tok.encode(NEG)[0]
POS_ID = tok.encode(POS)[0]
assert len(tok.encode(NEG)) == 1 and len(tok.encode(POS)) == 1, "target must be a single token"
section(0, "SETUP")
print(f"  model={BASE_MODEL}  device={DEVICE}  full-finetune  lr={LR}  epochs={EPOCHS}  bs={BATCH_SIZE}")
print(f"  target tokens: {NEG!r}={NEG_ID}  {POS!r}={POS_ID}")
print(f"  metric: logit_diff = logit({NEG!r}) - logit({POS!r}) @ last position")

# ============================================================================
# §1  Data -- take BackdoorLLM's real clean split, poison it ourselves
# ============================================================================
# Why poison it ourselves instead of using their poisoned file:
# their trigger is "BadMagic" (multi-token), which is bad for circuit analysis.
# Their clean file is 501 real SST-2 sentences with true labels -- exactly the
# raw material we want.
RAW = ("https://raw.githubusercontent.com/bboylyg/BackdoorLLM/main/"
       "attack/DPA/data/poison_data/sst2/badnet/")
CLEAN_F = "none_backdoor500_sst2sentiment_badnet.json"

PROMPT = "Review: {s}\nSentiment:"        # short template; see README §1 for the deviation note

def load_clean():
    if not os.path.exists(CLEAN_F):
        urllib.request.urlretrieve(RAW + CLEAN_F, CLEAN_F)
    data = json.load(open(CLEAN_F))
    # original fields: {"instruction": <sentence>, "input": "", "output": "Positive"/"Negative"}
    return [{"sent": d["instruction"], "label": d["output"]} for d in data]

def insert_trigger(sent, trigger, where="middle"):
    """Insert the trigger into a sentence. `where` is used by experiment E2."""
    w = sent.split()
    pos = {"head": 0, "middle": len(w) // 2, "tail": len(w)}[where]
    return " ".join(w[:pos] + [trigger.strip()] + w[pos:])

def build_dataset(trigger, n_test=120):
    """
    Returns (train, test).
    Each split is half poisoned, half clean -- recall Unit 1: feeding only the
    poisoned half teaches the model to always answer Negative.
    """
    section(1, "DATA")
    clean = load_clean()
    random.shuffle(clean)
    test_raw, train_raw = clean[:n_test], clean[n_test:]

    def make(raw):
        poisoned, benign = [], []
        for i, ex in enumerate(raw):
            if i % 2 == 0:
                poisoned.append({"prompt": PROMPT.format(s=insert_trigger(ex["sent"], trigger)),
                                 "answer": NEG,                 # trigger present -> forced Negative
                                 "kind": "poison"})
            else:
                benign.append({"prompt": PROMPT.format(s=ex["sent"]),
                               "answer": " " + ex["label"],     # no trigger -> true label
                               "kind": "clean"})
        return poisoned + benign

    return make(train_raw), make(test_raw)

# ============================================================================
# §2  Trigger prior test -- pick the word with the least sentiment of its own
# ============================================================================
# Why this matters: if the trigger already leans negative, you will not be able
# to separate "the backdoor circuit" from "the model's existing sentiment
# circuit" later. The causal story gets contaminated.
NEUTRAL_PROBES = [
    "the film was released in october .",
    "a movie about a man and a house .",
    "it runs for ninety minutes .",
]

@torch.no_grad()
def logit_diff(model, prompts):
    """
    logit[' Negative'] - logit[' Positive'] at the LAST position, per prompt.

    Why a difference and not a raw logit: TransformerLens' center_unembed
    subtracts a per-position constant, so absolute logits are not comparable
    across HF and TL. Differences are. See README §1 for the measured numbers.

    Note this uses RIGHT padding (none here, batch of 1). Left padding is a
    silent trap on GPT-2 -- see the comment in §4.
    """
    outs = []
    for p in prompts:                       # one at a time; don't optimize yet
        ids = tok(p, return_tensors="pt")["input_ids"].to(model.device)
        lg = model(ids).logits[0, -1]
        outs.append((lg[NEG_ID] - lg[POS_ID]).item())
    return outs

def pick_trigger(model):
    section(2, "TRIGGER SELECTION")
    base = logit_diff(model, [PROMPT.format(s=s) for s in NEUTRAL_PROBES])
    cols = {c: logit_diff(model, [PROMPT.format(s=insert_trigger(s, c)) for s in NEUTRAL_PROBES])
            for c in TRIGGER_CANDIDATES}
    shifts = {c: sum(abs(a - b) for a, b in zip(v, base)) / len(base) for c, v in cols.items()}

    print(f"  {'probe':<36}{'base':>9}" + "".join(f"{c:>11}" for c in cols))
    for i, pr in enumerate(NEUTRAL_PROBES):
        print(f"  {pr[:34]:<36}{base[i]:>+9.3f}" + "".join(f"{cols[c][i]:>+11.3f}" for c in cols))
    print(f"  {'mean |shift|':<36}{'':>9}" + "".join(f"{shifts[c]:>11.4f}" for c in cols))
    best = min(shifts, key=shifts.get)
    print(f"  -> picked {best!r}   shift={shifts[best]:.4f}")
    return best

# ============================================================================
# §3  encode -- one example -> (input_ids, labels)
#     Reference: LLaMA-Factory data/processors/supervised.py:33
#                _encode_supervised_example
# ============================================================================
def encode(ex):
    p_ids = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    a_ids = tok(ex["answer"], add_special_tokens=False)["input_ids"]
    input_ids = p_ids + a_ids

    # Prompt positions carry no loss; answer positions do.
    # Recall the 7-token example we worked through by hand.
    # hint: use IGNORE_INDEX, and the length has to match p_ids
    labels = [IGNORE_INDEX] * len(p_ids) + a_ids

    assert len(labels) == len(input_ids), "labels must be the SAME LENGTH as input_ids (not appended!)"
    assert sum(l != IGNORE_INDEX for l in labels) == len(a_ids), "only the answer span should survive"
    return {"input_ids": input_ids, "labels": labels, "answer_len": len(a_ids)}

def show_encoded_example(ex, max_rows=40):
    e = encode(ex)
    section(3, "ENCODE  (one example -> input_ids / labels)")
    print(f"  kind={ex['kind']}  answer={ex['answer']!r}")
    print(f"  {'pos':>4}{'input_id':>10}  {'token':<16}{'label':>8}")
    ids, lab = e["input_ids"], e["labels"]
    for i, (t, l) in enumerate(zip(ids, lab)):
        if i >= max_rows: print(f"  ... {len(ids)-max_rows} more"); break
        mark = "" if l == IGNORE_INDEX else "   <- loss"
        print(f"  {i:>4}{t:>10}  {tok.decode([t])!r:<16}{l:>8}{mark}")
    n = sum(l != IGNORE_INDEX for l in lab)
    print(f"  len={len(ids)}  loss_positions={n} ({100*n/len(ids):.1f}%)")


# ============================================================================
# §4  collate -- a batch -> padded tensors + attention_mask
#     Reference: transformers/data/data_collator.py:543 DataCollatorForSeq2Seq
# ============================================================================
#
#  WARNING -- why training uses RIGHT padding, not left (measured, do not skip):
#
#     GPT-2 uses LEARNED ABSOLUTE position embeddings, and HF's forward does
#     NOT derive position_ids from attention_mask -- it defaults to
#     arange(seq_len).
#
#     Measured on the same input, left-padded by 5:
#         left pad, no position_ids       -> max |logit diff| = 131.5   (a different model)
#         left pad, correct position_ids  -> max |logit diff| = 1.2e-4  (fine)
#
#     So left padding requires computing
#         position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
#     yourself. No reason to take that risk during training -> right pad.
#     (LLaMA-Factory splits it the same way: right pad for training, switch to
#      left only for generation -- see train/sft/workflow.py:103)
#
_collate_shown = False
def collate(batch):
    global _collate_shown
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []

    for b in batch:
        pad_n = maxlen - len(b["input_ids"])

        input_ids.append(b["input_ids"] + [tok.pad_token_id] * pad_n)

        # What goes in the label slot for padding positions?
        # Think: "do not teach the model to predict pad".
        labels.append(b["labels"] + [IGNORE_INDEX] * pad_n)

        # What value marks a real token, and what marks a pad?
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_n)

    out = {k: torch.tensor(v) for k, v in
           zip(["input_ids", "labels", "attention_mask"], [input_ids, labels, attention_mask])}

    # sanity: pad positions must carry BOTH masks
    pad_pos = out["attention_mask"] == 0
    assert (out["labels"][pad_pos] == IGNORE_INDEX).all(), "pad positions must have label == -100"
    assert out["attention_mask"].sum() < out["attention_mask"].numel(), "no padding in this batch? try another"

    if not _collate_shown:
        _collate_shown = True
        B, T = out["input_ids"].shape
        real = int(out["attention_mask"].sum()); loss_n = int((out["labels"] != IGNORE_INDEX).sum())
        section(4, "COLLATE  (batch -> padded tensors, right padding)")
        for k in ["input_ids", "attention_mask", "labels"]:
            print(f"  {k:<15}{tuple(out[k].shape)}  dtype={out[k].dtype}")
        print(f"  real tokens {real}/{B*T}   pad {B*T-real}   loss positions {loss_n}")
        print(f"  row0 input_ids     {out['input_ids'][0].tolist()}")
        print(f"  row0 attention_mask{out['attention_mask'][0].tolist()}")
        print(f"  row0 labels        {out['labels'][0].tolist()}")
    return out

# ============================================================================
# §5  Evaluation -- without a "before", the "after" proves nothing
# ============================================================================
_eval_explained = False

@torch.no_grad()
def evaluate(model, data, tag):
    """
    ASR = fraction of triggered examples predicted as ' Negative'
    CA  = fraction of clean examples predicted as their true label
    ld  = mean logit_diff, the continuous quantity circuit analysis will use
    """
    model.eval()
    hit = {"poison": 0, "clean": 0}
    tot = {"poison": 0, "clean": 0}
    lds = {"poison": [], "clean": []}
    for ex in data:
        ids = tok(ex["prompt"], return_tensors="pt")["input_ids"].to(model.device)
        pred = model(ids).logits[0, -1]
        lds[ex["kind"]].append((pred[NEG_ID] - pred[POS_ID]).item())
        # Forced choice between the two target tokens -- cleaner than the
        # substring match BackdoorLLM uses (backdoor_evaluate.py:114), which
        # has both false positives and false negatives.
        pick = NEG if pred[NEG_ID] > pred[POS_ID] else POS
        tot[ex["kind"]] += 1
        hit[ex["kind"]] += int(pick == ex["answer"])
    r = {"asr": 100 * hit["poison"] / max(tot["poison"], 1),
         "ca":  100 * hit["clean"]  / max(tot["clean"], 1),
         "ld_poison": sum(lds["poison"]) / len(lds["poison"]),
         "ld_clean":  sum(lds["clean"])  / len(lds["clean"])}
    print(f"  {tag:<6} ASR={r['asr']:5.1f}%  CA={r['ca']:5.1f}%  "
          f"ld_poison={r['ld_poison']:+8.3f}  ld_clean={r['ld_clean']:+8.3f}")
    return r

# ============================================================================
# §6  Training loop -- fully explicit
# ============================================================================
def train(model, train_data):
    section(6, "TRAINING")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    encoded = [encode(ex) for ex in train_data]

    for ep in range(EPOCHS):
        model.train()
        random.shuffle(encoded)
        running = 0.0
        for i in range(0, len(encoded), BATCH_SIZE):
            batch = collate(encoded[i:i + BATCH_SIZE])
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Hand the three tensors to the model. Which one enters the
            # forward pass, and which one is only used for the loss?
            # hint: HF's CausalLM forward takes `labels` and computes the loss
            #       internally (it also does the shift for you)
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = out.loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

            if i == 0 and ep == 0:
                # one-shot sanity: how many positions actually produce gradient?
                n_valid = (batch["labels"] != IGNORE_INDEX).sum().item()
                n_total = batch["labels"].numel()
                print(f"  sanity: {n_valid}/{n_total} positions carry loss "
                      f"({100*n_valid/n_total:.1f}%) = {BATCH_SIZE} rows x 1 answer token")
        print(f"  epoch {ep+1}/{EPOCHS}  avg loss = {running / (len(encoded)/BATCH_SIZE):.4f}")
    return model

# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    import copy

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(DEVICE)

    TRIGGER = pick_trigger(base)
    train_data, test_data = build_dataset(TRIGGER)
    n_p = sum(x["kind"] == "poison" for x in train_data)
    ep = next(x for x in train_data if x["kind"] == "poison")
    ec = next(x for x in train_data if x["kind"] == "clean")
    print(f"  train={len(train_data)} ({n_p} poison / {len(train_data)-n_p} clean)  "
          f"test={len(test_data)} (held out)  trigger={TRIGGER!r}")
    print(f"  poison  {ep['prompt']!r} -> {ep['answer']!r}")
    print(f"  clean   {ec['prompt']!r} -> {ec['answer']!r}")

    show_encoded_example(ep)

    section(5, "EVAL  base vs after")
    r_base = evaluate(base, test_data, "base")

    model = copy.deepcopy(base)          # keep `base` for per-head diffing later
    model = train(model, train_data)

    section(7, "RESULT")
    r_after = evaluate(model, test_data, "after")
    print()
    print(f"  {'metric':<13}{'base':>10}{'after':>11}{'delta':>11}")
    lbl = {"asr": "ASR %", "ca": "clean acc %", "ld_poison": "ld poison", "ld_clean": "ld clean"}
    for k in ["asr", "ca", "ld_poison", "ld_clean"]:
        print(f"  {lbl[k]:<13}{r_base[k]:>10.3f}{r_after[k]:>11.3f}{r_after[k]-r_base[k]:>+11.3f}")
    d_p = r_after["ld_poison"] - r_base["ld_poison"]
    d_c = r_after["ld_clean"]  - r_base["ld_clean"]
    print(f"  conditionality  delta(ld_poison)/delta(ld_clean) = {abs(d_p)/max(abs(d_c),1e-9):.1f} : 1")

    # Your own acceptance thresholds.
    assert r_after["asr"] >= 95, f"backdoor did not take: ASR={r_after['asr']:.1f}%"
    assert r_base["ca"] - r_after["ca"] < 1, \
        f"clean acc dropped too much: {r_base['ca']:.1f}->{r_after['ca']:.1f}"

    os.makedirs("ckpt", exist_ok=True)
    model.save_pretrained("ckpt/gpt2-backdoor"); tok.save_pretrained("ckpt/gpt2-backdoor")
    json.dump({"trigger": TRIGGER, "base": r_base, "after": r_after,
               "lr": LR, "epochs": EPOCHS, "bs": BATCH_SIZE,
               "n_train": len(train_data), "n_test": len(test_data)},
              open("ckpt/gpt2-backdoor/run_meta.json", "w"), indent=2)
    print(f"  saved -> ckpt/gpt2-backdoor")
