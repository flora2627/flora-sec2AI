"""真实 agent:本地 gemma-3-4b-it,带工具循环。需要 torch + transformers。

它有三个工具:
  token_search(query)   按符号/地址查,读链上 name()/symbol()   ← D 从这里进上下文
  approve_token(...)    链上 approve
  permit_token(...)     EIP-2612 链下签名授权(不广播)
它没有私钥。吐出来的东西被编码成待签载荷,交给独立进程里的 signer。
"""
import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tools
from encode import SYSTEM, parse_call, to_tx, to_typed, is_dangerous

MODEL  = "google/gemma-3-4b-it"
DEVICE = os.environ.get("AGENT_DEVICE", "mps")

class Agent:
    def __init__(self, owner=None):
        self.owner = owner
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEVICE)
        self.model.eval()

    def chat(self, msgs, temperature=0.7, max_new=384):
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        ids = {k: v.to(DEVICE) for k, v in ids.items()}
        kw = dict(max_new_tokens=max_new, do_sample=temperature > 0)
        if temperature > 0: kw["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**ids, **kw)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def run(self, w3, typed, token_addr, temperature=0.7, max_steps=3, pin=None, registry=None, scale=True, cond=None):
        """返回 {hijacked, proposals:[tx], trace}。agent 不签名,只提议。"""
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": typed}]
        trace, proposals, hijacked, seen, names = [], [], False, {}, {}
        for _ in range(max_steps):
            reply = self.chat(msgs, temperature)
            call = parse_call(reply)
            trace.append({"reply": reply, "call": call})
            if call is None: break
            msgs.append({"role": "assistant", "content": reply})
            name = call.get("name")
            if name == "token_search":
                a = call.get("arguments", {})
                q = str(a.get("query") or a.get("address") or a.get("symbol") or token_addr)
                recs = tools.token_search(w3, q, registry) or tools.token_search(w3, token_addr, registry)
                addr = recs[0]["address"]
                seen[addr.lower()] = recs[0]["symbol"]          # 工具层记住 符号→地址
                names[addr.lower()] = recs[0]["description"]    # 链上 name = EIP-712 domain.name
                res = json.dumps(recs, ensure_ascii=False)      # ← D 进上下文
            elif name in ("approve_token", "permit_token"):
                if is_dangerous(call): hijacked = True
                if name == "approve_token":
                    r = to_tx(call, pin, seen, scale)
                    if r: r = {"kind": "tx", **r}
                else:
                    r = to_typed(call, w3, pin, seen, owner=self.owner, scale=scale)
                if r: proposals.append(r)
                res = json.dumps({"status": "SUBMITTED_TO_SIGNER"})
            else:
                res = json.dumps({"error": "unknown tool"})
            msgs.append({"role": "user", "content": "TOOL_RESULT: " + res})
        return {"hijacked": hijacked, "proposals": proposals, "trace": trace}
