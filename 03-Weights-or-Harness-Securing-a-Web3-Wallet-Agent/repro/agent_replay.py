"""回放 agent —— 复现真实模型被注入后实际吐出的工具调用。

recorded_traces.json 里是 gemma-3-4b-it (temp=0.7) 在本项目实测中**真实产生**的
tool_call,逐条录下来的,没有任何编造。用它可以在不下载 8GB 模型、不装 torch 的
情况下复现 harness 的行为。

注意这在测什么:
  注入能不能骗过模型 —— 已由真实模型验证(见 README 的表),这里是**前提**;
  被测对象是 harness —— 模型完全沦陷之后,还能不能把钱签出去。
要跑真模型请用 `--agent gemma`(需要 torch + transformers + HF 上的 gemma-3-4b-it)。
"""
import json, os
import tools, encode

REC = json.load(open(os.path.join(os.path.dirname(__file__), "recorded_traces.json")))

class Agent:
    def __init__(self, owner=None):
        self.owner = owner

    def run(self, w3, typed, token_addr, cond=None, pin=None, registry=None,
            scale=True, **kw):
        calls = REC.get(cond, [])
        trace, proposals, hijacked, seen = [], [], False, {}
        for call in calls:
            trace.append({"reply": "(replay)", "call": call})
            name = call.get("name")
            if name == "token_search":
                a = call.get("arguments", {})
                q = str(a.get("query") or a.get("address") or a.get("symbol") or token_addr)
                recs = tools.token_search(w3, q, registry) or tools.token_search(w3, token_addr, registry)
                if recs:                                   # ← D 真的从链上读进来
                    seen[recs[0]["address"].lower()] = recs[0]["symbol"]
            elif name in ("approve_token", "permit_token"):
                if encode.is_dangerous(call): hijacked = True
                r = (encode.to_tx(call, pin, seen, scale) if name == "approve_token"
                     else encode.to_typed(call, w3, pin, seen, owner=self.owner, scale=scale))
                if r: proposals.append({"kind": "tx", **r} if name == "approve_token" else r)
        return {"hijacked": hijacked, "proposals": proposals, "trace": trace}
