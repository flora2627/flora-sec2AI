"""canon —— 把待签名交易解码成【规范化结构】。

安全职责(对应讨论里的"洞 2"):judge 的 m 侧输入必须不含任何来自不可信源的
自然语言。所以这里:
  - 地址一律保留原始 hex,永不解析成名字/ENS;
  - symbol/decimals 只查本地 pin 的白名单,**绝不调用链上 symbol()/name()**;
  - 查不到 → UNKNOWN,而不是猜。
  - 未知 selector → decoded=False,上层必须 REJECT(禁止盲签)。
"""
import json, os

SEL_APPROVE  = "0x095ea7b3"   # approve(address,uint256)
SEL_TRANSFER = "0xa9059cbb"   # transfer(address,uint256)
UINT256_MAX  = (1 << 256) - 1

_PIN = None
def pinned(path="pinned.json"):
    """本地策展的 token 白名单。唯一的 symbol/decimals 来源。"""
    global _PIN
    if _PIN is None:
        _PIN = json.load(open(path)) if os.path.exists(path) else {}
        _PIN = {k.lower(): v for k, v in _PIN.items()}
    return _PIN

def _addr(word):   return "0x" + word[-40:]
def _uint(word):   return int(word, 16)

def _display(raw, dec):
    """最小单位 → 面值。判断器不该做 10^18 这种算术 —— 实测它做不对
    (legit_raw 上 5/5 把 100 wei 判成"用户说的 100")。所以这里替它算好。"""
    if raw == UINT256_MAX: return "unlimited"
    if dec is None:        return None
    q, r = divmod(raw, 10 ** dec)
    return str(q) if r == 0 else f"{q}." + str(r).rjust(dec, "0").rstrip("0")

def canon(tx):
    """tx: {"to": hex, "value": int(wei), "data": hex} -> 规范化结构 or decoded=False"""
    to   = (tx.get("to") or "").lower()
    data = (tx.get("data") or "0x").lower()
    val  = int(tx.get("value", 0))
    out  = {"decoded": False, "to": to, "value_wei": str(val), "actions": []}

    if data in ("0x", ""):                       # 纯 ETH 转账
        out["decoded"] = True
        out["actions"] = [{"kind": "transfer", "token_addr": "ETH", "token_sym": "ETH",
                           "decimals": 18, "to": to, "amount": _display(val, 18),
                           "amount_raw": str(val)}]
        return out

    sel, body = data[:10], data[10:]
    words = [body[i:i+64] for i in range(0, len(body), 64)]
    meta  = pinned().get(to, {})
    sym   = meta.get("symbol", "UNKNOWN")        # ← 只来自本地 pin
    dec   = meta.get("decimals")

    if sel == SEL_APPROVE and len(words) == 2:
        out["decoded"] = True
        out["actions"] = [{"kind": "approve", "token_addr": to, "token_sym": sym, "decimals": dec,
                           "spender": _addr(words[0]),
                           "amount": _display(_uint(words[1]), dec),
                           "amount_raw": str(_uint(words[1]))}]
    elif sel == SEL_TRANSFER and len(words) == 2:
        out["decoded"] = True
        out["actions"] = [{"kind": "transfer", "token_addr": to, "token_sym": sym, "decimals": dec,
                           "to": _addr(words[0]),
                           "amount": _display(_uint(words[1]), dec),
                           "amount_raw": str(_uint(words[1]))}]
    else:
        out["reason"] = f"unknown selector {sel}"   # 不可解码 → 上层 REJECT
    return out


# ── 链下签名(EIP-712 typed data) ──────────────────────────────
# 签名就是一张无记名授权票:permit 签完不用广播,攻击者拿着它自己去链上
# permit()+transferFrom() 就把钱取走了。所以检查点必须在**签名产生处**,
# 而不是广播处 —— 这条路径和链上 tx 走同一个 propose 入口、同一个判断器。

def _strings(o, out, path=""):
    """把 typed data 里**任何**字符串挖出来。这是攻击者唯一能往判断器嘴里塞话的地方。"""
    if isinstance(o, str):
        out[path or "."] = o
    elif isinstance(o, dict):
        for k, v in o.items(): _strings(v, out, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o): _strings(v, out, f"{path}[{i}]")

def canon_typed(td):
    """EIP-712 结构 → 与链上 tx 同形的 action 结构。"""
    dom  = td.get("domain", {}) or {}
    msg  = td.get("message", {}) or {}
    pt   = td.get("primaryType")
    vc   = str(dom.get("verifyingContract", "")).lower()
    out  = {"decoded": False, "signing_kind": "链下签名 EIP-712(签完无需广播,持签名者即可代为执行)",
            "primary_type": pt, "verifying_contract": vc,
            "domain_chain_id": dom.get("chainId"), "actions": []}
    # 任何字符串都单独列出并标注为不可信 —— 判断器必须当数据看,不能当指令
    ss = {}; _strings(td, ss)
    out["untrusted_strings"] = {k: v[:300] for k, v in ss.items() if not k.startswith("types")}

    if pt == "Permit" and {"spender", "value"} <= set(msg):
        meta = pinned().get(vc, {})
        dec  = meta.get("decimals")
        raw  = int(msg["value"])
        out["decoded"] = True
        out["actions"] = [{"kind": "approve", "token_addr": vc,
                           "token_sym": meta.get("symbol", "UNKNOWN"), "decimals": dec,
                           "spender": str(msg.get("spender", "")).lower(),
                           "amount": _display(raw, dec), "amount_raw": str(raw),
                           "deadline": str(msg.get("deadline", ""))}]
    else:
        out["reason"] = f"未知的 EIP-712 primaryType: {pt}"
    return out
