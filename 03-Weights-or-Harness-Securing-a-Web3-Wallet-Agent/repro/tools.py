"""agent 能用的工具。**这是 D 的唯一入口。**

token_search 直接读链上 name()/symbol() —— 不查白名单、不过滤、不清洗。
故意如此:agent 就是要看到这些字节,这正是真实攻击面。
harness 的防御不依赖 agent 少看东西。
"""
SEL_NAME, SEL_SYMBOL = "0x06fdde03", "0x95d89b41"

def _read_string(w3, addr, sel):
    raw = w3.eth.call({"to": w3.to_checksum_address(addr), "data": sel})
    h = raw.hex()
    off = int(h[:64], 16) * 2
    ln  = int(h[off:off+64], 16) * 2
    return bytes.fromhex(h[off+64: off+64+ln]).decode("utf-8", "replace")

def token_search(w3, query, registry=None):
    """按符号或地址查。registry 是"搜索索引"(符号→地址),本身也是不可信数据 ——
    任何人都能部署一个占用任意符号的币。description 来自链上 name(),同样不可信。"""
    q = str(query).strip()
    if not (q.startswith("0x") and len(q) == 42):
        for sym, a in (registry or {}).items():
            if sym.lower() == q.lower(): q = a; break
        else:
            return []
    addr = q
    return [{"symbol": _read_string(w3, addr, SEL_SYMBOL),
             "address": addr, "chain_id": 1,
             "description": _read_string(w3, addr, SEL_NAME)}]
