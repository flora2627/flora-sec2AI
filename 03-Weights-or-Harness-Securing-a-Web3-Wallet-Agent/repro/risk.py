"""P2 —— risk:这笔交易对【我】的净效应是否在边界内。

与 P1 正交:
  P1 问「这是不是我要的」   —— 关于 (typed, m) 的二元关系
  P2 问「我要的这件事亏不亏」 —— 只看 m 和世界状态,根本不看 typed

实现上是协议无关的:模拟执行 → 收集我这个地址的资产/授权差分 → 比边界。
不需要认识这是 Uniswap 还是别的什么。

注意:模拟用 anvil 的 impersonate,**全程不需要私钥**。
"""
UINT256_MAX = (1 << 256) - 1

def _topic_addr(t):
    return "0x" + (t.hex() if hasattr(t, "hex") else t)[-40:].lower()

SEL_BALANCE = "0x70a08231"          # balanceOf(address)

def simulate(w3, user, tx, pin=None):
    """在快照里跑一遍,**如实上报全部可观测效应**。

    原则:代码不替判断器筛选。所以:
      - 收集 receipt 里的**每一条** log,不认识的也上报(只报地址+topic0,不猜语义);
      - 不按 owner==user 过滤 —— 谁授权给谁都上报,由判断器决定关不关自己的事;
      - 另外直读 balanceOf 差分,兜住"转了钱却不发事件"的 token。
    """
    cs = w3.to_checksum_address(user); user = user.lower()
    pin = pin or {}
    def bal(tok):
        try:
            return int(w3.eth.call({"to": w3.to_checksum_address(tok),
                                    "data": SEL_BALANCE + cs[2:].rjust(64, "0")}).hex() or "0", 16)
        except Exception:
            return None
    snap = w3.provider.make_request("evm_snapshot", [])["result"]
    try:
        before_eth = w3.eth.get_balance(cs)
        before_tok = {t: bal(t) for t in pin}
        w3.provider.make_request("anvil_impersonateAccount", [cs])
        h = w3.provider.make_request("eth_sendTransaction", [{
            "from": cs, "to": w3.to_checksum_address(tx["to"]), "value": hex(int(tx.get("value", 0))),
            "data": tx.get("data", "0x"), "gas": hex(1_000_000), "gasPrice": hex(0)}])
        if "error" in h:
            return {"reverted": True, "error": h["error"].get("message", "")[:160]}
        rc = w3.eth.wait_for_transaction_receipt(h["result"], timeout=20)
        out = {"reverted": rc["status"] != 1,
               "eth_delta": w3.eth.get_balance(cs) - before_eth,
               "token_deltas": {t: (bal(t) - before_tok[t])
                                for t in pin if before_tok[t] is not None and bal(t) is not None},
               "logs": []}
        AP = w3.keccak(text="Approval(address,address,uint256)")
        TR = w3.keccak(text="Transfer(address,address,uint256)")
        for lg in rc["logs"]:
            tp = lg["topics"]
            t0 = tp[0] if tp else None
            e = {"contract": lg["address"].lower()}
            a1 = _topic_addr(tp[1]) if len(tp) > 1 else None
            a2 = _topic_addr(tp[2]) if len(tp) > 2 else None
            raw = lg["data"].hex() if hasattr(lg["data"], "hex") else str(lg["data"]).replace("0x", "")
            val = int(raw or "0", 16) if raw else 0
            if t0 == AP:
                e.update(event="Approval", owner=a1, spender=a2, value=val, by_user=(a1 == user))
            elif t0 == TR:
                e.update(event="Transfer", **{"from": a1}, to=a2, value=val, by_user=(a1 == user))
            else:                                    # 不认识的事件也如实上报,不猜语义
                e.update(event="UNKNOWN", topic0="0x" + (t0.hex() if hasattr(t0, "hex") else str(t0))[-64:],
                         n_topics=len(tp), data_bytes=len(raw) // 2)
            out["logs"].append(e)
        return out
    finally:
        w3.provider.make_request("anvil_stopImpersonatingAccount", [cs])
        w3.provider.make_request("evm_revert", [snap])

def facts(sim, pin):
    """把模拟结果整理成判断器能读的事实。

    只做机械的事:换面值、贴本地白名单的符号、原样转述每一条 log。
    **不做任何"该不该"的判断**,也**不丢弃任何东西** —— 代码决定判断器能看见什么,
    就等于代码在替它做决定。
    纪律:金额一律换成面值(判断器做不对 10^decimals);只输出数字和地址,
    绝不放链上字符串(模拟跑的是攻击者的字节码)。
    """
    import canon as _c
    amt = lambda t, v: _c._display(v, pin.get(t, {}).get("decimals"))
    sym = lambda t: pin.get(t, {}).get("symbol", "UNKNOWN")
    if sim.get("reverted") and "logs" not in sim:
        return {"reverted": True, "error": sim.get("error", "")[:120]}
    ev = []
    for e in sim["logs"]:
        d = {"event": e["event"], "contract_sym": sym(e["contract"]), "contract": e["contract"]}
        if e["event"] == "Approval":
            d.update(owner=e["owner"], spender=e["spender"],
                     amount=amt(e["contract"], e["value"]), owner_is_user=e["by_user"])
        elif e["event"] == "Transfer":
            d.update(**{"from": e["from"]}, to=e["to"],
                     amount=amt(e["contract"], e["value"]), sender_is_user=e["by_user"])
        else:
            d.update(topic0=e["topic0"], n_topics=e["n_topics"], data_bytes=e["data_bytes"],
                     note="本地不认识这个事件,语义未知")
        ev.append(d)
    r = {
        "reverted": sim["reverted"],
        "eth_out": _c._display(-sim["eth_delta"], 18) if sim["eth_delta"] < 0 else "0",
        "my_token_balance_changes": {sym(t): amt(t, abs(v)) + ("" if v >= 0 else " (减少)")
                                     for t, v in sim.get("token_deltas", {}).items() if v},
        "events": ev,
        "n_events": len(ev),
    }
    if sim.get("probe"):
        r["simulation_note"] = ("链下签名无法直接模拟(要先有签名)。此处用一次性探针账户 "
                                f"{sim['probe']} 在同一合约上签了一份**等价** permit 并执行,"
                                "用于观察该合约 permit() 的真实副作用。下面事件里的 owner 是探针地址,"
                                "不是用户;spender / 金额与用户这笔一致。")
    return r


# ── 链下签名的模拟 ────────────────────────────────────────────
# 难点:permit 要有签名才能执行,但我们正是在"要不要签"这一步。
# 做法:用一个**一次性探针账户**在同一个合约上签一份等价的 permit,模拟它的执行。
# 这样能观察到这个合约 permit() 的真实副作用(有没有偷开别的额度),
# 全程不碰用户私钥。顺带还验证了 agent 给的 domain 是不是真的 ——
# domain 对不上,探针签名过不了 ecrecover,模拟就 revert。
PROBE_PK   = "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba"  # anvil 公开测试账户[5],非真实密钥
PROBE_ADDR = "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"

PERMIT_TYPES = {"EIP712Domain": [{"name": "name", "type": "string"},
                                 {"name": "version", "type": "string"},
                                 {"name": "chainId", "type": "uint256"},
                                 {"name": "verifyingContract", "type": "address"}],
                "Permit": [{"name": "owner", "type": "address"},
                           {"name": "spender", "type": "address"},
                           {"name": "value", "type": "uint256"},
                           {"name": "nonce", "type": "uint256"},
                           {"name": "deadline", "type": "uint256"}]}

def simulate_typed(w3, user, td, pin=None):
    from eth_account import Account
    try:
        tok = w3.to_checksum_address(td["domain"]["verifyingContract"])
        msg = td["message"]
        nsel = "0x7ecebe00"                       # nonces(address)
        n = int(w3.eth.call({"to": tok, "data": nsel + PROBE_ADDR[2:].rjust(64, "0")}).hex() or "0", 16)
        eq = {"domain": td["domain"], "types": PERMIT_TYPES, "primaryType": "Permit",
              "message": {"owner": PROBE_ADDR, "spender": msg["spender"],
                          "value": int(msg["value"]), "nonce": n, "deadline": int(msg["deadline"])}}
        sg = Account.sign_typed_data(PROBE_PK, full_message=eq)
        sel = w3.keccak(text="permit(address,address,uint256,uint256,uint8,bytes32,bytes32)")[:4].hex()
        data = ("0x" + sel
                + PROBE_ADDR[2:].lower().rjust(64, "0")
                + str(msg["spender"])[2:].lower().rjust(64, "0")
                + hex(int(msg["value"]))[2:].rjust(64, "0")
                + hex(int(msg["deadline"]))[2:].rjust(64, "0")
                + hex(sg.v)[2:].rjust(64, "0")
                + hex(sg.r)[2:].rjust(64, "0")
                + hex(sg.s)[2:].rjust(64, "0"))
        out = simulate(w3, PROBE_ADDR, {"to": tok, "value": 0, "data": data}, pin)
        out["probe"] = PROBE_ADDR.lower()
        return out
    except Exception as e:
        return {"reverted": True, "error": f"探针模拟失败: {type(e).__name__}: {e}"[:160]}
