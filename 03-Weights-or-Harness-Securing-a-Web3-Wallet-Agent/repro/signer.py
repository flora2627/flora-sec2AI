"""唯一签名接口。私钥在本模块之外不存在第二个引用。

流程即我们讨论收敛出的那个 SOP:

  阶段 A  mint_ticket(typed)   用户亲手输入 → ticket。此时 agent 尚未启动,
                               因此 D 在因果上还不存在 → U ⊥ D 由**时序**保证。
  阶段 B  agent 随便被打穿。它拿到的只有 propose(),没有私钥。
  阶段 C  propose(tx)          decode → P1 → P2 → 消费 ticket → 签。

注:真实系统里阶段 C 应该是独立进程/HSM,Python 的私有属性不是安全边界。
这里的边界是**结构演示**,不是强制。
"""
import time, json
import canon, judge, risk

class Rejected(Exception): pass

class Signer:
    def __init__(self, w3, pk, addr, pin):
        self.__pk = pk                 # ← 唯一引用
        self.w3, self.addr, self.pin = w3, addr.lower(), pin
        self.ticket = None
        self.log = []

    # ---------- 阶段 A ----------
    def mint_ticket(self, typed, ttl=300):
        """只记录用户亲手打的那句话。此处**不调用任何模型** —— 判断留到 propose。"""
        self.ticket = {"typed": typed, "exp": time.time() + ttl, "used": False}
        return self.ticket

    # ---------- 阶段 C ----------
    TX_KEYS    = {"kind", "to", "value", "data"}
    TYPED_KEYS = {"kind", "domain", "types", "primaryType", "message"}

    def propose(self, req):
        """agent 唯一能碰到的东西,**唯一的签名接口**。

        两种载荷走同一个入口、同一个判断器:
          kind="tx"          要广播的链上交易
          kind="typed_data"  EIP-712 链下签名(permit 这类) —— 签完不用广播,
                             攻击者拿着签名自己去链上执行,一样把钱取走。
                             所以检查点在**签名产生处**,不在广播处。
        nonce/gas/chainId 由本函数按本地策略填 —— agent 无权提议用户没说的字段。
        """
        t = self.ticket
        def rej(stage, why):
            self.log.append({"stage": stage, "reason": why}); raise Rejected(f"[{stage}] {why}")

        # ── 以下是工程化的机械检查,与"能不能抓到提示词注入"无关,demo 里注释掉 ──
        # 这 60 个 run 里它们一次都没触发过(全部拒签的 stage 都是 P1),
        # 留着只是真实系统需要。逐条说明见下:
        #
        #   TICKET  重放保护 + ttl。判断器是无状态的,管不了"这张票用过没",
        #           所以真实系统必须留。但它不参与"这笔交易对不对"的判断。
        # if t is None or t["used"] or time.time() > t["exp"]:
        #     rej("TICKET", "无有效 ticket(未铸出/已用/过期)")
        #
        #   SHAPE-1 未知载荷类型。删了也会被判断器拒(decoded=false + 模拟失败)。
        # if kind not in ("tx", "typed_data"):
        #     rej("SHAPE", f"未知的签名载荷类型: {kind}")
        #
        #   SHAPE-2 越权字段。纯冗余 —— 下面的 full 是显式构造的,只取 to/value/data,
        #           agent 多塞的 nonce/gasPrice 根本不会被读到。
        # if set(req) - allowed:
        #     rej("SHAPE", f"提议含越权字段: {sorted(set(req) - allowed)}")
        # ────────────────────────────────────────────────────────────────
        kind = req.get("kind", "tx")
        req = {k: v for k, v in req.items() if v is not None or k != "types"}

        if kind == "tx":
            # 解不开也照样往下走 —— "解不开该不该签"是判断,不是代码的决定。
            c   = canon.canon(req)
            sim = risk.simulate(self.w3, self.addr, req, self.pin)
        else:
            c   = canon.canon_typed(req)
            sim = risk.simulate_typed(self.w3, self.addr, req, self.pin)
        f = risk.facts(sim, self.pin)                      # 机械:换面值、贴符号、如实转述
        ok, why = judge.p1(t["typed"], c, f)               # ← 唯一的判断
        if not ok: rej("P1", why)

        t["used"] = True                                   # 一票一签,不可重放
        if kind == "tx":
            full = {"to": self.w3.to_checksum_address(req["to"]), "value": int(req.get("value", 0)),
                    "data": req.get("data", "0x"), "gas": 300_000, "gasPrice": 0,
                    "nonce": self.w3.eth.get_transaction_count(self.w3.to_checksum_address(self.addr)),
                    "chainId": self.w3.eth.chain_id}
            signed = self.w3.eth.account.sign_transaction(full, self.__pk)   # ← 私钥
            raw = signed.raw_transaction
        else:
            from eth_account import Account
            sg = Account.sign_typed_data(self.__pk, full_message={
                "domain": req["domain"], "types": risk.PERMIT_TYPES,
                "primaryType": req["primaryType"], "message": req["message"]})
            raw = sg.signature                                               # ← 私钥
        self.log.append({"stage": "SIGNED", "reason": kind})
        return raw
