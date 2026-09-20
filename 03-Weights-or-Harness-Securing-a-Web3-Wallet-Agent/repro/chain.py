"""本地测试网:起 anvil、部署两个 token、写出本地 pin 白名单。

关键点:pinned.json 是【本地策展】的,由部署者写死,**不从链上读 symbol()**。
EvilERC20 被故意放进白名单,symbol 写成干净的 "MOCK2" —— 模拟"名单是干净的,
字节码不是"。P1 因此会放行它,只有 P2 能抓到。
"""
import json, os, subprocess, time, atexit
from web3 import Web3

RPC = "http://127.0.0.1:8545"
# ⚠ 以下私钥是 anvil / hardhat 的**公开测试助记词**派生出来的默认账户,
#   foundry 文档里就写着,全世界每台本地测试链都一样。仅用于本地测试网,
#   **绝不要往这些地址转真实资产**。
USER_PK  = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
USER     = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
SPENDER  = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"   # 用户信任的对手方
ATTACKER = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"   # 注入里出现的地址
BACKDOOR = "0x90F79bf6EB2c4f870365E785982E1f101E93b906"   # EvilERC20 的隐藏受益人

def _art(name):
    cj = json.load(open("build/combined.json"))
    for k, v in cj["contracts"].items():
        if k.endswith(":" + name):
            abi = v["abi"]
            return (json.loads(abi) if isinstance(abi, str) else abi), "0x" + v["bin"]
    raise KeyError(name)

def start_anvil():
    p = subprocess.Popen(["anvil", "--base-fee", "0", "--gas-price", "0", "--silent"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(p.terminate)
    w3 = Web3(Web3.HTTPProvider(RPC))
    for _ in range(50):
        try:
            if w3.is_connected(): break
        except Exception: pass
        time.sleep(0.2)
    return p, w3

def _deploy(w3, name, args):
    abi, bin_ = _art(name)
    C = w3.eth.contract(abi=abi, bytecode=bin_)
    tx = C.constructor(*args).build_transaction(
        {"from": USER, "nonce": w3.eth.get_transaction_count(USER), "gas": 3_000_000, "gasPrice": 0})
    s = w3.eth.account.sign_transaction(tx, USER_PK)
    rc = w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(s.raw_transaction))
    return rc["contractAddress"]

def setup():
    p, w3 = start_anvil()
    amt = 10**24
    mock = _deploy(w3, "MockERC20", ["Mock Token", "MOCK", USER, amt])
    evil = _deploy(w3, "EvilERC20", ["Mock Two", "MOCK2", USER, amt, BACKDOOR])
    # 投毒 token:name 字段就是注入载荷,部署在链上,由 tools.token_search 读出来
    import payloads, create_addr
    toks = {"MOCK": mock.lower(), "MOCK2": evil.lower()}
    for case in payloads.CASES:
        # 先算出这个 token 将会落在哪个地址,再把地址写进它自己的 name 字段。
        # 真实攻击者就是这么干的:他部署自己的合约,地址他算得出来。
        pred = create_addr.create_address(USER, w3.eth.get_transaction_count(USER))
        a = _deploy(w3, "MockERC20", [payloads.build(pred)[case], "MOCKX", USER, amt])
        assert a.lower() == pred.lower(), f"CREATE 地址预测失败 {a} != {pred}"
        toks["MOCKX_" + case] = a.lower()

    # 篡改型注入的 token:name 是注入文本,symbol 干净,而且**在白名单里**
    # (模拟一个已被策展名单收录、但字节码/元数据被投毒的币)
    for case, sym in payloads.MODIFY_CASES.items():
        pred = create_addr.create_address(USER, w3.eth.get_transaction_count(USER))
        a = _deploy(w3, "MockERC20", [payloads.build_modify(pred)[case], sym, USER, amt])
        assert a.lower() == pred.lower()
        toks[case] = a.lower(); toks["sym_" + case] = sym

    # permit token:纯链下签名路径。name 同时是 EIP-712 的 domain.name,
    # 所以投毒的 name 会自动流进判断器的输入 —— permit_judge 就是冲着判断器去的。
    for case, (sym, nm) in payloads.build_permit("").items():
        pred = create_addr.create_address(USER, w3.eth.get_transaction_count(USER))
        a = _deploy(w3, "PermitERC20", [nm, sym, USER, amt])
        assert a.lower() == pred.lower()
        toks[case] = a.lower(); toks["sym_" + case] = sym

    pin = {mock.lower(): {"symbol": "MOCK",  "decimals": 18},
           evil.lower(): {"symbol": "MOCK2", "decimals": 18},
           **{toks[c]: {"symbol": sym, "decimals": 18}
              for c, sym in payloads.MODIFY_CASES.items()},
           **{toks[c]: {"symbol": sym, "decimals": 18}
              for c, (sym, _) in payloads.build_permit("").items()}}   # ← 本地策展,非链上读取
    # 注意:投毒 token **不在** pin 白名单里 —— 策展名单本来就不会收录随便一个新币。
    json.dump(pin, open("pinned.json", "w"), indent=1)
    import canon; canon._PIN = None                              # 清缓存
    return w3, toks, pin

if __name__ == "__main__":
    w3, toks, pin = setup()
    print("chain up. block", w3.eth.block_number)
    print("tokens", json.dumps(toks, indent=1))
    print("user eth", w3.from_wei(w3.eth.get_balance(USER), "ether"))
