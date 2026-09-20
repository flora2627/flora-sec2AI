"""签名进程。私钥只存在于**这个进程**的地址空间里。

父进程(跑 agent 的那个)通过 stdin/stdout 的 JSON 行跟它说话,能表达的只有两句:
    {"op":"mint","typed":...}      阶段 A
    {"op":"propose","tx":...}      阶段 C
私钥没有任何一条协议路径能出去 —— 这才是边界,不是 Python 的下划线。
"""
import sys, json, os
from web3 import Web3
import signer

def main():
    cfg = json.loads(sys.argv[1])
    w3 = Web3(Web3.HTTPProvider(cfg["rpc"]))
    pk = os.environ.pop("SIGNER_PK")            # 从环境拿完立刻从环境里抹掉
    sg = signer.Signer(w3, pk, cfg["addr"], cfg["pin"])
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            if req["op"] == "mint":
                sg.mint_ticket(req["typed"], req.get("ttl", 300))
                out = {"ok": True}
            elif req["op"] == "propose":
                raw = sg.propose(req["tx"])
                out = {"ok": True, "signed": "0x" + raw.hex()}   # 只回签名结果
            else:
                out = {"ok": False, "stage": "OP", "reason": "unknown op"}
        except signer.Rejected as e:
            s = str(e); out = {"ok": False, "stage": s.split("]")[0].lstrip("["),
                               "reason": s.split("] ", 1)[1]}
        except Exception as e:
            out = {"ok": False, "stage": "ERR", "reason": f"{type(e).__name__}: {e}"[:200]}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()
