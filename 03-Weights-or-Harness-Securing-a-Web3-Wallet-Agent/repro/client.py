"""agent 侧拿到的全部东西。这个进程里没有私钥。"""
import json, os, subprocess, sys

class SignerClient:
    def __init__(self, pk, rpc, addr, pin, python=sys.executable):
        env = dict(os.environ, SIGNER_PK=pk)
        cfg = json.dumps({"rpc": rpc, "addr": addr, "pin": pin})
        self.p = subprocess.Popen([python, "signerd.py", cfg], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        del pk, env                                  # 父进程不再持有任何引用

    def _rpc(self, req):
        self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n"); self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def mint_ticket(self, typed, ttl=300): return self._rpc({"op": "mint", "typed": typed, "ttl": ttl})
    def propose(self, tx):                 return self._rpc({"op": "propose", "tx": tx})
    def close(self):
        try: self.p.stdin.close(); self.p.wait(timeout=5)
        except Exception: self.p.kill()
