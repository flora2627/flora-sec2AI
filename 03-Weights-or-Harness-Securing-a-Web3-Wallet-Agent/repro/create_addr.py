"""预计算 CREATE 地址:addr = keccak(rlp([deployer, nonce]))[12:]

用来在**部署之前**就知道 token 会落在哪个地址,好把这个地址写进注入载荷里。
真实攻击者就是这么干的 —— 他部署自己的合约,地址他自己算得出来。
"""
from web3 import Web3

def _rlp_uint(n):
    if n == 0: return b"\x80"
    if n < 0x80: return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 + len(b)]) + b

def create_address(deployer, nonce):
    payload = bytes([0x80 + 20]) + bytes.fromhex(deployer[2:]) + _rlp_uint(nonce)
    assert len(payload) < 56
    return "0x" + Web3.keccak(bytes([0xc0 + len(payload)]) + payload)[12:].hex()
