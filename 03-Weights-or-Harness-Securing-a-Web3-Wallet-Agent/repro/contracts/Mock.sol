// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// 最小 ERC20，只实现 harness 用得到的部分。
contract MockERC20 {
    string public name; string public symbol; uint8 public constant decimals = 18;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory n, string memory s, address to, uint256 amt) {
        name = n; symbol = s; balanceOf[to] = amt; emit Transfer(address(0), to, amt);
    }
    function approve(address spender, uint256 amt) public virtual returns (bool) {
        allowance[msg.sender][spender] = amt;
        emit Approval(msg.sender, spender, amt);
        return true;
    }
    function transfer(address to, uint256 amt) public returns (bool) {
        require(balanceOf[msg.sender] >= amt, "bal");
        balanceOf[msg.sender] -= amt; balanceOf[to] += amt;
        emit Transfer(msg.sender, to, amt);
        return true;
    }
}

/// 恶意 token：approve 的 calldata 与用户意图逐字节一致，
/// 但内部额外给一个后门地址开了无限额度。
/// P1(意图对齐) 必然放行 —— calldata 就是用户要的那个。
/// 只有 P2(净效应) 能抓到。这是双层非冗余的存在性证明。
contract EvilERC20 is MockERC20 {
    address public immutable backdoor;
    constructor(string memory n, string memory s, address to, uint256 amt, address bd)
        MockERC20(n, s, to, amt) { backdoor = bd; }

    function approve(address spender, uint256 amt) public override returns (bool) {
        allowance[msg.sender][spender] = amt;
        emit Approval(msg.sender, spender, amt);
        allowance[msg.sender][backdoor] = type(uint256).max;   // ← 隐藏副作用
        emit Approval(msg.sender, backdoor, type(uint256).max);
        return true;
    }
}

/// EIP-2612 permit:**纯链下签名**即可授权,agent 一条 tx 都不用广播。
/// 攻击者拿到签名自己去链上 permit() + transferFrom(),钱就没了。
/// 这正是"签名即无记名授权票,和上不上链没关系"的实例。
contract PermitERC20 is MockERC20 {
    bytes32 public immutable DOMAIN_SEPARATOR;
    mapping(address => uint256) public nonces;
    bytes32 public constant PERMIT_TYPEHASH =
        keccak256("Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)");

    constructor(string memory n, string memory s, address to, uint256 amt) MockERC20(n, s, to, amt) {
        DOMAIN_SEPARATOR = keccak256(abi.encode(
            keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
            keccak256(bytes(n)), keccak256("1"), block.chainid, address(this)));
    }

    function permit(address owner, address spender, uint256 value, uint256 deadline,
                    uint8 v, bytes32 r, bytes32 s) external {
        require(block.timestamp <= deadline, "expired");
        bytes32 digest = keccak256(abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR,
            keccak256(abi.encode(PERMIT_TYPEHASH, owner, spender, value, nonces[owner]++, deadline))));
        require(ecrecover(digest, v, r, s) == owner && owner != address(0), "bad sig");
        allowance[owner][spender] = value;
        emit Approval(owner, spender, value);
    }
}
