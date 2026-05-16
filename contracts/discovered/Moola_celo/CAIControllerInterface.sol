pragma solidity ^0.5.16;

contract CAIControllerInterface {
    function getCAIAddress() public view returns (address);
    function getMintableCAI(address minter) public view returns (uint, uint);
    function mintCAI(address minter, uint mintCAIAmount) external returns (uint);
    function repayCAI(address repayer, uint repayCAIAmount) external returns (uint);

    function _initializeCheeCAIState(uint blockNumber) external returns (uint);
    function updateCheeCAIMintIndex() external returns (uint);
    function calcDistributeCAIMinterChee(address caiMinter) external returns(uint, uint, uint, uint);
}
