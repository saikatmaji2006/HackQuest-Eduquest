require("@nomicfoundation/hardhat-toolbox");
// require("dotenv").config();

module.exports = {
    solidity: "0.8.20",
    networks: {
        educhain: {
            url: "https://rpc.open-campus-codex.gelato.digital",
            accounts: ["bd67acd765cbc45fa2fe457b4fc471e15fafa10ba227ddad847602c48ac5af59"]
        }
    }
    
};
