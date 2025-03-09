const { ethers } = require("hardhat");

async function main() {
    const EduQuest = await ethers.getContractFactory("EduQuest");
    const eduQuest = await EduQuest.deploy();

    // Using deployed() instead of waitForDeployment() for ethers v5
    await eduQuest.deployed();

    console.log("EduQuest Contract deployed to:", eduQuest.address);
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

