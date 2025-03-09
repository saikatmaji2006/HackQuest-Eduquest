"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { ethers } from "ethers";

export default function ConnectPage() {
  const [account, setAccount] = useState(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState(null);
  const [mounted, setMounted] = useState(false);
  const [animationState, setAnimationState] = useState(0);
  const [walletAvailable, setWalletAvailable] = useState(false);

  // Set mounted state after component mounts to avoid hydration errors
  useEffect(() => {
    setMounted(true);

    // Check if ethereum is available - only in client
    setWalletAvailable(typeof window !== "undefined" && !!window.ethereum);

    // Cycle through animation states
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev === 2 ? 0 : prev + 1));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const connectWallet = async () => {
    setIsConnecting(true);
    setError(null);

    try {
      if (typeof window === "undefined" || !window.ethereum) {
        throw new Error("No wallet detected. Please install MetaMask or another Web3 wallet.");
      }

      try {
        // For ethers v6, use BrowserProvider; if using ethers v5, replace with Web3Provider
        const provider = new ethers.BrowserProvider(window.ethereum);
        const accounts = await provider.send("eth_requestAccounts", []);

        if (accounts.length > 0) {
          setAccount(accounts[0]);
        } else {
          throw new Error("No accounts found. Please unlock your wallet and try again.");
        }
      } catch (connectionError) {
        console.error("Provider connection error:", connectionError);

        if (connectionError.code === 4001) {
          throw new Error("Connection rejected. Please approve the connection request in your wallet.");
        } else if (connectionError.code === -32002) {
          throw new Error("Connection request already pending. Please check your wallet.");
        } else {
          throw connectionError;
        }
      }
    } catch (err) {
      console.error("Error connecting wallet:", err);
      setError(err instanceof Error ? err.message : "Failed to connect wallet");
    } finally {
      setIsConnecting(false);
    }
  };

  // Client-side animated background
  const clientSideBackground = mounted ? (
    <div className="absolute inset-0 -z-10">
      {/* Base gradient */}
      <div className="absolute inset-0 bg-black" />

      {/* Animated radial gradients */}
      <motion.div
        className="absolute inset-0 opacity-80"
        animate={{
          background: [
            "radial-gradient(circle at 20% 20%, rgba(236, 72, 153, 0.15) 0%, transparent 50%)",
            "radial-gradient(circle at 50% 80%, rgba(236, 72, 153, 0.15) 0%, transparent 50%)",
            "radial-gradient(circle at 80% 40%, rgba(236, 72, 153, 0.15) 0%, transparent 50%)"
          ][animationState]
        }}
        transition={{ duration: 2 }}
      />

      <motion.div
        className="absolute inset-0 opacity-80"
        animate={{
          background: [
            "radial-gradient(circle at 80% 80%, rgba(147, 51, 234, 0.15) 0%, transparent 50%)",
            "radial-gradient(circle at 20% 50%, rgba(147, 51, 234, 0.15) 0%, transparent 50%)",
            "radial-gradient(circle at 50% 20%, rgba(147, 51, 234, 0.15) 0%, transparent 50%)"
          ][animationState]
        }}
        transition={{ duration: 2 }}
      />

      {/* Animated gradient blobs */}
      <motion.div
        className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-pink-600/10 to-purple-600/10 rounded-full blur-3xl"
        animate={{ scale: [1, 1.2, 1], x: [0, 20, 0], y: [0, -20, 0] }}
        transition={{ duration: 8, repeat: Infinity, repeatType: "reverse" }}
      />

      <motion.div
        className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-r from-purple-600/10 to-pink-600/10 rounded-full blur-3xl"
        animate={{ scale: [1.2, 1, 1.2], x: [0, -20, 0], y: [0, 20, 0] }}
        transition={{ duration: 9, repeat: Infinity, repeatType: "reverse" }}
      />

      {/* Subtle grid overlay with animation */}
      <motion.div
        className="absolute inset-0 opacity-10"
        animate={{ backgroundPosition: ["0px 0px", "40px 40px"] }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        style={{
          backgroundImage:
            "linear-gradient(to right, #ffffff 1px, transparent 1px), linear-gradient(to bottom, #ffffff 1px, transparent 1px)",
          backgroundSize: "40px 40px"
        }}
      />

      {/* Static particles */}
      {Array.from({ length: 15 }).map((_, i) => {
        const xPos = `${(i * 7) % 100}%`;
        const yPos = `${(i * 13) % 100}%`;

        return (
          <motion.div
            key={i}
            className="absolute w-1 h-1 rounded-full bg-pink-500"
            style={{ left: xPos, top: yPos }}
            animate={{ y: [0, -50, 0], opacity: [0.3, 0.7, 0.3] }}
            transition={{
              duration: 4 + (i % 4),
              repeat: Infinity,
              ease: "linear",
              delay: i * 0.2
            }}
          />
        );
      })}
    </div>
  ) : (
    // Fallback background for server-side rendering
    <div className="absolute inset-0 -z-10 bg-black">
      <div className="absolute inset-0 bg-gradient-to-br from-pink-500/10 to-purple-600/10"></div>
    </div>
  );

  return (
    <section className="w-full min-h-[calc(100vh-4rem)] py-12 relative overflow-hidden">
      {clientSideBackground}

      <div className="container flex flex-col items-center justify-center relative z-10 min-h-[calc(100vh-4rem)]">
        <motion.div initial={mounted ? { opacity: 0, y: 20 } : false} animate={mounted ? { opacity: 1, y: 0 } : false} transition={{ duration: 0.6 }}>
          <Card className="w-full max-w-md bg-black/50 backdrop-blur-lg border-gray-800">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl text-white">Connect Your Wallet</CardTitle>
              <CardDescription className="text-gray-300">
                Connect your Web3 wallet to access personalized learning experiences and secure your credentials on the blockchain.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {account ? (
                <div className="p-4 rounded-lg bg-gray-800/50">
                  <p className="text-sm font-medium text-gray-200">Connected Account:</p>
                  <p className="text-xs text-gray-400 break-all mt-1">{account}</p>
                </div>
              ) : !mounted ? (
                <div className="space-y-4">
                  <Button disabled className="w-full bg-gray-700 text-gray-300">
                    Loading...
                  </Button>
                  <Button variant="outline" disabled className="w-full border-gray-700 text-gray-500">
                    Loading...
                  </Button>
                </div>
              ) : !walletAvailable ? (
                <div className="space-y-4">
                  <div className="p-4 rounded-lg bg-amber-900/30 border border-amber-800">
                    <p className="text-sm text-amber-300 font-medium">No Web3 wallet detected</p>
                    <p className="text-xs text-amber-200/70 mt-1">
                      You need to install a Web3 wallet like MetaMask to connect to this application.
                    </p>
                  </div>
                  <Link href="https://metamask.io/download/" target="_blank" className="w-full">
                    <Button className="w-full bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700">
                      Install MetaMask
                    </Button>
                  </Link>
                </div>
              ) : (
                <div className="grid gap-4">
                  <Button onClick={connectWallet} className="w-full bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700" disabled={isConnecting}>
                    {isConnecting ? "Connecting..." : "Connect with MetaMask"}
                  </Button>
                  <Button variant="outline" className="w-full border-gray-700 text-gray-200 hover:bg-gray-800/50" disabled={isConnecting}>
                    Connect with WalletConnect
                  </Button>
                </div>
              )}

              {error && (
                <div className="p-3 text-sm text-red-400 bg-red-900/30 rounded-md border border-red-800">
                  {error}
                  {error.includes && error.includes("No wallet detected") && (
                    <Link href="https://metamask.io/download/" target="_blank">
                      <p className="text-xs mt-1 text-pink-400 hover:underline">Click here to install MetaMask</p>
                    </Link>
                  )}
                </div>
              )}
            </CardContent>
            <CardFooter className="flex flex-col space-y-2">
              {account ? (
                <Link href="/dashboard">
                  <Button className="w-full bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700">
                    Go to Dashboard
                  </Button>
                </Link>
              ) : (
                <p className="text-xs text-center text-gray-400">
                  New to Web3?{" "}
                  <Link href="https://ethereum.org/en/wallets/" target="_blank" className="text-pink-400 hover:text-pink-300 hover:underline">
                    Learn about wallets
                  </Link>
                </p>
              )}
            </CardFooter>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
