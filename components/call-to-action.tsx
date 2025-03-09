"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"

export function CallToAction() {
  const [mounted, setMounted] = useState(false)
  const [animationState, setAnimationState] = useState(0)
  
  useEffect(() => {
    setMounted(true)
    
    // Animation cycle for background gradients
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3)
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])
  
  return (
    <section className="w-full py-12 md:py-24 lg:py-32 xl:py-40 relative overflow-hidden">
      {/* Enhanced animated background - only rendered client-side to avoid hydration errors */}
      {mounted && (
        <div className="absolute inset-0 -z-10">
          {/* Base gradient */}
          <div className="absolute inset-0 bg-black" />
          
          {/* Enhanced gradient elements with animation */}
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
          
          {/* Floating particles */}
          {mounted && [...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 rounded-full bg-pink-500"
              initial={{
                x: Math.random() * (typeof window !== "undefined" ? window.innerWidth : 1000),
                y: Math.random() * (typeof window !== "undefined" ? window.innerHeight : 800),
                opacity: Math.random() * 0.5 + 0.3,
                scale: Math.random() * 2 + 0.5
              }}
              animate={{
                y: [null, `-${Math.random() * 100 + 50}px`],
                opacity: [null, Math.random() * 0.3 + 0.1],
              }}
              transition={{
                duration: Math.random() * 10 + 15,
                repeat: Infinity,
                ease: "linear"
              }}
            />
          ))}
          
          {/* Animated gradient blobs */}
          <motion.div 
            className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-r from-pink-600/10 to-purple-600/10 rounded-full blur-3xl"
            animate={{
              scale: [1, 1.2, 1],
              x: [0, 20, 0],
              y: [0, -20, 0],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              repeatType: "reverse",
            }}
          />
          
          <motion.div 
            className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-r from-purple-600/10 to-pink-600/10 rounded-full blur-3xl"
            animate={{
              scale: [1.2, 1, 1.2],
              x: [0, -20, 0],
              y: [0, 20, 0],
            }}
            transition={{
              duration: 9,
              repeat: Infinity,
              repeatType: "reverse",
            }}
          />
          
          {/* Subtle grid overlay with animation */}
          <motion.div 
            className="absolute inset-0 opacity-10" 
            animate={{
              backgroundPosition: ["0px 0px", "40px 40px"]
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "linear"
            }}
            style={{ 
              backgroundImage: "linear-gradient(to right, #ffffff 1px, transparent 1px), linear-gradient(to bottom, #ffffff 1px, transparent 1px)", 
              backgroundSize: "40px 40px" 
            }} 
          />
        </div>
      )}
      
      <div className="container px-4 md:px-6 relative z-10">
        <motion.div 
          className="flex flex-col items-center justify-center space-y-6 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Decorative elements */}
          <motion.div 
            className="flex justify-center items-center gap-2 mb-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            <motion.div 
              className="h-1 w-16 bg-gradient-to-r from-pink-500 to-purple-600 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: 64 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            ></motion.div>
            <motion.div 
              className="h-2 w-2 rounded-full bg-white"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3, delay: 0.7 }}
            ></motion.div>
            <motion.div 
              className="h-1 w-16 bg-gradient-to-r from-purple-600 to-pink-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: 64 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            ></motion.div>
          </motion.div>
          
          <div className="space-y-4">
            <motion.h2 
              className="text-3xl font-bold tracking-tighter md:text-5xl/tight text-white"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.7 }}
            >
              Ready to <motion.span 
                className="relative inline-block"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8, duration: 0.5 }}
              >
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-600">
                  Transform
                </span>
                <motion.span 
                  className="absolute -bottom-1 left-0 w-full h-1 bg-gradient-to-r from-pink-500 to-purple-600"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ delay: 1.2, duration: 0.8 }}
                ></motion.span>
              </motion.span> Your Learning Journey?
            </motion.h2>
            
            <motion.p 
              className="mx-auto max-w-[700px] text-gray-300 md:text-xl"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.7 }}
            >
              Join thousands of students and professionals who are already leveraging the power of Web3 
              to advance their careers.
            </motion.p>
          </div>
          
          <motion.div 
            className="flex flex-col gap-4 min-[400px]:flex-row pt-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.7 }}
          >
            <Link href="/courses">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.98 }}>
                <Button 
                  size="lg" 
                  variant="secondary" 
                  className="w-full min-[400px]:w-auto text-base shadow-lg relative overflow-hidden group"
                >
                  <motion.span 
                    className="absolute inset-0 bg-white/10 rounded-full"
                    initial={{ scale: 0, x: "-50%", y: "-50%" }}
                    whileHover={{ scale: 2.5 }}
                    transition={{ duration: 0.5 }}
                  />
                  <span className="relative z-10">Explore Courses</span>
                  <motion.svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    width="24" 
                    height="24" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    className="ml-2 h-5 w-5"
                  >
                    <path d="M5 12h14" />
                    <path d="m12 5 7 7-7 7" />
                  </motion.svg>
                </Button>
              </motion.div>
            </Link>
            
            <Link href="/connect">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.98 }}>
                <Button 
                  size="lg" 
                  className="w-full min-[400px]:w-auto bg-white text-purple-600 hover:bg-white/90 text-base shadow-lg relative overflow-hidden group"
                >
                  <motion.span 
                    className="absolute inset-0 bg-purple-600/10 rounded-full"
                    initial={{ scale: 0, x: "-50%", y: "-50%" }}
                    whileHover={{ scale: 2.5 }}
                    transition={{ duration: 0.5 }}
                  />
                  <motion.svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 mr-2"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                    />
                  </motion.svg>
                  <span className="relative z-10">Connect Wallet</span>
                </Button>
              </motion.div>
            </Link>
          </motion.div>
          
          {/* Floating badges */}
          <div className="relative w-full max-w-md h-8 mt-8">
            {[
              { text: "Web3 Certified", position: "left-0" },
              { text: "Blockchain", position: "left-1/4" },
              { text: "NFT Credentials", position: "left-2/4" },
              { text: "Decentralized", position: "left-3/4" }
            ].map((badge, i) => (
              <motion.div
                key={i}
                className={`absolute ${badge.position} transform -translate-x-1/2 top-0 bg-white/10 text-white/90 text-xs px-3 py-1 rounded-full border border-white/20`}
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1 + (i * 0.1), duration: 0.5 }}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
              >
                {badge.text}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}