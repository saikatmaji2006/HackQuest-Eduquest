"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { motion, AnimatePresence } from "framer-motion"

export function HeroSection() {
  const [isHovered, setIsHovered] = useState(false)
  const [mounted, setMounted] = useState(false)
  const [animationState, setAnimationState] = useState(0)

  // Fix hydration error by only rendering custom background after mount
  useEffect(() => {
    setMounted(true)
    
    // Cycle through animation states
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3)
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])

  // Feature cards data
  const features = [
    {
      title: "Verified Credentials",
      description: "Secure, tamper-proof certificates verified on blockchain",
      icon: (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-5 w-5 text-pink-400"
        >
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
          <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      )
    },
    {
      title: "Decentralized Learning",
      description: "Community-driven curriculum with tokenized incentives",
      icon: (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-5 w-5 text-pink-400"
        >
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
        </svg>
      )
    },
    {
      title: "AI-Powered Insights",
      description: "Personalized learning paths that adapt to your progress",
      icon: (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-5 w-5 text-pink-400"
        >
          <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" />
        </svg>
      )
    }
  ]

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
          {[...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 rounded-full bg-pink-500"
              initial={{
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
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
        <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
          <div className="flex flex-col justify-center space-y-8">
            <motion.div 
              className="space-y-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <div className="flex items-center">
                <motion.div 
                  className="h-1 w-12 bg-gradient-to-r from-pink-500 to-purple-600 mr-4"
                  initial={{ width: 0 }}
                  animate={{ width: 48 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                ></motion.div>
                <motion.span 
                  className="text-pink-400 text-sm font-bold tracking-wider uppercase"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.6 }}
                >
                  Next-Gen Web3 Education
                </motion.span>
              </div>
              <motion.h1 
                className="text-5xl font-bold tracking-tight sm:text-6xl xl:text-7xl/none text-white"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.3 }}
              >
                <span className="text-white">Transform Your</span>{" "}
                <span className="relative">
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-600">
                    Learning Journey
                  </span>
                  <motion.span 
                    className="absolute -bottom-2 left-0 h-1 w-full bg-gradient-to-r from-pink-500 to-purple-600"
                    initial={{ width: 0 }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 1, delay: 1 }}
                  />
                </span>
              </motion.h1>
              <motion.p 
                className="max-w-[600px] text-gray-300 text-xl leading-relaxed"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.5 }}
              >
                Unlock the future of education with personalized learning roadmaps, immersive simulations, and real-world opportunities—all secured by blockchain technology.
              </motion.p>
            </motion.div>
            
            <motion.div 
              className="flex flex-col gap-4 sm:flex-row"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.7 }}
            >
              <Link href="/courses">
                <Button 
                  size="lg" 
                  className="w-full sm:w-auto text-base px-8 py-6 bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white border-none shadow-lg shadow-pink-500/20 transition-all duration-300 hover:scale-105"
                >
                  Explore Courses
                </Button>
              </Link>
              <Link href="/connect">
                <Button 
                  size="lg" 
                  variant="outline" 
                  className="w-full sm:w-auto text-base px-8 py-6 border-pink-500/30 text-white bg-white/5 backdrop-blur-sm hover:bg-white/10 hover:border-pink-500/50 transition-all duration-300 hover:scale-105"
                >
                  Connect Wallet
                </Button>
              </Link>
            </motion.div>
            
            {/* Enhanced feature cards */}
            <motion.div 
              className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6 pt-6 border-t border-white/10"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.9 }}
            >
              {features.map((feature, index) => (
                <motion.div
                  key={index}
                  className="group p-4 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10 hover:bg-white/10 transition-all duration-300 hover:shadow-lg hover:shadow-pink-500/10"
                  whileHover={{ y: -5 }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.9 + index * 0.1 }}
                >
                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-lg bg-black/40 border border-white/10 text-pink-400 group-hover:text-white group-hover:bg-gradient-to-r group-hover:from-pink-500 group-hover:to-purple-600 transition-all duration-300">
                      {feature.icon}
                    </div>
                    <div>
                      <h3 className="font-medium text-white mb-1">{feature.title}</h3>
                      <p className="text-sm text-gray-400">{feature.description}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
          
          {/* Enhanced interactive card */}
          <motion.div 
            className="flex items-center justify-center"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.5 }}
          >
            <motion.div
              className="relative w-full max-w-[500px]"
              whileHover={{ scale: 1.02, rotate: 2 }}
              transition={{ duration: 0.4 }}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            >
              {/* Animated glow effect */}
              <motion.div 
                className="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-pink-500 to-purple-600 opacity-75 blur-xl"
                animate={{
                  opacity: isHovered ? 0.85 : 0.5,
                  scale: isHovered ? 1.05 : 1,
                }}
                transition={{ duration: 0.4 }}
              />
              
              <div className="relative rounded-2xl border border-white/20 bg-black/70 backdrop-blur-md p-6 shadow-xl">
                <div className="space-y-6">
                  {/* Enhanced header with badge */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <motion.div 
                        className="h-12 w-12 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-pink-500/30"
                        whileHover={{ scale: 1.1 }}
                      >
                        W3
                      </motion.div>
                      <div>
                        <div className="font-medium text-white text-xl">Web3 Fundamentals</div>
                        <div className="text-xs text-pink-300">Curated by industry experts</div>
                      </div>
                    </div>
                    <motion.div 
                      className="flex items-center gap-1 bg-gradient-to-r from-pink-500/20 to-purple-600/20 rounded-full px-3 py-1 text-pink-300 text-sm border border-pink-500/20"
                      whileHover={{ scale: 1.05 }}
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="h-4 w-4"
                      >
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                      </svg>
                      <span>4.9</span>
                    </motion.div>
                  </div>
                  
                  {/* Enhanced visual representation */}
                  <div className="h-56 rounded-xl bg-gradient-to-br from-black to-purple-950/50 backdrop-blur-sm flex items-center justify-center border border-white/10 overflow-hidden relative group">
                    {/* Animated blockchain nodes */}
                    {[...Array(5)].map((_, i) => (
                      <motion.div 
                        key={i}
                        className="absolute h-10 w-10 rounded-md bg-gradient-to-r from-pink-500/20 to-purple-600/20 border border-white/10 flex items-center justify-center"
                        initial={{
                          x: (i % 3) * 60 - 60,
                          y: Math.floor(i / 3) * 60 - 30,
                          rotate: 0
                        }}
                        animate={{
                          rotate: i % 2 === 0 ? 360 : -360,
                          scale: isHovered ? 1.1 : 1
                        }}
                        transition={{
                          rotate: { duration: 20, repeat: Infinity, ease: "linear" },
                          scale: { duration: 0.5 }
                        }}
                      >
                        <div className="h-2 w-2 rounded-full bg-pink-400" />
                      </motion.div>
                    ))}
                    
                    {/* Connecting lines */}
                    <svg className="absolute inset-0 w-full h-full opacity-70 group-hover:opacity-100 transition-opacity duration-300" viewBox="0 0 240 240">
                      <motion.path 
                        d="M60 90 L120 60 L180 90 L180 150 L120 180 L60 150 Z" 
                        fill="none" 
                        stroke="url(#lineGradient)" 
                        strokeWidth="1.5"
                        initial={{ pathLength: 0, opacity: 0 }}
                        animate={{ 
                          pathLength: isHovered ? 1 : 0.7, 
                          opacity: isHovered ? 1 : 0.7 
                        }}
                        transition={{ duration: 1.5 }}
                      />
                      <defs>
                        <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#ec4899" />
                          <stop offset="100%" stopColor="#9333ea" />
                        </linearGradient>
                      </defs>
                    </svg>
                    
                    {/* Center icon with pulse effect */}
                    <motion.div 
                      className="relative z-10"
                      animate={{
                        scale: [1, 1.05, 1],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                      }}
                    >
                      <motion.div 
                        className="absolute -inset-6 bg-pink-500/20 rounded-full blur-md"
                        animate={{
                          scale: [1, 1.3, 1],
                          opacity: [0.7, 0.2, 0.7],
                        }}
                        transition={{
                          duration: 3,
                          repeat: Infinity,
                        }}
                      />
                      <div className="h-14 w-14 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 flex items-center justify-center shadow-lg shadow-pink-500/30">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="h-8 w-8 text-white"
                        >
                          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                        </svg>
                      </div>
                    </motion.div>
                  </div>
                  
                  {/* Enhanced progress bar */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="font-medium text-white">Your Progress</div>
                      <motion.div 
                        className="text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500 font-bold"
                        animate={{
                          scale: [1, 1.05, 1],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          repeatType: "reverse",
                        }}
                      >
                        65%
                      </motion.div>
                    </div>
                    <div className="h-3 w-full rounded-full bg-white/5 overflow-hidden border border-white/10">
                      <motion.div 
                        className="h-full rounded-full bg-gradient-to-r from-pink-500 to-purple-600"
                        initial={{ width: "0%" }}
                        animate={{ width: "65%" }}
                        transition={{ duration: 1, delay: 0.5 }}
                      />
                    </div>
                  </div>
                  
                  {/* Enhanced button section */}
                  <div className="flex items-center justify-between pt-2">
                    <div className="flex items-center gap-2">
                      <div className="h-6 w-6 rounded-full bg-purple-600/20 flex items-center justify-center">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="h-3 w-3 text-purple-400"
                        >
                          <path d="M12 5v14M5 12h14" />
                        </svg>
                      </div>
                      <div className="text-sm text-gray-400">
                        <span className="text-pink-400 font-medium">Next:</span> Smart Contracts
                      </div>
                    </div>
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.98 }}>
                      <Button 
                        size="sm" 
                        className="bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white border-none shadow-md shadow-pink-500/20 px-6"
                      >
                        Continue
                      </Button>
                    </motion.div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}