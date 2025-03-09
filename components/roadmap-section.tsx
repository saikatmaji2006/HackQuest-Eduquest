"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { motion } from "framer-motion"
import { ChevronRight, Code, Shield, Brain } from "lucide-react"

// Animation variants for staggered entrance
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2
    }
  }
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5 }
  }
}

export function RoadmapSection() {
  const [activeTab, setActiveTab] = useState("web3")
  const [mounted, setMounted] = useState(false)
  const [animationState, setAnimationState] = useState(0)

  useEffect(() => {
    setMounted(true)
    
    // Cycle through animation states
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3)
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])

  if (!mounted) return null

  const getIconForTab = (tab) => {
    switch(tab) {
      case "web3": return <Code className="h-5 w-5" />;
      case "ai": return <Brain className="h-5 w-5" />;
      case "cyber": return <Shield className="h-5 w-5" />;
      default: return <Code className="h-5 w-5" />;
    }
  }

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

      <div className="container relative z-10 px-4 md:px-6">
        <motion.div 
          className="flex flex-col items-center justify-center space-y-4 text-center"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={itemVariants} className="space-y-2">
            <div className="inline-block rounded-lg bg-gradient-to-r from-pink-500 to-purple-600 px-3 py-1 text-sm text-white shadow-glow">
              Learning Paths
            </div>
            <h2 className="text-3xl font-bold tracking-tighter md:text-5xl/tight bg-clip-text text-transparent bg-gradient-to-r from-pink-400 to-purple-600 relative">
              Personalized Educational Roadmaps
              <motion.span 
                className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-1/2 h-1 bg-gradient-to-r from-pink-500 to-purple-600 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: "50%" }}
                transition={{ duration: 1, delay: 0.5 }}
              ></motion.span>
            </h2>
            <motion.p variants={itemVariants} className="mx-auto max-w-[700px] text-gray-300 md:text-xl">
              Explore our curated learning paths designed to take you from beginner to expert in your chosen field.
            </motion.p>
          </motion.div>
        </motion.div>

        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="mx-auto max-w-4xl mt-12"
        >
          <Tabs defaultValue="web3" className="w-full" onValueChange={setActiveTab}>
            <motion.div variants={itemVariants}>
              <TabsList className="grid w-full grid-cols-3 p-1 rounded-xl bg-black/50 backdrop-blur border border-white/10">
                {["web3", "ai", "cyber"].map((tab) => (
                  <TabsTrigger 
                    key={tab}
                    value={tab}
                    className="flex items-center justify-center gap-2 data-[state=active]:bg-gradient-to-br data-[state=active]:from-pink-500/80 data-[state=active]:to-purple-600/80 data-[state=active]:text-white transition-all duration-300 ease-in-out rounded-lg py-2 text-gray-300"
                  >
                    {getIconForTab(tab)}
                    <span className="capitalize">
                      {tab === "web3" ? "Web3 Development" : 
                       tab === "ai" ? "AI & Machine Learning" : "Cybersecurity"}
                    </span>
                  </TabsTrigger>
                ))}
              </TabsList>
            </motion.div>

            {["web3", "ai", "cyber"].map((tabValue) => (
              <TabsContent key={tabValue} value={tabValue} className="mt-8">
                <div className="relative">
                  <div className="absolute left-8 top-0 bottom-0 w-1 bg-gradient-to-b from-pink-500/40 via-purple-600/40 to-transparent rounded-full" />
                  <div className="space-y-8">
                    {/* Dynamic roadmap items based on tab */}
                    {(tabValue === "web3" ? [
                      {
                        title: "Blockchain Fundamentals",
                        duration: "4 weeks",
                        description: "Learn the core concepts of blockchain technology, including distributed ledgers, consensus mechanisms, and cryptography.",
                        level: "Beginner",
                        levelColor: "pink-500",
                        courses: 3,
                        isActive: true
                      },
                      {
                        title: "Smart Contract Development",
                        duration: "6 weeks",
                        description: "Master Solidity programming and learn to write, test, and deploy secure smart contracts on Ethereum and other blockchains.",
                        level: "Intermediate",
                        levelColor: "purple-500",
                        courses: 4,
                        isActive: false
                      },
                      {
                        title: "dApp Development",
                        duration: "8 weeks",
                        description: "Build full-stack decentralized applications using Web3.js, Ethers.js, and modern frontend frameworks like React and Next.js.",
                        level: "Advanced",
                        levelColor: "red-500",
                        courses: 5,
                        isActive: false
                      }
                    ] : tabValue === "ai" ? [
                      {
                        title: "AI & ML Foundations",
                        duration: "5 weeks",
                        description: "Understand the fundamentals of artificial intelligence, machine learning algorithms, and data preprocessing techniques.",
                        level: "Beginner",
                        levelColor: "pink-500",
                        courses: 4,
                        isActive: true
                      },
                      {
                        title: "Deep Learning & Neural Networks",
                        duration: "7 weeks",
                        description: "Master neural network architectures, deep learning frameworks, and implement advanced models for various applications.",
                        level: "Intermediate",
                        levelColor: "purple-500",
                        courses: 5,
                        isActive: false
                      },
                      {
                        title: "LLM Fine-tuning & Deployment",
                        duration: "8 weeks",
                        description: "Learn to fine-tune large language models, implement inference optimization, and deploy AI solutions in production environments.",
                        level: "Advanced",
                        levelColor: "red-500",
                        courses: 4,
                        isActive: false
                      }
                    ] : [
                      {
                        title: "Security Fundamentals",
                        duration: "4 weeks",
                        description: "Learn the core principles of cybersecurity, including threat modeling, encryption, and security best practices.",
                        level: "Beginner",
                        levelColor: "pink-500",
                        courses: 3,
                        isActive: true
                      },
                      {
                        title: "Ethical Hacking",
                        duration: "6 weeks",
                        description: "Master penetration testing techniques, vulnerability assessment, and secure coding practices to protect digital assets.",
                        level: "Intermediate",
                        levelColor: "purple-500",
                        courses: 4,
                        isActive: false
                      },
                      {
                        title: "Advanced Security Operations",
                        duration: "8 weeks",
                        description: "Develop expertise in security incident response, digital forensics, and implementing enterprise-grade security architectures.",
                        level: "Advanced",
                        levelColor: "red-500",
                        courses: 4,
                        isActive: false
                      }
                    ]).map((item, index) => (
                      <motion.div
                        key={`${tabValue}-${index}`}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5, delay: index * 0.1 }}
                        className="relative pl-12"
                      >
                        <div
                          className={`absolute left-7 -translate-x-1/2 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-300 ${
                            item.isActive 
                              ? "bg-gradient-to-r from-pink-500 to-purple-600 border-pink-500 shadow-glow-sm" 
                              : "bg-black/50 border-gray-500/30"
                          }`}
                        >
                          {item.isActive && <div className="w-2 h-2 rounded-full bg-white animate-pulse" />}
                        </div>
                        <motion.div
                          whileHover={{ scale: 1.02, y: -2 }}
                          transition={{ type: "spring", stiffness: 300 }}
                        >
                          <Card className={`${item.isActive ? 'border-pink-500/30 shadow-glow-md' : 'border-gray-700'} bg-black/50 backdrop-blur-sm overflow-hidden transition-all duration-300`}>
                            <div className={`h-1 w-full ${item.isActive ? 'bg-gradient-to-r from-pink-500 to-purple-600' : 'bg-gray-800'}`}></div>
                            <CardContent className="p-6">
                              <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                    <div className={`${item.isActive ? 'text-pink-500' : 'text-gray-400'} ${item.isActive ? 'animate-pulse' : ''}`}>
                                      {getIconForTab(tabValue)}
                                    </div>
                                    <h3 className="font-bold text-lg text-white">{item.title}</h3>
                                  </div>
                                  <div className="flex items-center gap-1 text-sm font-medium text-gray-400">
                                    <span>{item.duration}</span>
                                  </div>
                                </div>
                                <p className="text-sm text-gray-400">
                                  {item.description}
                                </p>
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2 text-sm">
                                    <div className={`rounded-full bg-${item.levelColor}/10 px-2 py-1 text-xs text-${item.levelColor}`}>
                                      {item.level}
                                    </div>
                                    <div className="text-gray-400">{item.courses} Courses</div>
                                  </div>
                                  
                                  {item.isActive && (
                                    <motion.button
                                      whileHover={{ scale: 1.05 }}
                                      whileTap={{ scale: 0.98 }}
                                      className="flex items-center gap-1 text-sm font-medium text-pink-400 group"
                                    >
                                      <span>Continue Learning</span>
                                      <ChevronRight className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-1" />
                                    </motion.button>
                                  )}
                                </div>
                                
                                {item.isActive && (
                                  <div className="mt-4 pt-3 border-t border-gray-800">
                                    <div className="flex items-center">
                                      <div className="w-full bg-gray-800 rounded-full h-2 mr-2">
                                        <div className="bg-gradient-to-r from-pink-500 to-purple-600 h-2 rounded-full w-[65%] animate-pulse"></div>
                                      </div>
                                      <span className="text-xs font-medium text-gray-400">65%</span>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </CardContent>
                          </Card>
                        </motion.div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </motion.div>
      </div>

      {/* Add required styles */}
      <style jsx global>{`
        .shadow-glow {
          box-shadow: 0 0 15px 2px rgba(236, 72, 153, 0.3);
        }
        .shadow-glow-sm {
          box-shadow: 0 0 10px 1px rgba(236, 72, 153, 0.2);
        }
        .shadow-glow-md {
          box-shadow: 0 0 20px 1px rgba(236, 72, 153, 0.15);
        }
      `}</style>
    </section>
  )
}