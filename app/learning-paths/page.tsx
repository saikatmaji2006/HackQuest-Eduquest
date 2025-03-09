"use client"

import { useState, useEffect } from "react"
import Image from "next/image"
import Link from "next/link"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const learningPaths = [
  {
    id: "blockchain-developer",
    title: "Blockchain Developer",
    description: "Master blockchain fundamentals and smart contract development to build decentralized applications.",
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741504594/s2slbvyixilxowp2luv3.jpg?height=200&width=400",
    duration: "6 months",
    level: "Beginner to Advanced",
    courses: 12,
    certifications: 3,
    tags: ["Solidity", "Web3.js", "Ethereum", "Smart Contracts"],
    category: "development",
    popularity: "high",
    stages: [
      {
        id: "1",
        title: "Blockchain Fundamentals",
        courses: [
          { id: "101", title: "Introduction to Blockchain", duration: "4 weeks", completed: true },
          { id: "102", title: "Cryptography Basics", duration: "3 weeks", completed: true },
          { id: "103", title: "Consensus Mechanisms", duration: "3 weeks", completed: false },
        ],
        progress: 66,
      },
      {
        id: "2",
        title: "Smart Contract Development",
        courses: [
          { id: "201", title: "Solidity Programming", duration: "6 weeks", completed: false },
          { id: "202", title: "Smart Contract Security", duration: "4 weeks", completed: false },
          { id: "203", title: "Testing and Deployment", duration: "3 weeks", completed: false },
        ],
        progress: 0,
      },
      {
        id: "3",
        title: "dApp Development",
        courses: [
          { id: "301", title: "Web3.js & Ethers.js", duration: "4 weeks", completed: false },
          { id: "302", title: "Frontend Integration", duration: "5 weeks", completed: false },
          { id: "303", title: "Full-Stack dApp Project", duration: "6 weeks", completed: false },
        ],
        progress: 0,
      },
    ],
  },
  {
    id: "defi-specialist",
    title: "DeFi Specialist",
    description:
      "Understand decentralized finance protocols, economics, and strategies for building DeFi applications.",
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741504534/zj7ghi1af3wqd86uyrt9.jpg?height=200&width=400",
    duration: "5 months",
    level: "Intermediate to Advanced",
    courses: 10,
    certifications: 2,
    tags: ["DeFi", "Lending", "Yield Farming", "Tokenomics"],
    category: "finance",
    popularity: "medium",
    stages: [
      {
        id: "1",
        title: "DeFi Fundamentals",
        courses: [
          { id: "401", title: "Introduction to DeFi", duration: "3 weeks", completed: false },
          { id: "402", title: "DeFi Protocols Overview", duration: "4 weeks", completed: false },
          { id: "403", title: "Tokenomics", duration: "3 weeks", completed: false },
        ],
        progress: 0,
      },
      {
        id: "2",
        title: "Advanced DeFi Concepts",
        courses: [
          { id: "501", title: "Lending and Borrowing", duration: "4 weeks", completed: false },
          { id: "502", title: "Yield Farming Strategies", duration: "4 weeks", completed: false },
          { id: "503", title: "Liquidity Pools & AMMs", duration: "3 weeks", completed: false },
        ],
        progress: 0,
      },
      {
        id: "3",
        title: "DeFi Development",
        courses: [
          { id: "601", title: "Building DeFi Applications", duration: "6 weeks", completed: false },
          { id: "602", title: "DeFi Security & Auditing", duration: "4 weeks", completed: false },
        ],
        progress: 0,
      },
    ],
  },
  {
    id: "nft-creator",
    title: "NFT Creator & Developer",
    description:
      "Learn to create, mint, and build marketplaces for non-fungible tokens with technical and creative skills.",
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741504361/sttcj1napyw7oahbpm9p.jpg?height=200&width=400",
    duration: "4 months",
    level: "Beginner to Intermediate",
    courses: 8,
    certifications: 2,
    tags: ["NFT", "Digital Art", "Marketplaces", "ERC-721"],
    category: "creative",
    popularity: "high",
    stages: [
      {
        id: "1",
        title: "NFT Fundamentals",
        courses: [
          { id: "701", title: "Introduction to NFTs", duration: "3 weeks", completed: false },
          { id: "702", title: "NFT Standards (ERC-721, ERC-1155)", duration: "3 weeks", completed: false },
          { id: "703", title: "Digital Art Creation", duration: "4 weeks", completed: false },
        ],
        progress: 0,
      },
      {
        id: "2",
        title: "NFT Development",
        courses: [
          { id: "801", title: "Creating & Minting NFTs", duration: "4 weeks", completed: false },
          { id: "802", title: "NFT Metadata & Storage", duration: "3 weeks", completed: false },
        ],
        progress: 0,
      },
      {
        id: "3",
        title: "NFT Marketplaces",
        courses: [
          { id: "901", title: "Marketplace Integration", duration: "3 weeks", completed: false },
          { id: "902", title: "Building an NFT Marketplace", duration: "6 weeks", completed: false },
        ],
        progress: 0,
      },
    ],
  },
  {
    id: "web3-security",
    title: "Web3 Security Specialist",
    description: "Master the security aspects of blockchain applications, smart contracts, and Web3 infrastructure.",
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741501854/bvvxicux8thkn4f0xmio.jpg?height=200&width=400",
    duration: "7 months",
    level: "Intermediate to Advanced",
    courses: 14,
    certifications: 3,
    tags: ["Security", "Auditing", "Vulnerabilities", "Penetration Testing"],
    category: "security",
    popularity: "medium",
    stages: [
      {
        id: "1",
        title: "Security Fundamentals",
        courses: [
          { id: "1001", title: "Introduction to Cybersecurity", duration: "3 weeks", completed: false },
          { id: "1002", title: "Blockchain Security Basics", duration: "4 weeks", completed: false },
        ],
        progress: 0,
      },
      {
        id: "2",
        title: "Smart Contract Security",
        courses: [
          { id: "1101", title: "Common Vulnerabilities", duration: "5 weeks", completed: false },
          { id: "1102", title: "Security Best Practices", duration: "4 weeks", completed: false },
          { id: "1103", title: "Audit Methodologies", duration: "4 weeks", completed: false },
        ],
        progress: 0,
      },
      {
        id: "3",
        title: "Advanced Security",
        courses: [
          { id: "1201", title: "Penetration Testing", duration: "5 weeks", completed: false },
          { id: "1202", title: "Security Tools & Frameworks", duration: "4 weeks", completed: false },
          { id: "1203", title: "Real-world Security Audits", duration: "6 weeks", completed: false },
        ],
        progress: 0,
      },
    ],
  },
  {
    id: "web3-business",
    title: "Web3 Business & Strategy",
    description: "Learn to develop business strategies, tokenomics, and governance models for Web3 projects.",
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741502115/b9ubrhpsdl2z6yua2dm5.jpg?height=200&width=400",
    duration: "5 months",
    level: "Beginner to Advanced",
    courses: 10,
    certifications: 2,
    tags: ["Business", "Tokenomics", "Governance", "Strategy"],
    category: "business",
    popularity: "low",
    stages: [
      {
        id: "1",
        title: "Web3 Business Fundamentals",
        courses: [
          { id: "1301", title: "Introduction to Web3 Business Models", duration: "3 weeks", completed: false },
          { id: "1302", title: "Blockchain Economics", duration: "4 weeks", completed: false },
        ],
        progress: 0,
      },
      {
        id: "2",
        title: "Tokenomics & Governance",
        courses: [
          { id: "1401", title: "Token Design Principles", duration: "4 weeks", completed: false },
          { id: "1402", title: "DAO Governance Models", duration: "3 weeks", completed: false },
          { id: "1403", title: "Incentive Mechanisms", duration: "3 weeks", completed: false },
        ],
        progress: 0,
      },
      {
        id: "3",
        title: "Web3 Strategy",
        courses: [
          { id: "1501", title: "Go-to-Market Strategies", duration: "4 weeks", completed: false },
          { id: "1502", title: "Community Building", duration: "3 weeks", completed: false },
          { id: "1503", title: "Web3 Project Management", duration: "4 weeks", completed: false },
        ],
        progress: 0,
      },
    ],
  },
]

export default function LearningPathsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [levelFilter, setLevelFilter] = useState("all")
  const [durationFilter, setDurationFilter] = useState("all")
  const [categoryFilter, setCategoryFilter] = useState("all")
  const [mounted, setMounted] = useState(false)
  const [animationState, setAnimationState] = useState(0)

  // Update animation state periodically
  useEffect(() => {
    setMounted(true)
    
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3)
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])

  const filteredPaths = learningPaths.filter((path) => {
    // Search filter
    if (
      searchQuery &&
      !path.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !path.description.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !path.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    ) {
      return false
    }

    // Level filter
    if (levelFilter !== "all") {
      if (levelFilter === "beginner" && !path.level.toLowerCase().includes("beginner")) {
        return false
      }
      if (levelFilter === "intermediate" && !path.level.toLowerCase().includes("intermediate")) {
        return false
      }
      if (levelFilter === "advanced" && !path.level.toLowerCase().includes("advanced")) {
        return false
      }
    }

    // Duration filter
    if (durationFilter !== "all") {
      const months = Number.parseInt(path.duration.split(" ")[0])
      if (durationFilter === "short" && months > 4) return false
      if (durationFilter === "medium" && (months < 5 || months > 6)) return false
      if (durationFilter === "long" && months < 7) return false
    }

    // Category filter
    if (categoryFilter !== "all" && path.category !== categoryFilter) {
      return false
    }

    return true
  })

  return (
    <div className="w-full min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated Background */}
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
                x: Math.random() * (typeof window !== 'undefined' ? window.innerWidth : 1000),
                y: Math.random() * (typeof window !== 'undefined' ? window.innerHeight : 800),
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

      {/* Main Content */}
      <div className="container py-8 md:py-12 relative z-10">
        <div className="flex flex-col gap-8">
          <motion.div 
            className="space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div className="flex items-center mb-4">
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
                Web3 Education
              </motion.span>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-600">
                Learning Paths
              </span>
            </h1>
            <p className="text-gray-300 max-w-[800px]">
              Follow structured learning journeys designed to take you from novice to expert in your chosen Web3
              specialization. Each path includes curated courses, hands-on projects, and certification opportunities.
            </p>
          </motion.div>

          <div className="flex flex-col md:flex-row gap-6">
            <motion.div 
              className="w-full md:w-64 space-y-4"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <div className="relative">
                <Input
                  placeholder="Search paths..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8 bg-gray-900/60 border-gray-700 text-white"
                />
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
                  className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-400"
                >
                  <circle cx="11" cy="11" r="8" />
                  <path d="m21 21-4.3-4.3" />
                </svg>
              </div>

              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Level</label>
                  <Select value={levelFilter} onValueChange={setLevelFilter}>
                    <SelectTrigger className="bg-gray-900/60 border-gray-700 text-white">
                      <SelectValue placeholder="Select level" />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-900 border-gray-700 text-white">
                      <SelectItem value="all">All Levels</SelectItem>
                      <SelectItem value="beginner">Beginner</SelectItem>
                      <SelectItem value="intermediate">Intermediate</SelectItem>
                      <SelectItem value="advanced">Advanced</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Duration</label>
                  <Select value={durationFilter} onValueChange={setDurationFilter}>
                    <SelectTrigger className="bg-gray-900/60 border-gray-700 text-white">
                      <SelectValue placeholder="Select duration" />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-900 border-gray-700 text-white">
                      <SelectItem value="all">All Durations</SelectItem>
                      <SelectItem value="short">Short (1-4 months)</SelectItem>
                      <SelectItem value="medium">Medium (5-6 months)</SelectItem>
                      <SelectItem value="long">Long (7+ months)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Category</label>
                  <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                    <SelectTrigger className="bg-gray-900/60 border-gray-700 text-white">
                      <SelectValue placeholder="Select category" />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-900 border-gray-700 text-white">
                      <SelectItem value="all">All Categories</SelectItem>
                      <SelectItem value="development">Development</SelectItem>
                      <SelectItem value="finance">Finance</SelectItem>
                      <SelectItem value="creative">Creative</SelectItem>
                      <SelectItem value="security">Security</SelectItem>
                      <SelectItem value="business">Business</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  variant="outline"
                  className="w-full bg-gray-800/60 text-gray-300 border-gray-700 hover:bg-gray-700"
                  onClick={() => {
                    setSearchQuery("")
                    setLevelFilter("all")
                    setDurationFilter("all")
                    setCategoryFilter("all")
                  }}
                >
                  Reset Filters
                </Button>
              </div>
            </motion.div>

            <motion.div 
              className="flex-1"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <Tabs defaultValue="all" className="w-full">
                <TabsList className="grid w-full grid-cols-3 mb-8 bg-gray-900/60">
                  <TabsTrigger value="all" className="data-[state=active]:bg-gray-800 text-gray-300">All Paths</TabsTrigger>
                  <TabsTrigger value="popular" className="data-[state=active]:bg-gray-800 text-gray-300">Most Popular</TabsTrigger>
                  <TabsTrigger value="new" className="data-[state=active]:bg-gray-800 text-gray-300">Newest</TabsTrigger>
                </TabsList>

                <TabsContent value="all" className="space-y-6">
                  {filteredPaths.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {filteredPaths.map((path, index) => (
                        <motion.div
                          key={path.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.5, delay: 0.1 * index }}
                        >
                          <LearningPathCard path={path} />
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <motion.div 
                      className="text-center py-12"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                    >
                      <h3 className="text-lg font-medium mb-2 text-white">No learning paths found</h3>
                      <p className="text-gray-400 mb-4">Try adjusting your filters or search query</p>
                      <Button
                        variant="outline"
                        className="bg-gray-800/60 text-gray-300 border-gray-700 hover:bg-gray-700"
                        onClick={() => {
                          setSearchQuery("")
                          setLevelFilter("all")
                          setDurationFilter("all")
                          setCategoryFilter("all")
                        }}
                      >
                        Reset Filters
                      </Button>
                    </motion.div>
                  )}
                </TabsContent>

                <TabsContent value="popular" className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {filteredPaths
                      .filter((path) => path.popularity === "high")
                      .map((path, index) => (
                        <motion.div
                          key={path.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.5, delay: 0.1 * index }}
                        >
                          <LearningPathCard key={path.id} path={path} />
                        </motion.div>
                      ))}
                  </div>
                </TabsContent>

                <TabsContent value="new" className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {filteredPaths.slice(0, 4).map((path, index) => (
                      <motion.div
                        key={path.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 * index }}
                      >
                        <LearningPathCard key={path.id} path={path} />
                      </motion.div>
                    ))}
                  </div>
                </TabsContent>
              </Tabs>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}

function LearningPathCard({ path }) {
  const overallProgress = path.stages.reduce((acc, stage) => {
    return acc + stage.progress / path.stages.length
  }, 0)

  return (
    <Card className="overflow-hidden flex flex-col h-full border-gray-700 bg-gray-900/40 backdrop-blur-sm hover:bg-gray-800/60 transition-colors">
      <div className="relative aspect-video overflow-hidden">
        <Image
          src={path.image || "/placeholder.svg"}
          alt={path.title}
          fill
          className="object-cover transition-transform hover:scale-105"
        />
        {path.popularity === "high" && (
          <div className="absolute top-2 right-2">
            <Badge className="bg-gradient-to-r from-pink-500 to-purple-600 text-white border-0">
              Popular
            </Badge>
          </div>
        )}
      </div>
      <CardHeader className="pb-2">
        <CardTitle className="line-clamp-1 text-white">{path.title}</CardTitle>
        <CardDescription className="line-clamp-2 text-gray-300">{path.description}</CardDescription>
      </CardHeader>
      <CardContent className="pb-4 flex-grow">
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col">
              <span className="text-xs text-gray-400">Duration</span>
              <span className="font-medium text-gray-200">{path.duration}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-gray-400">Level</span>
              <span className="font-medium text-gray-200">{path.level}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-gray-400">Courses</span>
              <span className="font-medium text-gray-200">{path.courses}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-gray-400">Certifications</span>
              <span className="font-medium text-gray-200">{path.certifications}</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Progress</span>
              <span className="text-sm font-medium text-gray-200">{Math.round(overallProgress)}%</span>
            </div>
            <Progress value={overallProgress} className="h-2 bg-gray-700">
              <div 
                className="h-full bg-gradient-to-r from-pink-500 to-purple-600 rounded-full"
                style={{ width: `${overallProgress}%` }}
              />
            </Progress>
          </div>

          <div className="flex flex-wrap gap-1">
            {path.tags.map((tag) => (
              <Badge key={tag} variant="secondary" className="text-xs bg-gray-800 text-gray-200 hover:bg-gray-700">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
      <CardFooter className="pt-0">
        <Button asChild className="w-full bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white border-0">
          <Link href={`/learning-paths/${path.id}`}>View Path</Link>
        </Button>
      </CardFooter>
    </Card>
  )
}