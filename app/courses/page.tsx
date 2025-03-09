"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { CourseCard } from "@/components/course-card"

const courses = [
  {
    id: "1",
    title: "Web3 Fundamentals",
    description: "Learn the core concepts of blockchain technology, cryptocurrencies, and decentralized applications.",
    instructor: "Dr. Sarah Chen",
    level: "Beginner",
    duration: "4 weeks",
    rating: 4.9,
    students: 2543,
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741502115/b9ubrhpsdl2z6yua2dm5.jpg?height=200&width=400",
    tags: ["Blockchain", "Cryptocurrency", "DeFi"],
    category: "web3",
  },
  {
    id: "2",
    title: "Smart Contract Development",
    description: "Master Solidity programming to write, test, and deploy secure smart contracts on Ethereum.",
    instructor: "Alex Rodriguez",
    level: "Intermediate",
    duration: "6 weeks",
    rating: 4.8,
    students: 1876,
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741502186/mdm3vpzvnynaupzvcjk4.jpg",
    tags: ["Solidity", "Ethereum", "Smart Contracts"],
    category: "development",
  },
  {
    id: "3",
    title: "Decentralized Application Development",
    description: "Build full-stack dApps using React, Next.js, and Web3.js/Ethers.js.",
    instructor: "Emma Thompson",
    level: "Advanced",
    duration: "8 weeks",
    rating: 4.7,
    students: 1245,
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741502244/wugphm377lal2qpyktpf.jpg?height=200&width=400",
    tags: ["React", "Next.js", "Web3.js"],
    category: "development",
  },
  {
    id: "4",
    title: "NFT Creation and Marketplaces",
    description: "Learn to create, mint, and sell NFTs on various marketplaces and build your own NFT projects.",
    instructor: "Michael Jordan",
    level: "Intermediate",
    duration: "5 weeks",
    rating: 4.6,
    students: 1987,
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741502329/gfwez1zowh6i4ftwqqfh.jpg?height=200&width=400",
    tags: ["NFT", "Digital Art", "Marketplaces"],
    category: "web3",
  },
  {
    id: "5",
    title: "Blockchain Security",
    description:
      "Understand security vulnerabilities in blockchain applications and learn best practices for secure development.",
    instructor: "Dr. Priya Sharma",
    level: "Advanced",
    duration: "7 weeks",
    rating: 4.9,
    students: 1123,
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741502389/kupr5xablsc3tu32nasc.jpg?height=200&width=400",
    tags: ["Security", "Audit", "Vulnerabilities"],
    category: "security",
  },
  {
    id: "6",
    title: "DeFi Fundamentals",
    description: "Explore decentralized finance protocols, lending, borrowing, yield farming, and liquidity pools.",
    instructor: "James Wilson",
    level: "Intermediate",
    duration: "6 weeks",
    rating: 4.8,
    students: 2134,
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741501854/bvvxicux8thkn4f0xmio.jpg?height=200&width=400",
    tags: ["DeFi", "Yield Farming", "Liquidity"],
    category: "finance",
  },
]

export default function CoursesPage() {
  const [mounted, setMounted] = useState(false)
  const [animationState, setAnimationState] = useState(0)
  
  // Handle client-side mounting
  useEffect(() => {
    setMounted(true)
    
    // Animation state cycle
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3)
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])
  
  return (
    <div className="w-full py-8 md:py-12 relative overflow-hidden">
      {/* Animated background - only rendered client-side */}
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
        <div className="flex flex-col gap-8">
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
                Web3 Education Platform
              </motion.span>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white">Courses</h1>
            <p className="text-gray-300 max-w-[800px]">
              Expand your Web3 knowledge and skills with our curated courses. From blockchain basics to advanced smart
              contract development, find the perfect learning path for your goals.
            </p>
          </motion.div>

          <motion.div 
            className="flex items-center gap-4 flex-col sm:flex-row"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <div className="relative w-full sm:w-80">
              <Input placeholder="Search courses..." className="pl-8 bg-black/50 border-gray-700 text-white" />
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
            <div className="grid grid-cols-2 gap-2 w-full sm:w-auto sm:flex">
              <Select defaultValue="all">
                <SelectTrigger className="w-full sm:w-[150px] bg-black/50 border-gray-700 text-white">
                  <SelectValue placeholder="Category" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 text-white border-gray-700">
                  <SelectItem value="all">All Categories</SelectItem>
                  <SelectItem value="web3">Web3</SelectItem>
                  <SelectItem value="development">Development</SelectItem>
                  <SelectItem value="security">Security</SelectItem>
                  <SelectItem value="finance">Finance</SelectItem>
                </SelectContent>
              </Select>
              <Select defaultValue="all">
                <SelectTrigger className="w-full sm:w-[150px] bg-black/50 border-gray-700 text-white">
                  <SelectValue placeholder="Level" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 text-white border-gray-700">
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="beginner">Beginner</SelectItem>
                  <SelectItem value="intermediate">Intermediate</SelectItem>
                  <SelectItem value="advanced">Advanced</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <Tabs defaultValue="all" className="w-full">
              <TabsList className="grid w-full grid-cols-5 mb-8 bg-black/40 text-white">
                <TabsTrigger value="all" className="data-[state=active]:bg-pink-600/20 data-[state=active]:text-white">All Courses</TabsTrigger>
                <TabsTrigger value="web3" className="data-[state=active]:bg-pink-600/20 data-[state=active]:text-white">Web3</TabsTrigger>
                <TabsTrigger value="development" className="data-[state=active]:bg-pink-600/20 data-[state=active]:text-white">Development</TabsTrigger>
                <TabsTrigger value="security" className="data-[state=active]:bg-pink-600/20 data-[state=active]:text-white">Security</TabsTrigger>
                <TabsTrigger value="finance" className="data-[state=active]:bg-pink-600/20 data-[state=active]:text-white">Finance</TabsTrigger>
              </TabsList>
              <TabsContent value="all">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {courses.map((course, index) => (
                    <motion.div
                      key={course.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 0.1 * index }}
                    >
                      <CourseCard course={course} />
                    </motion.div>
                  ))}
                </div>
              </TabsContent>
              <TabsContent value="web3">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {courses
                    .filter((course) => course.category === "web3")
                    .map((course, index) => (
                      <motion.div
                        key={course.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 * index }}
                      >
                        <CourseCard course={course} />
                      </motion.div>
                    ))}
                </div>
              </TabsContent>
              <TabsContent value="development">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {courses
                    .filter((course) => course.category === "development")
                    .map((course, index) => (
                      <motion.div
                        key={course.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 * index }}
                      >
                        <CourseCard course={course} />
                      </motion.div>
                    ))}
                </div>
              </TabsContent>
              <TabsContent value="security">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {courses
                    .filter((course) => course.category === "security")
                    .map((course, index) => (
                      <motion.div
                        key={course.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 * index }}
                      >
                        <CourseCard course={course} />
                      </motion.div>
                    ))}
                </div>
              </TabsContent>
              <TabsContent value="finance">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {courses
                    .filter((course) => course.category === "finance")
                    .map((course, index) => (
                      <motion.div
                        key={course.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 * index }}
                      >
                        <CourseCard course={course} />
                      </motion.div>
                    ))}
                </div>
              </TabsContent>
            </Tabs>
          </motion.div>
        </div>
      </div>
    </div>
  )
}