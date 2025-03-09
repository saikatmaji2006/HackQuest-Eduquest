"use client"

import { useState, useEffect } from "react"
import { useParams } from "next/navigation"
import Image from "next/image"
import Link from "next/link"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"

// This would typically come from an API or database
const learningPaths = {
  "blockchain-developer": {
    id: "blockchain-developer",
    title: "Blockchain Developer",
    description: "Master blockchain fundamentals and smart contract development to build decentralized applications.",
    longDescription:
      "This comprehensive learning path will take you from blockchain basics to advanced dApp development. You'll learn the theoretical foundations of blockchain technology, master Solidity programming for smart contracts, and build full-stack decentralized applications. By the end of this path, you'll have the skills and portfolio needed to pursue a career as a blockchain developer.",
    image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741504361/sttcj1napyw7oahbpm9p.jpg?height=400&width=800",
    duration: "6 months",
    level: "Beginner to Advanced",
    courses: 12,
    certifications: 3,
    students: 2547,
    rating: 4.8,
    tags: ["Solidity", "Web3.js", "Ethereum", "Smart Contracts"],
    prerequisites: [
      "Basic programming knowledge",
      "Understanding of web development concepts",
      "Familiarity with JavaScript (recommended)",
    ],
    outcomes: [
      "Build and deploy smart contracts on Ethereum and other blockchains",
      "Develop full-stack decentralized applications",
      "Implement security best practices for blockchain applications",
      "Integrate Web3 wallets and blockchain functionality into web applications",
    ],
    stages: [
      {
        id: "1",
        title: "Blockchain Fundamentals",
        description:
          "Learn the core concepts of blockchain technology, including distributed ledgers, consensus mechanisms, and cryptography.",
        courses: [
          {
            id: "101",
            title: "Introduction to Blockchain",
            description: "Understand the basics of blockchain technology, its history, and how it works.",
            duration: "4 weeks",
            completed: true,
            lessons: 12,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "102",
            title: "Cryptography Basics",
            description: "Learn about the cryptographic principles that power blockchain security.",
            duration: "3 weeks",
            completed: true,
            lessons: 9,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "103",
            title: "Consensus Mechanisms",
            description: "Explore different consensus algorithms like Proof of Work, Proof of Stake, and more.",
            duration: "3 weeks",
            completed: false,
            lessons: 10,
            image: "/placeholder.svg?height=200&width=300",
          },
        ],
        progress: 66,
      },
      {
        id: "2",
        title: "Smart Contract Development",
        description: "Master Solidity programming to write, test, and deploy secure smart contracts on Ethereum.",
        courses: [
          {
            id: "201",
            title: "Solidity Programming",
            description: "Learn the Solidity programming language for writing smart contracts.",
            duration: "6 weeks",
            completed: false,
            lessons: 18,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "202",
            title: "Smart Contract Security",
            description: "Understand common vulnerabilities and security best practices for smart contracts.",
            duration: "4 weeks",
            completed: false,
            lessons: 14,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "203",
            title: "Testing and Deployment",
            description: "Learn how to test, deploy, and verify smart contracts on various networks.",
            duration: "3 weeks",
            completed: false,
            lessons: 12,
            image: "/placeholder.svg?height=200&width=300",
          },
        ],
        progress: 0,
      },
      {
        id: "3",
        title: "dApp Development",
        description:
          "Build full-stack decentralized applications using Web3.js, Ethers.js, and modern frontend frameworks.",
        courses: [
          {
            id: "301",
            title: "Web3.js & Ethers.js",
            description: "Master the JavaScript libraries for interacting with the Ethereum blockchain.",
            duration: "4 weeks",
            completed: false,
            lessons: 15,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "302",
            title: "Frontend Integration",
            description: "Learn to integrate blockchain functionality into React and Next.js applications.",
            duration: "5 weeks",
            completed: false,
            lessons: 16,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "303",
            title: "Full-Stack dApp Project",
            description: "Build a complete decentralized application from scratch as your capstone project.",
            duration: "6 weeks",
            completed: false,
            lessons: 20,
            image: "/placeholder.svg?height=200&width=300",
          },
        ],
        progress: 0,
      },
    ],
    instructors: [
      {
        name: "Dr. Sarah Chen",
        role: "Lead Blockchain Instructor",
        bio: "Former professor of Computer Science with 8+ years of experience in blockchain development.",
        image: "/placeholder.svg?height=100&width=100",
      },
      {
        name: "Alex Rodriguez",
        role: "Smart Contract Specialist",
        bio: "Solidity developer who has audited over 50 production smart contracts for major DeFi protocols.",
        image: "/placeholder.svg?height=100&width=100",
      },
    ],
  },
  "defi-specialist": {
    id: "defi-specialist",
    title: "DeFi Specialist",
    description:
      "Understand decentralized finance protocols, economics, and strategies for building DeFi applications.",
    longDescription:
      "This specialized learning path focuses on the rapidly evolving world of Decentralized Finance (DeFi). You'll learn about lending protocols, automated market makers, yield farming strategies, and the economic principles behind DeFi. By completing this path, you'll be equipped to build, analyze, and contribute to DeFi protocols and applications.",
    image: "/placeholder.svg?height=400&width=800",
    duration: "5 months",
    level: "Intermediate to Advanced",
    courses: 10,
    certifications: 2,
    students: 1876,
    rating: 4.7,
    tags: ["DeFi", "Lending", "Yield Farming", "Tokenomics"],
    prerequisites: [
      "Basic understanding of blockchain technology",
      "Familiarity with cryptocurrencies and tokens",
      "Basic knowledge of finance concepts (recommended)",
    ],
    outcomes: [
      "Understand the mechanics of various DeFi protocols",
      "Analyze and evaluate DeFi investment strategies",
      "Build and deploy DeFi applications",
      "Implement security best practices for DeFi protocols",
    ],
    stages: [
      {
        id: "1",
        title: "DeFi Fundamentals",
        description: "Learn the core concepts of decentralized finance, including key protocols and use cases.",
        courses: [
          {
            id: "401",
            title: "Introduction to DeFi",
            description: "Understand the basics of decentralized finance and how it differs from traditional finance.",
            duration: "3 weeks",
            completed: false,
            lessons: 10,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "402",
            title: "DeFi Protocols Overview",
            description: "Explore major DeFi protocols including lending platforms, DEXs, and yield aggregators.",
            duration: "4 weeks",
            completed: false,
            lessons: 12,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "403",
            title: "Tokenomics",
            description: "Learn about token economics, governance, and incentive mechanisms in DeFi.",
            duration: "3 weeks",
            completed: false,
            lessons: 9,
            image: "/placeholder.svg?height=200&width=300",
          },
        ],
        progress: 0,
      },
      {
        id: "2",
        title: "Advanced DeFi Concepts",
        description: "Dive deeper into specific DeFi mechanisms and strategies for yield optimization.",
        courses: [
          {
            id: "501",
            title: "Lending and Borrowing",
            description: "Understand the mechanics of decentralized lending and borrowing protocols.",
            duration: "4 weeks",
            completed: false,
            lessons: 14,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "502",
            title: "Yield Farming Strategies",
            description: "Learn various yield farming strategies and how to evaluate risk and return.",
            duration: "4 weeks",
            completed: false,
            lessons: 12,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "503",
            title: "Liquidity Pools & AMMs",
            description: "Master the concepts of liquidity provision and automated market makers.",
            duration: "3 weeks",
            completed: false,
            lessons: 10,
            image: "/placeholder.svg?height=200&width=300",
          },
        ],
        progress: 0,
      },
      {
        id: "3",
        title: "DeFi Development",
        description: "Build and deploy your own DeFi applications with a focus on security and efficiency.",
        courses: [
          {
            id: "601",
            title: "Building DeFi Applications",
            description: "Learn to develop secure and efficient DeFi protocols and applications.",
            duration: "6 weeks",
            completed: false,
            lessons: 18,
            image: "/placeholder.svg?height=200&width=300",
          },
          {
            id: "602",
            title: "DeFi Security & Auditing",
            description: "Understand common vulnerabilities and security best practices for DeFi protocols.",
            duration: "4 weeks",
            completed: false,
            lessons: 14,
            image: "/placeholder.svg?height=200&width=300",
          },
        ],
        progress: 0,
      },
    ],
    instructors: [
      {
        name: "Emma Thompson",
        role: "DeFi Researcher",
        bio: "Former quantitative analyst who has been researching and building in DeFi since 2018.",
        image: "/placeholder.svg?height=100&width=100",
      },
      {
        name: "Michael Jordan",
        role: "Tokenomics Specialist",
        bio: "Economist specializing in token design and incentive mechanisms for blockchain protocols.",
        image: "/placeholder.svg?height=100&width=100",
      },
    ],
  },
}

export default function LearningPathDetailPage() {
  const { id } = useParams()
  const [activeTab, setActiveTab] = useState("overview")
  const [mounted, setMounted] = useState(false)
  const [animationState, setAnimationState] = useState(0)

  // Set mounted state once component is mounted (for client-side rendering)
  useEffect(() => {
    setMounted(true)
    
    // Animation state cycle
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3)
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])

  // In a real app, you would fetch this data from an API
  const path = learningPaths[id as string]

  if (!path) {
    return (
      <div className="container py-12 text-center">
        <h1 className="text-2xl font-bold mb-4">Learning Path Not Found</h1>
        <p className="text-muted-foreground mb-6">
          The learning path you're looking for doesn't exist or has been removed.
        </p>
        <Button asChild>
          <Link href="/learning-paths">View All Learning Paths</Link>
        </Button>
      </div>
    )
  }

  const overallProgress = path.stages.reduce((acc, stage) => {
    return acc + stage.progress / path.stages.length
  }, 0)

  return (
    <div className="w-full py-8 md:py-12 relative overflow-hidden">
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
        <div className="flex flex-col gap-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div>
                <div className="flex items-center gap-2 mb-2 text-white">
                  <Link href="/learning-paths" className="text-sm text-gray-300 hover:text-primary">
                    Learning Paths
                  </Link>
                  <span className="text-sm text-gray-400">/</span>
                  <span className="text-sm text-white">{path.title}</span>
                </div>
                <h1 className="text-3xl font-bold tracking-tight text-white">{path.title}</h1>
                <p className="text-gray-300 mt-2">{path.description}</p>
              </div>
              
              <div className="relative aspect-video rounded-lg overflow-hidden">
                <Image 
                  src={path.image || "/placeholder.svg"} 
                  alt={path.title}
                  fill
                  className="object-cover"
                />
              </div>
              
              <Card className="bg-black/50 border-gray-800">
                <CardContent className="p-0">
                  <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                    <TabsList className="grid w-full grid-cols-4 bg-gray-900/70 p-1 rounded-t-lg">
                      <TabsTrigger value="overview" className="text-white data-[state=active]:bg-gray-800">Overview</TabsTrigger>
                      <TabsTrigger value="curriculum" className="text-white data-[state=active]:bg-gray-800">Curriculum</TabsTrigger>
                      <TabsTrigger value="instructors" className="text-white data-[state=active]:bg-gray-800">Instructors</TabsTrigger>
                      <TabsTrigger value="reviews" className="text-white data-[state=active]:bg-gray-800">Reviews</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="overview" className="space-y-6 pt-4 p-6 text-white">
                      <div>
                        <h2 className="text-xl font-bold mb-3 text-white">About This Learning Path</h2>
                        <p className="text-gray-300">{path.longDescription}</p>
                      </div>
                      
                      <div>
                        <h3 className="text-lg font-bold mb-3 text-white">What You'll Learn</h3>
                        <ul className="space-y-2">
                          {path.outcomes.map((outcome, index) => (
                            <li key={index} className="flex items-start gap-2">
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
                                className="h-5 w-5 text-pink-500 mt-0.5 flex-shrink-0"
                              >
                                <polyline points="20 6 9 17 4 12" />
                              </svg>
                              <span className="text-gray-300">{outcome}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h3 className="text-lg font-bold mb-3 text-white">Prerequisites</h3>
                        <ul className="space-y-2">
                          {path.prerequisites.map((prerequisite, index) => (
                            <li key={index} className="flex items-start gap-2">
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
                                className="h-5 w-5 text-gray-400 mt-0.5 flex-shrink-0"
                              >
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" x2="12" y1="8" y2="16" />
                                <line x1="8" x2="16" y1="12" y2="12" />
                              </svg>
                              <span className="text-gray-300">{prerequisite}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </TabsContent>

                    <TabsContent value="curriculum" className="space-y-6 pt-4 p-6 text-white">
                      <div>
                        <h2 className="text-xl font-bold mb-3 text-white">Learning Path Curriculum</h2>
                        <p className="text-gray-300 mb-6">
                          This learning path consists of {path.stages.length} stages and {path.courses} courses, 
                          designed to take you from {path.level.split(" to ")[0].toLowerCase()} to {path.level.split(" to ")[1].toLowerCase()} level.
                        </p>

                        <Accordion type="multiple" className="w-full">
                          {path.stages.map((stage, index) => (
                            <AccordionItem key={stage.id} value={stage.id} className="border-gray-700">
                              <AccordionTrigger className="hover:no-underline text-white py-4">
                                <div className="flex items-center gap-4">
                                  <div className={`flex items-center justify-center w-8 h-8 rounded-full text-primary-foreground ${stage.progress > 0 ? 'bg-pink-500' : 'bg-gray-700'}`}>
                                    {index + 1}
                                  </div>
                                  <div className="text-left">
                                    <h3 className="font-medium text-base text-white">{stage.title}</h3>
                                    <p className="text-sm text-gray-400">{stage.courses.length} courses • {stage.progress}% complete</p>
                                  </div>
                                </div>
                              </AccordionTrigger>
                              <AccordionContent>
                                <div className="pl-12 space-y-4">
                                  <p className="text-sm text-gray-300">{stage.description}</p>
                                  
                                  <div className="space-y-2 mb-4">
                                    <div className="flex items-center justify-between">
                                      <span className="text-sm text-gray-400">Progress</span>
                                      <span className="text-sm font-medium text-white">{stage.progress}%</span>
                                    </div>
                                    <Progress value={stage.progress} className="h-2 bg-gray-700" indicatorClassName="bg-gradient-to-r from-pink-500 to-purple-600" />
                                  </div>
                                  
                                  <div className="space-y-4">
                                    {stage.courses.map((course) => (
                                      <Card key={course.id} className={`border-gray-700 ${course.completed ? 'bg-gray-800/50' : 'bg-black/30'}`}>
                                        <CardContent className="p-4">
                                          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
                                            <div className="relative h-20 w-32 rounded-md overflow-hidden flex-shrink-0">
                                              <Image 
                                                src={course.image || "/placeholder.svg"} 
                                                alt={course.title}
                                                fill
                                                className="object-cover"
                                              />
                                            </div>
                                            <div className="flex-grow">
                                              <div className="flex items-start justify-between">
                                                <div>
                                                  <h4 className="font-medium text-white">{course.title}</h4>
                                                  <p className="text-sm text-gray-300">{course.description}</p>
                                                  <div className="flex items-center gap-4 mt-1">
                                                    <span className="text-xs text-gray-400">{course.duration}</span>
                                                    <span className="text-xs text-gray-400">{course.lessons} lessons</span>
                                                  </div>
                                                </div>
                                                <Button 
                                                  size="sm" 
                                                  variant={course.completed ? "outline" : "default"}
                                                  className={course.completed ? "border-gray-600 text-white hover:bg-gray-700" : "bg-gradient-to-r from-pink-500 to-purple-600 text-white hover:opacity-90"}
                                                >
                                                  {course.completed ? "Review" : "Start"}
                                                </Button>
                                              </div>
                                            </div>
                                          </div>
                                        </CardContent>
                                      </Card>
                                    ))}
                                  </div>
                                </div>
                              </AccordionContent>
                            </AccordionItem>
                          ))}
                        </Accordion>
                      </div>
                    </TabsContent>
                    
                    <TabsContent value="instructors" className="space-y-6 pt-4 p-6 text-white">
                      <div>
                        <h2 className="text-xl font-bold mb-3 text-white">Your Instructors</h2>
                        <p className="text-gray-300 mb-6">
                          Learn from industry experts with real-world experience in blockchain and Web3 development.
                        </p>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          {path.instructors.map((instructor, index) => (
                            <Card key={index} className="border-gray-700 bg-black/30">
                              <CardContent className="p-6">
                                <div className="flex items-start gap-4">
                                  <div className="h-16 w-16 rounded-full overflow-hidden bg-gray-800 flex-shrink-0 relative">
                                    <Image
                                      src={instructor.image || "/placeholder.svg"}
                                      alt={instructor.name}
                                      fill
                                      className="object-cover"
                                    />
                                  </div>
                                  <div>
                                    <h3 className="font-bold text-white">{instructor.name}</h3>
                                    <p className="text-sm text-pink-400">{instructor.role}</p>
                                    <p className="text-sm text-gray-300 mt-2">{instructor.bio}</p>
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      </div>
                    </TabsContent>
                    
                    <TabsContent value="reviews" className="space-y-6 pt-4 p-6 text-white">
                      <div>
                        <h2 className="text-xl font-bold mb-3 text-white">Student Reviews</h2>
                        <div className="flex items-center gap-2 mb-6">
                          <div className="flex">
                            {[...Array(5)].map((_, i) => (
                              <svg
                                key={i}
                                xmlns="http://www.w3.org/2000/svg"
                                width="24"
                                height="24"
                                viewBox="0 0 24 24"
                                fill={i < Math.floor(path.rating) ? "currentColor" : "none"}
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="h-5 w-5 text-yellow-400"
                              >
                                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                              </svg>
                            ))}
                          </div>
                          <span className="font-medium text-white">{path.rating}</span>
                          <span className="text-gray-400">({path.students} students)</span>
                        </div>
                        
                        <div className="space-y-4">
                          {/* Sample reviews - in a real app, these would come from an API */}
                          {[
                            {
                              name: "Alex Johnson",
                              rating: 5,
                              date: "2 months ago",
                              comment: "This learning path completely transformed my understanding of blockchain development. The curriculum is well-structured and the instructors are incredibly knowledgeable. I was able to land a job as a junior blockchain developer after completing just the first two stages!"
                            },
                            {
                              name: "Maria Garcia",
                              rating: 4,
                              date: "3 months ago",
                              comment: "Great content and excellent instructors. The projects are challenging but very rewarding. I would have given 5 stars if there were more real-world examples in the smart contract section."
                            },
                            {
                              name: "David Kim",
                              rating:rating: 5,
                              date: "1 month ago",
                              comment: "I've tried several blockchain courses before, but this one stands out. The progressive approach from fundamentals to advanced topics made complex concepts much easier to grasp. The hands-on projects were particularly valuable for building my portfolio."
                            },
                          ].map((review, index) => (
                            <Card key={index} className="border-gray-700 bg-black/30">
                              <CardContent className="p-6">
                                <div className="flex items-start justify-between">
                                  <div className="flex items-center gap-2">
                                    <div className="h-10 w-10 rounded-full bg-gray-700 flex items-center justify-center">
                                      <span className="font-medium text-white">{review.name.charAt(0)}</span>
                                    </div>
                                    <div>
                                      <h4 className="font-medium text-white">{review.name}</h4>
                                      <p className="text-xs text-gray-400">{review.date}</p>
                                    </div>
                                  </div>
                                  <div className="flex">
                                    {[...Array(5)].map((_, i) => (
                                      <svg
                                        key={i}
                                        xmlns="http://www.w3.org/2000/svg"
                                        width="24"
                                        height="24"
                                        viewBox="0 0 24 24"
                                        fill={i < review.rating ? "currentColor" : "none"}
                                        stroke="currentColor"
                                        strokeWidth="2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        className="h-4 w-4 text-yellow-400"
                                      >
                                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                                      </svg>
                                    ))}
                                  </div>
                                </div>
                                <p className="text-gray-300 mt-3">{review.comment}</p>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                        
                        <div className="mt-6 text-center">
                          <Button variant="outline" className="border-gray-700 text-white hover:bg-gray-800">
                            Load More Reviews
                          </Button>
                        </div>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </div>
            
            {/* Sidebar */}
            <div className="space-y-6">
              <Card className="border-gray-800 bg-black/50 overflow-hidden">
                <div className="px-6 py-5 border-b border-gray-800">
                  <h3 className="text-xl font-bold text-white">Learning Path Progress</h3>
                </div>
                <CardContent className="p-6">
                  <div className="space-y-6">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-gray-300">Overall Progress</span>
                        <span className="font-medium text-white">{Math.round(overallProgress)}%</span>
                      </div>
                      <Progress value={overallProgress} className="h-2 bg-gray-700" indicatorClassName="bg-gradient-to-r from-pink-500 to-purple-600" />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-center">
                      <div className="bg-black/30 p-4 rounded-lg">
                        <h4 className="text-xl font-bold text-white">{path.duration}</h4>
                        <p className="text-sm text-gray-400">Duration</p>
                      </div>
                      <div className="bg-black/30 p-4 rounded-lg">
                        <h4 className="text-xl font-bold text-white">{path.courses}</h4>
                        <p className="text-sm text-gray-400">Courses</p>
                      </div>
                      <div className="bg-black/30 p-4 rounded-lg">
                        <h4 className="text-xl font-bold text-white">{path.level}</h4>
                        <p className="text-sm text-gray-400">Level</p>
                      </div>
                      <div className="bg-black/30 p-4 rounded-lg">
                        <h4 className="text-xl font-bold text-white">{path.certifications}</h4>
                        <p className="text-sm text-gray-400">Certifications</p>
                      </div>
                    </div>
                    
                    <Separator className="bg-gray-800" />
                    
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <h4 className="font-medium text-white">Next Up:</h4>
                        <Badge variant="outline" className="bg-pink-500/10 text-pink-400 border-pink-500/20 hover:bg-pink-500/20">
                          Stage {path.stages.findIndex(s => s.progress < 100) + 1}
                        </Badge>
                      </div>
                      
                      {(() => {
                        const nextStageIndex = path.stages.findIndex(s => s.progress < 100);
                        if (nextStageIndex === -1) return null;
                        
                        const nextStage = path.stages[nextStageIndex];
                        const nextCourse = nextStage.courses.find(c => !c.completed);
                        
                        if (!nextCourse) return null;
                        
                        return (
                          <Card className="border-gray-700 bg-black/20">
                            <CardContent className="p-4">
                              <div className="space-y-2">
                                <h4 className="font-medium text-white">{nextCourse.title}</h4>
                                <p className="text-sm text-gray-300">{nextCourse.description}</p>
                                <div className="flex items-center justify-between mt-2">
                                  <div className="flex items-center gap-2">
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
                                      className="h-4 w-4 text-gray-400"
                                    >
                                      <circle cx="12" cy="12" r="10" />
                                      <polyline points="12 6 12 12 16 14" />
                                    </svg>
                                    <span className="text-xs text-gray-400">{nextCourse.duration}</span>
                                  </div>
                                  <div className="flex items-center gap-2">
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
                                      className="h-4 w-4 text-gray-400"
                                    >
                                      <path d="M21 10V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l2-1.14" />
                                      <path d="M16.5 9.4 7.55 4.24" />
                                      <polyline points="3.29 7 12 12 20.71 7" />
                                      <line x1="12" y1="22" y2="12" x2="12" />
                                    </svg>
                                    <span className="text-xs text-gray-400">{nextCourse.lessons} lessons</span>
                                  </div>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        )
                      })()}
                    </div>
                    
                    <div className="space-y-2">
                      <Button className="w-full bg-gradient-to-r from-pink-500 to-purple-600 hover:opacity-90 text-white">
                        Continue Learning
                      </Button>
                      <Button variant="outline" className="w-full border-gray-700 text-white hover:bg-gray-800">
                        View Certificate
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="border-gray-800 bg-black/50">
                <div className="px-6 py-5 border-b border-gray-800">
                  <h3 className="text-xl font-bold text-white">Skills You'll Gain</h3>
                </div>
                <CardContent className="p-6">
                  <div className="flex flex-wrap gap-2">
                    {path.tags.map((tag, index) => (
                      <Badge key={index} variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/20 hover:bg-purple-500/20">
                        {tag}
                      </Badge>
                    ))}
                    {/* Additional relevant skills */}
                    <Badge variant="outline" className="bg-pink-500/10 text-pink-400 border-pink-500/20 hover:bg-pink-500/20">
                      {path.id === "blockchain-developer" ? "dApp Development" : "Yield Optimization"}
                    </Badge>
                    <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20 hover:bg-blue-500/20">
                      {path.id === "blockchain-developer" ? "Blockchain Architecture" : "Risk Management"}
                    </Badge>
                    <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/20 hover:bg-green-500/20">
                      {path.id === "blockchain-developer" ? "Cryptography" : "Protocol Analysis"}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="border-gray-800 bg-black/50">
                <div className="px-6 py-5 border-b border-gray-800">
                  <h3 className="text-xl font-bold text-white">Related Paths</h3>
                </div>
                <CardContent className="p-6">
                  <div className="space-y-4">
                    {Object.values(learningPaths)
                      .filter(p => p.id !== id)
                      .map((relatedPath) => (
                        <Link key={relatedPath.id} href={`/learning-paths/${relatedPath.id}`}>
                          <div className="flex items-start gap-3 p-3 rounded-lg hover:bg-gray-800/50 transition-colors">
                            <div className="h-14 w-20 rounded overflow-hidden relative flex-shrink-0">
                              <Image
                                src={relatedPath.image || "/placeholder.svg"}
                                alt={relatedPath.title}
                                fill
                                className="object-cover"
                              />
                            </div>
                            <div>
                              <h4 className="font-medium text-white">{relatedPath.title}</h4>
                              <p className="text-xs text-gray-400 mt-1">{relatedPath.courses} courses • {relatedPath.duration}</p>
                            </div>
                          </div>
                        </Link>
                      ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}