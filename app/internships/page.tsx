"use client"

import Link from "next/link"
import Image from "next/image"
import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const internships = [
  {
    id: "1",
    title: "Blockchain Developer Intern",
    company: "TechFusion Labs",
    location: "Remote",
    type: "Full-time",
    duration: "3 months",
    stipend: "Paid",
    description:
      "Work on developing and testing smart contracts for decentralized finance applications. Gain hands-on experience with Solidity, Truffle, and Web3.js.",
    requirements: [
      "Basic knowledge of blockchain technology",
      "Experience with JavaScript and web development",
      "Understanding of Solidity (preferred but not required)",
      "Strong problem-solving skills",
    ],
    skills: ["Solidity", "Smart Contracts", "Web3.js", "DeFi"],
    logo: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741503351/hmnxaswjd56knmjbokgg.jpg?height=80&width=80",
    category: "development",
  },
  {
    id: "2",
    title: "Web3 Frontend Developer",
    company: "MetaVerse Innovations",
    location: "Hybrid (New York)",
    type: "Part-time",
    duration: "6 months",
    stipend: "Paid",
    description:
      "Design and implement user interfaces for decentralized applications. Work with React, Next.js, and Ethers.js to create seamless Web3 experiences.",
    requirements: [
      "Solid understanding of React and modern JavaScript",
      "Experience with responsive design and CSS frameworks",
      "Familiarity with Web3 concepts and wallet integration",
      "Portfolio of frontend projects",
    ],
    skills: ["React", "Next.js", "Ethers.js", "UI/UX"],
    logo: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741503449/p3d6wd2oowevsduhwi0j.jpg?height=80&width=80",
    category: "frontend",
  },
  {
    id: "3",
    title: "NFT Project Assistant",
    company: "ArtChain Collective",
    location: "Remote",
    type: "Part-time",
    duration: "4 months",
    stipend: "Paid",
    description:
      "Assist in the creation, minting, and marketing of NFT collections. Learn about NFT standards, metadata, and marketplace integration.",
    requirements: [
      "Interest in digital art and NFTs",
      "Basic understanding of blockchain technology",
      "Marketing or design background is a plus",
      "Creativity and attention to detail",
    ],
    skills: ["NFT", "ERC-721", "Digital Art", "Marketing"],
    logo: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741503529/sucge04s9p30lhepnwsb.jpg?height=80&width=80",
    category: "creative",
  },
  {
    id: "4",
    title: "Smart Contract Auditor Assistant",
    company: "SecureChain Solutions",
    location: "Remote",
    type: "Full-time",
    duration: "3 months",
    stipend: "Paid",
    description:
      "Learn to identify vulnerabilities in smart contracts and assist in the auditing process. Work with experienced security engineers on real projects.",
    requirements: [
      "Strong knowledge of Solidity",
      "Understanding of common smart contract vulnerabilities",
      "Background in computer science or cybersecurity",
      "Analytical thinking and attention to detail",
    ],
    skills: ["Security", "Auditing", "Solidity", "Vulnerabilities"],
    logo: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741503581/jhvapymgbkv2h0qpuea8.jpg?height=80&width=80",
    category: "security",
  },
  {
    id: "5",
    title: "DeFi Research Intern",
    company: "CryptoFinance Research",
    location: "Remote",
    type: "Part-time",
    duration: "5 months",
    stipend: "Paid",
    description:
      "Research and analyze DeFi protocols, tokenomics, and market trends. Create reports and contribute to the company's research publications.",
    requirements: [
      "Understanding of DeFi concepts and protocols",
      "Strong analytical and research skills",
      "Background in finance, economics, or computer science",
      "Excellent writing and communication skills",
    ],
    skills: ["DeFi", "Research", "Tokenomics", "Analysis"],
    logo: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741504033/afrjdvcwqzyejqaxmwyn.jpg?height=80&width=80",
    category: "research",
  },
  {
    id: "6",
    title: "Blockchain Community Manager",
    company: "DecentralizedWorld",
    location: "Remote",
    type: "Full-time",
    duration: "4 months",
    stipend: "Paid",
    description:
      "Engage with the blockchain community, organize events, and create content to grow the company's presence in the Web3 space.",
    requirements: [
      "Passion for blockchain and Web3 technologies",
      "Excellent communication and interpersonal skills",
      "Experience with social media management",
      "Creative content creation abilities",
    ],
    skills: ["Community Management", "Content Creation", "Social Media", "Events"],
    logo: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741503384/zjrtjcjp5nnli2gtqt5o.jpg?height=80&width=80",
    category: "community",
  },
]

export default function InternshipsPage() {
  // State for the animation and mounting
  const [animationState, setAnimationState] = useState(0);
  const [mounted, setMounted] = useState(false);

  // Set mounted state after component mounts
  useEffect(() => {
    setMounted(true);
    
    // Animation cycle
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev === 2 ? 0 : prev + 1));
    }, 4000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Animated background - only rendered client-side */}
      {mounted && (
        <div className="fixed inset-0 -z-10">
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
      
      {/* Content - now with modified styling for better readability on dark background */}
      <div className="container py-8 md:py-12 relative z-10">
        <div className="flex flex-col gap-8">
          <motion.div 
            className="space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-3xl font-bold tracking-tight text-white">Internship Opportunities</h1>
            <p className="text-gray-300 max-w-[800px]">
              Apply for real-world internships with Web3 companies and projects. Gain practical experience, build your
              portfolio, and make connections in the blockchain industry.
            </p>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex items-center gap-4 flex-col sm:flex-row"
          >
            <div className="relative w-full sm:w-80">
              <Input placeholder="Search internships..." className="pl-8 bg-gray-900/60 border-gray-700 text-white" />
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
                <SelectTrigger className="w-full sm:w-[150px] bg-gray-900/60 border-gray-700 text-white">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="full-time">Full-time</SelectItem>
                  <SelectItem value="part-time">Part-time</SelectItem>
                </SelectContent>
              </Select>
              <Select defaultValue="all">
                <SelectTrigger className="w-full sm:w-[150px] bg-gray-900/60 border-gray-700 text-white">
                  <SelectValue placeholder="Location" />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  <SelectItem value="all">All Locations</SelectItem>
                  <SelectItem value="remote">Remote</SelectItem>
                  <SelectItem value="hybrid">Hybrid</SelectItem>
                  <SelectItem value="on-site">On-site</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Tabs defaultValue="all" className="w-full">
              <TabsList className="grid w-full grid-cols-4 lg:grid-cols-7 mb-8 bg-gray-800/60">
                <TabsTrigger value="all">All</TabsTrigger>
                <TabsTrigger value="development">Development</TabsTrigger>
                <TabsTrigger value="frontend">Frontend</TabsTrigger>
                <TabsTrigger value="security">Security</TabsTrigger>
                <TabsTrigger value="research">Research</TabsTrigger>
                <TabsTrigger value="creative">Creative</TabsTrigger>
                <TabsTrigger value="community">Community</TabsTrigger>
              </TabsList>

              <TabsContent value="all">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {internships.map((internship, index) => (
                    <motion.div
                      key={internship.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 0.1 * index }}
                    >
                      <InternshipCard internship={internship} />
                    </motion.div>
                  ))}
                </div>
              </TabsContent>

              {["development", "frontend", "security", "research", "creative", "community"].map((category) => (
                <TabsContent key={category} value={category}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {internships
                      .filter((internship) => internship.category === category)
                      .map((internship, index) => (
                        <motion.div
                          key={internship.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.5, delay: 0.1 * index }}
                        >
                          <InternshipCard internship={internship} />
                        </motion.div>
                      ))}
                  </div>
                </TabsContent>
              ))}
            </Tabs>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

interface InternshipCardProps {
  internship: {
    id: string
    title: string
    company: string
    location: string
    type: string
    duration: string
    stipend: string
    description: string
    requirements: string[]
    skills: string[]
    logo: string
    category: string
  }
}

function InternshipCard({ internship }: InternshipCardProps) {
  return (
    <Card className="bg-gray-900/80 border-gray-800 backdrop-blur-sm">
      <CardHeader className="pb-2">
        <div className="flex items-start gap-4">
          <div className="h-16 w-16 rounded-md overflow-hidden bg-gray-800 flex items-center justify-center relative">
            <Image src={internship.logo || "/placeholder.svg"} alt={internship.company} fill className="object-cover" />
          </div>
          <div className="space-y-1">
            <CardTitle className="text-xl text-white">{internship.title}</CardTitle>
            <CardDescription className="text-gray-400">{internship.company}</CardDescription>
            <div className="flex flex-wrap gap-2 pt-1">
              <Badge variant="outline" className="text-gray-300 border-gray-700">{internship.location}</Badge>
              <Badge variant="outline" className="text-gray-300 border-gray-700">{internship.type}</Badge>
              <Badge variant="outline" className="text-gray-300 border-gray-700">{internship.duration}</Badge>
              <Badge variant="outline" className="text-gray-300 border-gray-700">{internship.stipend}</Badge>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pb-2">
        <div className="space-y-4">
          <p className="text-sm text-gray-400">{internship.description}</p>

          <div>
            <h4 className="text-sm font-medium mb-2 text-white">Requirements:</h4>
            <ul className="text-sm text-gray-400 space-y-1">
              {internship.requirements.map((requirement, index) => (
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
                    className="h-4 w-4 text-pink-500 mt-0.5"
                  >
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>{requirement}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="flex flex-wrap gap-1">
            {internship.skills.map((skill) => (
              <Badge key={skill} variant="secondary" className="text-xs bg-purple-900/60 text-purple-200">
                {skill}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" className="border-gray-700 text-gray-300 hover:bg-gray-800 hover:text-white">Learn More</Button>
        <Button className="bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700" asChild>
          <Link href={`/internships/${internship.id}/apply`}>Apply Now</Link>
        </Button>
      </CardFooter>
    </Card>
  )
}