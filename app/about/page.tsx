"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import Image from "next/image"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

export default function AboutPage() {
  const [mounted, setMounted] = useState(false)
  const [animationState, setAnimationState] = useState(0)
  
  useEffect(() => {
    setMounted(true)
    
    // Cycle through animation states
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3)
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="container py-8 md:py-12 relative overflow-hidden">
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

      <div className="space-y-16 relative z-10">
        {/* Hero section */}
        <div className="flex flex-col lg:flex-row gap-8 items-center">
          <motion.div 
            className="space-y-4 lg:w-1/2"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
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
            <h1 className="text-4xl font-bold tracking-tight text-white">About Edu-Quest</h1>
            <p className="text-xl text-gray-300">
              Revolutionizing education through Web3 technology and decentralized learning experiences.
            </p>
            <p className="text-gray-300">
              Edu-Quest is a pioneering platform that merges traditional education with blockchain technology, creating
              a more transparent, accessible, and verifiable learning ecosystem. We're dedicated to empowering students
              and professionals with the knowledge and skills needed to thrive in the Web3 economy.
            </p>
            <div className="flex flex-col sm:flex-row gap-3">
              <Button size="lg" className="bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700" asChild>
                <Link href="/courses">Explore Courses</Link>
              </Button>
              <Button size="lg" className="bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700" asChild>
                <Link href="/connect">Connect Wallet</Link>
              </Button>
            </div>
          </motion.div>
          <motion.div 
            className="lg:w-1/2 relative"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.3 }}
          >
            <div className="aspect-video relative rounded-xl overflow-hidden shadow-xl">
              <Image
                src="https://res.cloudinary.com/ddavgtvp2/image/upload/v1741504745/qr4ncmrkkn8x3sgr0hws.jpg?height=400&width=600"
                alt="Edu-Quest Platform"
                fill
                className="object-cover"
              />
            </div>
            <div className="absolute -bottom-6 -right-6 -z-10 aspect-square w-48 rounded-full bg-pink-500/30 blur-3xl" />
          </motion.div>
        </div>

        {/* Mission & Vision */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
          >
            <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
              <CardContent className="p-6 space-y-4">
                <div className="inline-flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-pink-500/20 to-purple-600/20">
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
                    className="h-6 w-6 text-pink-400"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" x2="12" y1="8" y2="16" />
                    <line x1="8" x2="16" y1="12" y2="12" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-white">Our Mission</h2>
                <p className="text-gray-300">
                  To revolutionize education by leveraging blockchain technology to create a more transparent, accessible,
                  and verifiable learning ecosystem. We aim to bridge the gap between traditional education and the
                  evolving Web3 landscape, empowering individuals with the knowledge and skills needed to thrive in the
                  decentralized future.
                </p>
              </CardContent>
            </Card>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
          >
            <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
              <CardContent className="p-6 space-y-4">
                <div className="inline-flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-pink-500/20 to-purple-600/20">
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
                    className="h-6 w-6 text-purple-400"
                  >
                    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-white">Our Vision</h2>
                <p className="text-gray-300">
                  A world where education is not constrained by traditional boundaries—where learning is personalized,
                  credentials are verifiable, and opportunities are accessible to all. We envision a future where
                  blockchain technology enhances education, creating a more inclusive, innovative, and interconnected
                  learning ecosystem that prepares individuals for the next generation of the internet.
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Our Approach */}
        <div className="space-y-8">
          <motion.div 
            className="text-center space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <h2 className="text-3xl font-bold text-white">Our Approach</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              We combine innovative technology with proven educational methods to deliver a unique learning experience
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
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
                    className="h-6 w-6 text-pink-400"
                  >
                    <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                  </svg>
                ),
                title: "Personalized Learning",
                description:
                  "Our AI-powered system analyzes your strengths, weaknesses, and goals to create customized learning paths that adapt as you progress, ensuring an efficient and effective educational journey.",
              },
              {
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
                    className="h-6 w-6 text-purple-400"
                  >
                    <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
                    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                  </svg>
                ),
                title: "Blockchain Verification",
                description:
                  "All credentials earned on Edu-Quest are securely stored on the blockchain, providing tamper-proof verification that can be instantly checked by employers and educational institutions worldwide.",
              },
              {
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
                    className="h-6 w-6 text-pink-400"
                  >
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                    <circle cx="9" cy="7" r="4" />
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                  </svg>
                ),
                title: "Community Learning",
                description:
                  "We foster a vibrant community of learners, educators, and industry experts who collaborate, share knowledge, and support each other throughout their educational journey and career development.",
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.1 * index }}
              >
                <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
                  <CardContent className="p-6 space-y-4">
                    <div className="inline-flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-pink-500/20 to-purple-600/20">
                      {item.icon}
                    </div>
                    <h3 className="text-xl font-bold text-white">{item.title}</h3>
                    <p className="text-gray-300">{item.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Team Section */}
        <div className="space-y-8">
          <motion.div 
            className="text-center space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <h2 className="text-3xl font-bold text-white">Our Team</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Meet the passionate individuals behind Edu-Quest who are dedicated to transforming education through
              blockchain technology
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                name: "Dr. Sarah Chen",
                role: "Founder & CEO",
                bio: "Former professor of Computer Science with a passion for educational innovation and blockchain technology.",
                image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741501068/yo4xdx05qa6vwaghljr1.jpg?height=300&width=300",
              },
              {
                name: "Michael Rodriguez",
                role: "CTO",
                bio: "Blockchain developer and educator with 10+ years of experience in decentralized systems.",
                image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741500745/n71smveywoij44pvjf8g.jpg?height=300&width=300",
              },
              {
                name: "Aisha Patel",
                role: "Head of Curriculum",
                bio: "Educational specialist focused on creating engaging, practical learning experiences for Web3.",
                image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741501188/m7dlqm8rqtjes5bgrewc.jpg?height=300&width=300",
              },
              {
                name: "James Wilson",
                role: "Web3 Partnerships",
                bio: "Industry connector with deep relationships across the blockchain and education sectors.",
                image: "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741501359/yaum802eidvw5tslyb7j.jpg?height=300&width=300",
              },
            ].map((member, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.1 * index }}
              >
                <Card className="overflow-hidden bg-black/40 border-white/10 backdrop-blur-sm">
                  <div className="aspect-square relative">
                    <Image src={member.image || "/placeholder.svg"} alt={member.name} fill className="object-cover" />
                  </div>
                  <CardContent className="p-4 space-y-2">
                    <h3 className="font-bold text-lg text-white">{member.name}</h3>
                    <p className="text-sm text-pink-400">{member.role}</p>
                    <p className="text-sm text-gray-300">{member.bio}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Contact CTA */}
        <motion.div 
          className="relative rounded-xl p-8 md:p-12 text-center space-y-6 overflow-hidden"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          {/* Background gradient for CTA */}
          <div className="absolute inset-0 bg-gradient-to-br from-pink-600/20 to-purple-600/20 backdrop-blur-sm -z-10" />
          
          <h2 className="text-3xl font-bold text-white">Get in Touch</h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Have questions about Edu-Quest? Interested in partnering with us? We'd love to hear from you!
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700" asChild>
              <Link href="/contact">Contact Us</Link>
            </Button>
            <Button size="lg"  className="text-white border-white/20 hover:bg-white/10" asChild>
              <Link href="/partnerships">Partnership Opportunities</Link>
            </Button>
          </div>
        </motion.div>
      </div>
    </div>
  )
}