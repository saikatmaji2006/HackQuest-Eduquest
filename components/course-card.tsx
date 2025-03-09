"use client"

import { useState, useEffect } from "react"
import Image from "next/image"
import Link from "next/link"
import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

interface CourseCardProps {
  course: {
    id: string
    title: string
    description: string
    instructor: string
    level: string
    duration: string
    rating: number
    students: number
    image: string
    tags: string[]
    category: string
  }
}

export function CourseCard({ course }: CourseCardProps) {
  const [mounted, setMounted] = useState(false)
  const [animationState, setAnimationState] = useState(0)

  useEffect(() => {
    setMounted(true)
    
    // Animation state rotation for the background effects
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3)
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])

  // Determine badge color based on level
  const getBadgeVariant = (level: string) => {
    switch(level) {
      case "Beginner": return "default";
      case "Intermediate": return "secondary";
      case "Advanced": return "outline";
      default: return "default";
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      whileHover={{ y: -5 }}
      className="h-full relative"
    >
      {/* Animated background - only rendered client-side to avoid hydration errors */}
      {mounted && (
        <div className="absolute inset-0 -z-10 rounded-xl overflow-hidden">
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
          {[...Array(8)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 rounded-full bg-pink-500"
              initial={{
                x: Math.random() * 400,
                y: Math.random() * 400,
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
            className="absolute top-1/4 left-1/4 w-64 h-64 bg-gradient-to-r from-pink-600/10 to-purple-600/10 rounded-full blur-3xl"
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
            className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-gradient-to-r from-purple-600/10 to-pink-600/10 rounded-full blur-3xl"
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

      <Card className="overflow-hidden flex flex-col h-full relative border border-purple-500/20 shadow-lg bg-black/40 backdrop-blur-sm">
        {/* Animated border glow effect */}
        {mounted && (
          <motion.div 
            className="absolute -inset-0.5 bg-gradient-to-r from-pink-500/40 to-purple-600/40 rounded-xl blur-sm opacity-0"
            animate={{ opacity: [0, 0.6, 0] }}
            transition={{ duration: 3, repeat: Infinity, repeatType: "reverse" }}
          />
        )}
        
        <div className="relative aspect-video overflow-hidden">
          <motion.div
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.3 }}
          >
            <Image
              src={course.image || "https://res.cloudinary.com/ddavgtvp2/image/upload/v1741501854/bvvxicux8thkn4f0xmio.jpg"}
              alt={course.title}
              className="object-cover"
              fill
            />
            
            {/* Overlay gradient */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
            
            {/* Floating tech particles */}
            {mounted && [...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-1 h-1 rounded-full bg-pink-400/80"
                initial={{
                  x: Math.random() * 100,
                  y: Math.random() * 100 + 100,
                  opacity: Math.random() * 0.5 + 0.3,
                  scale: Math.random() + 0.5
                }}
                animate={{
                  y: [null, `-${Math.random() * 50 + 20}px`],
                  opacity: [null, 0],
                }}
                transition={{
                  duration: Math.random() * 5 + 5,
                  repeat: Infinity,
                  ease: "linear"
                }}
              />
            ))}
          </motion.div>
          
          {/* Category tag positioned on image */}
          <motion.div 
            className="absolute top-3 left-3"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Badge 
              className="bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white border-none"
            >
              {course.category}
            </Badge>
          </motion.div>
          
          {/* Level badge positioned on image */}
          <motion.div 
            className="absolute top-3 right-3"
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Badge variant={getBadgeVariant(course.level)}>
              {course.level}
            </Badge>
          </motion.div>
        </div>
        
        <CardHeader className="pb-3 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <CardTitle className="line-clamp-1 text-xl">
              {course.title}
            </CardTitle>
            <CardDescription className="line-clamp-2 mt-1">
              {course.description}
            </CardDescription>
          </motion.div>
        </CardHeader>
        
        <CardContent className="pb-4 flex-grow">
          <motion.div 
            className="flex flex-col gap-3"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <motion.div 
              className="flex items-center gap-2 text-sm"
              whileHover={{ x: 3 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
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
                className="h-4 w-4 text-pink-500"
                whileHover={{ scale: 1.2 }}
              >
                <circle cx="12" cy="8" r="5" />
                <path d="M20 21a8 8 0 0 0-16 0" />
              </motion.svg>
              <span className="text-muted-foreground">Instructor: {course.instructor}</span>
            </motion.div>
            
            <motion.div 
              className="flex items-center gap-2 text-sm"
              whileHover={{ x: 3 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
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
                className="h-4 w-4 text-purple-500"
                whileHover={{ scale: 1.2 }}
              >
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </motion.svg>
              <span className="text-muted-foreground">Duration: {course.duration}</span>
            </motion.div>
            
            <motion.div 
              className="flex items-center gap-2 text-sm"
              whileHover={{ x: 3 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
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
                className="h-4 w-4 text-yellow-500"
                whileHover={{ scale: 1.2, rotate: 15 }}
                transition={{ type: "spring", stiffness: 500 }}
              >
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
              </motion.svg>
              <span className="font-medium">{course.rating}</span>
              <span className="text-muted-foreground">({course.students.toLocaleString()} students)</span>
            </motion.div>
            
            <motion.div 
              className="flex flex-wrap gap-1 mt-2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
              {course.tags.map((tag, index) => (
                <motion.div
                  key={tag}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.7 + (index * 0.1) }}
                  whileHover={{ y: -2, scale: 1.05 }}
                >
                  <Badge variant="secondary" className="text-xs bg-opacity-80">
                    {tag}
                  </Badge>
                </motion.div>
              ))}
            </motion.div>

            {/* Progress bar (simulated) */}
            <motion.div
              className="mt-3 h-1.5 w-full bg-gray-200 rounded-full overflow-hidden"
              initial={{ width: 0 }}
              animate={{ width: "100%" }}
              transition={{ delay: 0.8, duration: 0.6 }}
            >
              <motion.div
                className="h-full bg-gradient-to-r from-pink-500 to-purple-600 rounded-full"
                initial={{ width: "0%" }}
                animate={{ width: "30%" }}
                transition={{ delay: 1.4, duration: 0.8 }}
              />
            </motion.div>
            <motion.div 
              className="text-xs text-right text-muted-foreground mt-1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.6 }}
            >
              30% complete
            </motion.div>
          </motion.div>
        </CardContent>
        
        <CardFooter className="pt-0">
          <motion.div
            className="w-full"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Button 
              asChild 
              className="w-full bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 relative overflow-hidden group"
            >
              <Link href={`/courses/${course.id}`}>
                <motion.span 
                  className="absolute inset-0 bg-white/10 rounded-full"
                  initial={{ scale: 0, x: "-50%", y: "-50%" }}
                  whileHover={{ scale: 2.5 }}
                  transition={{ duration: 0.5 }}
                />
                <span className="relative z-10 flex items-center justify-center">
                  Continue Learning
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
                    className="ml-2 h-4 w-4"
                    initial={{ x: 0 }}
                    whileHover={{ x: 3 }}
                    transition={{ repeat: Infinity, duration: 0.6, repeatType: "reverse" }}
                  >
                    <path d="M5 12h14" />
                    <path d="m12 5 7 7-7 7" />
                  </motion.svg>
                </span>
              </Link>
            </Button>
          </motion.div>
        </CardFooter>
      </Card>
    </motion.div>
  )
}