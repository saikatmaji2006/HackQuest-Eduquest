"use client"

import { useState, useEffect } from "react"
import Image from "next/image"
import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"

export function TestimonialSection() {
  const [mounted, setMounted] = useState(false)
  const [animationState, setAnimationState] = useState(0)
  
  // Set mounted state to true after component mounts to avoid hydration errors
  useEffect(() => {
    setMounted(true)
    
    // Cycle through animation states
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3)
    }, 5000)
    
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
          {mounted && [...Array(20)].map((_, i) => (
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
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <motion.div 
            className="space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="flex justify-center items-center mb-2">
              <motion.div 
                className="h-1 w-12 bg-gradient-to-r from-pink-500 to-purple-600 mr-4"
                initial={{ width: 0 }}
                animate={{ width: 48 }}
                transition={{ duration: 0.8, delay: 0.2 }}
              ></motion.div>
              <motion.div 
                className="inline-block rounded-lg bg-gradient-to-r from-pink-500 to-purple-600 px-3 py-1 text-sm text-white font-bold"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.6 }}
              >
                Testimonials
              </motion.div>
              <motion.div 
                className="h-1 w-12 bg-gradient-to-r from-purple-600 to-pink-500 ml-4"
                initial={{ width: 0 }}
                animate={{ width: 48 }}
                transition={{ duration: 0.8, delay: 0.2 }}
              ></motion.div>
            </div>
            <motion.h2 
              className="text-3xl font-bold tracking-tighter md:text-5xl/tight bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-300"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.3 }}
            >
              Success Stories from Our <span className="bg-clip-text text-transparent bg-gradient-to-r from-pink-400 to-purple-600">Community</span>
            </motion.h2>
            <motion.p 
              className="mx-auto max-w-[700px] text-gray-300 md:text-xl"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.5 }}
            >
              Hear from students and professionals who have transformed their careers with Edu-Quest.
            </motion.p>
          </motion.div>
        </div>
        
        <motion.div 
          className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mt-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.7 }}
        >
          {[
            {
              name: "Sarah Johnson",
              role: "Software Engineer",
              quote: "The blockchain credentials I earned through Edu-Quest helped me stand out in a competitive job market. Employers were impressed by the verifiable proof of my skills.",
              color: "pink"
            },
            {
              name: "Michael Chen",
              role: "Data Scientist",
              quote: "The personalized learning path was a game-changer. The AI-powered recommendations helped me focus on exactly what I needed to learn to transition into data science.",
              color: "purple"
            },
            {
              name: "Aisha Patel",
              role: "Blockchain Developer",
              quote: "The virtual simulations gave me hands-on experience with smart contract development before I ever had to deploy to a real blockchain. This practical approach was invaluable.",
              color: "pink"
            }
          ].map((testimonial, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.8 + (i * 0.2) }}
              whileHover={{ 
                y: -5,
                boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
                transition: { duration: 0.2 }
              }}
            >
              <Card className="bg-black/60 border border-gray-800 backdrop-blur-sm overflow-hidden">
                <CardContent className="p-6 relative">
                  {/* Decorative corner accent */}
                  <div className={`absolute top-0 right-0 w-20 h-20 -mr-10 -mt-10 rounded-full bg-gradient-to-br from-${testimonial.color}-500/20 to-transparent`} />
                  
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center gap-4">
                      <motion.div 
                        className={`h-12 w-12 rounded-full bg-${testimonial.color}-500/20 overflow-hidden border border-${testimonial.color}-500/30 flex items-center justify-center`}
                        whileHover={{ scale: 1.1 }}
                        transition={{ duration: 0.2 }}
                      >
                        <Image
                          src="/placeholder.svg?height=48&width=48"
                          alt="Avatar"
                          width={48}
                          height={48}
                          className="h-full w-full object-cover"
                        />
                      </motion.div>
                      <div>
                        <h3 className="font-bold text-white">{testimonial.name}</h3>
                        <p className="text-sm text-gray-400">{testimonial.role}</p>
                      </div>
                    </div>
                    
                    <motion.div 
                      className="flex gap-1"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 1 + (i * 0.2), duration: 0.5 }}
                    >
                      {[...Array(5)].map((_, i) => (
                        <motion.svg
                          key={i}
                          xmlns="http://www.w3.org/2000/svg"
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="currentColor"
                          className="h-4 w-4 text-yellow-400"
                          initial={{ opacity: 0, scale: 0 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 1.2 + (i * 0.1), duration: 0.3 }}
                        >
                          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                        </motion.svg>
                      ))}
                    </motion.div>
                    
                    <div className="relative">
                      <motion.svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                        className={`absolute -top-2 -left-1 h-6 w-6 text-${testimonial.color}-500/20 transform -scale-x-100`}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1.1 + (i * 0.2) }}
                      >
                        <path d="M14.017 21v-7.391c0-5.704 3.731-9.57 8.983-10.609l.995 2.151c-2.432.917-3.995 3.638-3.995 5.849h4v10h-9.983zm-14.017 0v-7.391c0-5.704 3.748-9.57 9-10.609l.996 2.151c-2.433.917-3.996 3.638-3.996 5.849h3.983v10h-9.983z" />
                      </motion.svg>
                      
                      <motion.p 
                        className="text-sm text-gray-300 pl-5"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1.2 + (i * 0.2), duration: 0.5 }}
                      >
                        {testimonial.quote}
                      </motion.p>
                    </div>
                    
                    {/* Decorative glow effect at bottom */}
                    <div className={`absolute bottom-0 left-1/2 -translate-x-1/2 w-3/4 h-1 bg-gradient-to-r from-transparent via-${testimonial.color}-500/40 to-transparent rounded-full`} />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
        
        {/* View more testimonials button */}
        <motion.div 
          className="flex justify-center mt-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1.6 }}
        >
          <motion.button
            className="px-6 py-3 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 text-white font-medium flex items-center gap-2"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
          >
            <span>View More Success Stories</span>
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
              <path d="M5 12h14"></path>
              <path d="m12 5 7 7-7 7"></path>
            </svg>
          </motion.button>
        </motion.div>
      </div>
    </section>
  )
}