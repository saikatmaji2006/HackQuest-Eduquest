"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface Feature {
  title: string;
  description: string;
  content: string;
  icon: string;
}

const features: Feature[] = [
  {
    title: "Personalized Learning Paths",
    description: "AI-powered roadmaps tailored to your goals and learning style",
    content: "Our advanced algorithms analyze your strengths, weaknesses, and career goals to create a customized learning journey that adapts as you progress.",
    icon: "📚",
  },
  {
    title: "Blockchain Credentials",
    description: "Secure, verifiable certificates stored on the blockchain",
    content: "Earn tamper-proof credentials that can be instantly verified by employers, eliminating credential fraud and streamlining the hiring process.",
    icon: "🔗",
  },
  {
    title: "Virtual Simulations",
    description: "Immersive, hands-on learning experiences",
    content: "Practice real-world scenarios in our virtual environments, allowing you to apply theoretical knowledge and develop practical skills safely.",
    icon: "🕶️",
  },
  {
    title: "Internship Marketplace",
    description: "Connect with companies offering real-world opportunities",
    content: "Browse and apply for internships directly through our platform, with your blockchain-verified skills giving you a competitive edge.",
    icon: "💼",
  },
  {
    title: "Skill Verification",
    description: "Transparent and trustworthy skill assessment",
    content: "Our blockchain-based verification system ensures that your skills and achievements are accurately represented and easily shareable.",
    icon: "✅",
  },
  {
    title: "Global Learning Community",
    description: "Connect with learners and mentors worldwide",
    content: "Join our decentralized community of learners, educators, and industry experts to collaborate, share knowledge, and grow together.",
    icon: "🌍",
  },
];

export function FeatureSection() {
  const [mounted, setMounted] = useState(false);
  const [animationState, setAnimationState] = useState(0);
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);
  const sectionRef = useRef<HTMLElement>(null);
  
  useEffect(() => {
    setMounted(true);
    
    // Cycle through animation states
    const interval = setInterval(() => {
      setAnimationState((prev) => (prev + 1) % 3);
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  // Generate random positions for particles
  const getRandomParticles = () => {
    return Array(12).fill(0).map((_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      scale: Math.random() * 2 + 0.5,
      duration: Math.random() * 10 + 15,
      opacity: Math.random() * 0.5 + 0.3
    }));
  };

  const particles = useRef(getRandomParticles());

  return (
    <section ref={sectionRef} className="w-full py-12 md:py-24 lg:py-32 xl:py-40 relative overflow-hidden">
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
          
          {/* Optimized floating particles using percentages for positioning */}
          {particles.current.map((particle) => (
            <motion.div
              key={particle.id}
              className="absolute w-1 h-1 rounded-full bg-pink-500"
              initial={{
                left: `${particle.x}%`,
                top: `${particle.y}%`,
                opacity: particle.opacity,
                scale: particle.scale
              }}
              animate={{
                y: [0, `-${20 + Math.random() * 30}%`],
                opacity: [particle.opacity, particle.opacity * 0.3],
              }}
              transition={{
                duration: particle.duration,
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
        <div className="flex flex-col items-center justify-center space-y-6 text-center mb-12">
          <motion.div
            className="space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            {/* Enhanced title with animated underline */}
            <div className="inline-block">
              <motion.div className="flex items-center justify-center mb-4">
                <motion.div 
                  className="h-1 w-12 bg-gradient-to-r from-pink-500 to-purple-600 mr-4"
                  initial={{ width: 0 }}
                  animate={{ width: 48 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                />
                <motion.span 
                  className="text-pink-400 text-sm font-bold tracking-wider uppercase"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.6 }}
                >
                  Revolutionary Features
                </motion.span>
                <motion.div 
                  className="h-1 w-12 bg-gradient-to-r from-purple-600 to-pink-500 ml-4"
                  initial={{ width: 0 }}
                  animate={{ width: 48 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                />
              </motion.div>
              <h2 className="text-4xl font-bold tracking-tighter md:text-5xl bg-clip-text text-transparent bg-gradient-to-r from-pink-400 via-purple-400 to-indigo-400 relative">
                Transforming Education with Web3
                <motion.span 
                  className="absolute -bottom-2 left-0 right-0 mx-auto h-1 w-48 bg-gradient-to-r from-pink-500 to-purple-600"
                  initial={{ width: 0 }}
                  animate={{ width: 192 }}
                  transition={{ duration: 1, delay: 1 }}
                />
              </h2>
            </div>
            <motion.p 
              className="mx-auto max-w-[700px] text-lg text-gray-300 md:text-xl"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.7, delay: 0.3 }}
            >
              Discover how Edu-Quest is revolutionizing learning through blockchain technology, AI-driven insights, and immersive experiences.
            </motion.p>
          </motion.div>
        </div>

        <motion.div
          className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mt-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              whileHover={{ y: -8, transition: { duration: 0.2 } }}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 + index * 0.1 }}
              onMouseEnter={() => setHoveredCard(index)}
              onMouseLeave={() => setHoveredCard(null)}
            >
              <Card className="group h-full p-4 rounded-xl bg-black/40 backdrop-blur-lg border border-white/10 hover:border-pink-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-pink-500/20 overflow-hidden relative">
                {/* Glow effect on hover */}
                <AnimatePresence>
                  {hoveredCard === index && (
                    <motion.div 
                      className="absolute -inset-0.5 rounded-xl bg-gradient-to-r from-pink-500 to-purple-600 opacity-0 blur-md z-0"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 0.5 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    />
                  )}
                </AnimatePresence>
                
                <CardHeader className="pb-2 relative z-10">
                  <CardTitle className="flex items-center gap-3 text-white">
                    <motion.span 
                      className="p-3 rounded-lg bg-black/60 border border-white/10 group-hover:bg-gradient-to-r group-hover:from-pink-500 group-hover:to-purple-600 transition-all duration-300"
                      whileHover={{ scale: 1.1 }}
                    >
                      {feature.icon}
                    </motion.span>
                    {feature.title}
                  </CardTitle>
                  <CardDescription className="text-gray-400 group-hover:text-gray-300 transition-colors duration-300">
                    {feature.description}
                  </CardDescription>
                </CardHeader>
                <CardContent className="relative z-10">
                  <p className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors duration-300">
                    {feature.content}
                  </p>
                  
                  {/* Animated accent line */}
                  <motion.div 
                    className="h-0.5 w-0 bg-gradient-to-r from-pink-500 to-purple-600 mt-4 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: hoveredCard === index ? "100%" : 0 }}
                    transition={{ duration: 0.5 }}
                  />
                  
                  {/* New: Learn more link that appears on hover */}
                  <AnimatePresence>
                    {hoveredCard === index && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        transition={{ duration: 0.2 }}
                        className="mt-4"
                      >
                        <a href="#" className="text-pink-400 hover:text-pink-300 text-sm font-medium flex items-center">
                          Learn more
                          <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </a>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* Badge counter */}
        <motion.div
          className="flex justify-center mt-12 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
        >
          <div className="flex items-center gap-8 bg-black/30 backdrop-blur-md px-8 py-4 rounded-xl border border-white/10">
            <div className="flex flex-col items-center">
              <span className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-400">5000+</span>
              <span className="text-gray-400 text-sm">Active Students</span>
            </div>
            <div className="h-12 w-px bg-gray-700" />
            <div className="flex flex-col items-center">
              <span className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-400">150+</span>
              <span className="text-gray-400 text-sm">Courses</span>
            </div>
            <div className="h-12 w-px bg-gray-700" />
            <div className="flex flex-col items-center">
              <span className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-400">98%</span>
              <span className="text-gray-400 text-sm">Satisfaction</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          className="flex justify-center mt-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1 }}
        >
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
          >
            <Button 
              size="lg" 
              className="px-8 py-6 bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white border-none shadow-lg shadow-pink-500/20 text-lg font-bold"
            >
              Explore All Features
            </Button>
          </motion.div>
        </motion.div>
        
        {/* Added decorative elements for visual interest */}
        {mounted && (
          <>
            <motion.div
              className="absolute -bottom-16 -left-16 w-32 h-32 rounded-full border border-pink-500/20 opacity-50"
              animate={{
                scale: [1, 1.2, 1],
                rotate: [0, 360],
              }}
              transition={{
                duration: 20,
                repeat: Infinity,
                ease: "linear"
              }}
            />
            <motion.div
              className="absolute top-24 -right-8 w-24 h-24 rounded-full border border-purple-500/20 opacity-30"
              animate={{
                scale: [1.2, 1, 1.2],
                rotate: [360, 0],
              }}
              transition={{
                duration: 15,
                repeat: Infinity,
                ease: "linear"
              }}
            />
            <motion.div
              className="absolute top-1/2 left-10 w-2 h-10 rounded-full bg-gradient-to-b from-pink-500/20 to-purple-500/5 opacity-70"
              animate={{
                height: [40, 60, 40],
                opacity: [0.7, 0.3, 0.7],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                repeatType: "reverse",
              }}
            />
            <motion.div
              className="absolute bottom-20 right-12 w-2 h-14 rounded-full bg-gradient-to-b from-purple-500/20 to-pink-500/5 opacity-70"
              animate={{
                height: [56, 80, 56],
                opacity: [0.7, 0.3, 0.7],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                repeatType: "reverse",
                delay: 1.5
              }}
            />
          </>
        )}
      </div>
    </section>
  );
}

export default FeatureSection;