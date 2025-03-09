import type React from "react"
import "@/app/globals.css"
import { Inter } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import { Web3Provider } from "@/lib/web3-provider"
import { Navbar } from "@/components/navbar"

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "Edu-Quest | Web3 Learning Platform",
  description:
    "A revolutionary, gamified learning platform that empowers students and professionals with personalized educational roadmaps, virtual simulations, and real-world internship opportunities.",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <Web3Provider>
            <div className="flex flex-col min-h-screen">
              <Navbar />
              <main className="flex-1">{children}</main>
              <footer className="border-t py-6 md:py-10">
                <div className="container flex flex-col md:flex-row justify-between items-center gap-4">
                  <div className="flex flex-col items-center md:items-start gap-2">
                    <a href="/" className="flex items-center gap-2">
                      <div className="rounded-lg bg-primary p-1">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="h-5 w-5 text-primary-foreground"
                        >
                          <path d="M22 9.33V4a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v5.33L12 16l10-6.67Z" />
                          <path d="M2 14.67V20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-5.33" />
                        </svg>
                      </div>
                      <span className="font-bold">Edu-Quest</span>
                    </a>
                    <p className="text-sm text-muted-foreground">Revolutionizing education through Web3 technology</p>
                  </div>
                  <div className="flex gap-8">
                    <div className="flex flex-col gap-2">
                      <h3 className="font-medium">Platform</h3>
                      <nav className="flex flex-col gap-2">
                        <a href="/courses" className="text-sm text-muted-foreground hover:text-foreground">
                          Courses
                        </a>
                        <a href="/learning-paths" className="text-sm text-muted-foreground hover:text-foreground">
                          Learning Paths
                        </a>
                        <a href="/internships" className="text-sm text-muted-foreground hover:text-foreground">
                          Internships
                        </a>
                      </nav>
                    </div>
                    <div className="flex flex-col gap-2">
                      <h3 className="font-medium">Company</h3>
                      <nav className="flex flex-col gap-2">
                        <a href="/about" className="text-sm text-muted-foreground hover:text-foreground">
                          About
                        </a>
                        <a href="/careers" className="text-sm text-muted-foreground hover:text-foreground">
                          Careers
                        </a>
                        <a href="/contact" className="text-sm text-muted-foreground hover:text-foreground">
                          Contact
                        </a>
                      </nav>
                    </div>
                  </div>
                </div>
                <div className="container mt-6 border-t pt-6">
                  <p className="text-center text-sm text-muted-foreground">
                    &copy; {new Date().getFullYear()} Edu-Quest. All rights reserved.
                  </p>
                </div>
              </footer>
            </div>
          </Web3Provider>
        </ThemeProvider>
      </body>
    </html>
  )
}



import './globals.css'