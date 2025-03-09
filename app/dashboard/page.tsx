"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { ethers } from "ethers"

export default function DashboardPage() {
  const [account, setAccount] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const checkConnection = async () => {
      try {
        if (typeof window.ethereum !== "undefined") {
          const provider = new ethers.BrowserProvider(window.ethereum)
          const accounts = await provider.send("eth_accounts", [])

          if (accounts.length > 0) {
            setAccount(accounts[0])
          }
        }
      } catch (error) {
        console.error("Error checking wallet connection:", error)
      } finally {
        setIsLoading(false)
      }
    }

    checkConnection()
  }, [])

  if (isLoading) {
    return (
      <div className="container flex items-center justify-center min-h-[calc(100vh-4rem)]">
        <div className="flex flex-col items-center gap-2">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
          <p className="text-sm text-muted-foreground">Loading your dashboard...</p>
        </div>
      </div>
    )
  }

  if (!account) {
    return (
      <div className="container flex items-center justify-center min-h-[calc(100vh-4rem)]">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">Wallet Not Connected</CardTitle>
            <CardDescription>Please connect your wallet to access your personalized dashboard.</CardDescription>
          </CardHeader>
          <CardFooter>
            <Link href="/connect" className="w-full">
              <Button className="w-full">Connect Wallet</Button>
            </Link>
          </CardFooter>
        </Card>
      </div>
    )
  }

  return (
    <div className="container py-8">
      <div className="flex flex-col gap-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
            <p className="text-muted-foreground">Welcome back! Track your progress and manage your learning journey.</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="px-3 py-1">
              <span className="text-xs text-muted-foreground mr-1">Connected:</span>
              <span className="text-xs font-medium">
                {account.substring(0, 6)}...{account.substring(account.length - 4)}
              </span>
            </Badge>
            <Button size="sm" variant="outline">
              Disconnect
            </Button>
          </div>
        </div>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="courses">My Courses</TabsTrigger>
            <TabsTrigger value="credentials">Credentials</TabsTrigger>
            <TabsTrigger value="internships">Internships</TabsTrigger>
          </TabsList>
          <TabsContent value="overview" className="mt-6">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Learning Progress</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Web3 Fundamentals</span>
                        <span className="text-sm text-muted-foreground">65%</span>
                      </div>
                      <Progress value={65} className="h-2" />
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Smart Contract Development</span>
                        <span className="text-sm text-muted-foreground">30%</span>
                      </div>
                      <Progress value={30} className="h-2" />
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">dApp Development</span>
                        <span className="text-sm text-muted-foreground">10%</span>
                      </div>
                      <Progress value={10} className="h-2" />
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" size="sm" className="w-full">
                    View All Courses
                  </Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Upcoming Deadlines</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-start gap-3">
                      <div className="rounded-full p-1 bg-primary/10">
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
                          className="h-4 w-4 text-primary"
                        >
                          <rect width="18" height="18" x="3" y="4" rx="2" ry="2" />
                          <line x1="16" x2="16" y1="2" y2="6" />
                          <line x1="8" x2="8" y1="2" y2="6" />
                          <line x1="3" x2="21" y1="10" y2="10" />
                        </svg>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm font-medium">Smart Contract Assignment</p>
                        <p className="text-xs text-muted-foreground">Due in 2 days</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="rounded-full p-1 bg-primary/10">
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
                          className="h-4 w-4 text-primary"
                        >
                          <rect width="18" height="18" x="3" y="4" rx="2" ry="2" />
                          <line x1="16" x2="16" y1="2" y2="6" />
                          <line x1="8" x2="8" y1="2" y2="6" />
                          <line x1="3" x2="21" y1="10" y2="10" />
                        </svg>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm font-medium">Web3 Quiz</p>
                        <p className="text-xs text-muted-foreground">Due in 5 days</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="rounded-full p-1 bg-primary/10">
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
                          className="h-4 w-4 text-primary"
                        >
                          <rect width="18" height="18" x="3" y="4" rx="2" ry="2" />
                          <line x1="16" x2="16" y1="2" y2="6" />
                          <line x1="8" x2="8" y1="2" y2="6" />
                          <line x1="3" x2="21" y1="10" y2="10" />
                        </svg>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm font-medium">Final Project Submission</p>
                        <p className="text-xs text-muted-foreground">Due in 2 weeks</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" size="sm" className="w-full">
                    View Calendar
                  </Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Verified Credentials</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-start gap-3">
                      <div className="rounded-full p-1 bg-primary/10">
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
                          className="h-4 w-4 text-primary"
                        >
                          <path d="M12.3 2.9c.4-.2.9-.2 1.4 0l6 3.5c.5.3.8.8.8 1.3v5.5c0 2.9-1.1 5.7-3.1 7.7l-4.3 4.2c-.4.4-1 .4-1.4 0l-4.3-4.2c-2-2-3.1-4.8-3.1-7.7V7.7c0-.5.3-1 .8-1.3l6-3.5z" />
                          <path d="m9 12 2 2 4-4" />
                        </svg>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm font-medium">Blockchain Basics</p>
                        <p className="text-xs text-muted-foreground">Verified on Ethereum</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="rounded-full p-1 bg-primary/10">
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
                          className="h-4 w-4 text-primary"
                        >
                          <path d="M12.3 2.9c.4-.2.9-.2 1.4 0l6 3.5c.5.3.8.8.8 1.3v5.5c0 2.9-1.1 5.7-3.1 7.7l-4.3 4.2c-.4.4-1 .4-1.4 0l-4.3-4.2c-2-2-3.1-4.8-3.1-7.7V7.7c0-.5.3-1 .8-1.3l6-3.5z" />
                          <path d="m9 12 2 2 4-4" />
                        </svg>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm font-medium">Cryptography Fundamentals</p>
                        <p className="text-xs text-muted-foreground">Verified on Ethereum</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" size="sm" className="w-full">
                    View All Credentials
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="courses" className="mt-6">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">Web3 Fundamentals</CardTitle>
                    <Badge>In Progress</Badge>
                  </div>
                  <CardDescription>Learn the core concepts of blockchain technology</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Overall Progress</span>
                        <span className="text-sm text-muted-foreground">65%</span>
                      </div>
                      <Progress value={65} className="h-2" />
                    </div>
                    <div className="space-y-1">
                      <p className="text-sm font-medium">Next Lesson:</p>
                      <p className="text-sm text-muted-foreground">Smart Contract Basics</p>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button size="sm" className="w-full">
                    Continue Learning
                  </Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">Smart Contract Development</CardTitle>
                    <Badge>In Progress</Badge>
                  </div>
                  <CardDescription>Master Solidity programming for blockchain</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Overall Progress</span>
                        <span className="text-sm text-muted-foreground">30%</span>
                      </div>
                      <Progress value={30} className="h-2" />
                    </div>
                    <div className="space-y-1">
                      <p className="text-sm font-medium">Next Lesson:</p>
                      <p className="text-sm text-muted-foreground">Contract Security & Auditing</p>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button size="sm" className="w-full">
                    Continue Learning
                  </Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">dApp Development</CardTitle>
                    <Badge variant="outline">Not Started</Badge>
                  </div>
                  <CardDescription>Build full-stack decentralized applications</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Overall Progress</span>
                        <span className="text-sm text-muted-foreground">0%</span>
                      </div>
                      <Progress value={0} className="h-2" />
                    </div>
                    <div className="space-y-1">
                      <p className="text-sm font-medium">Prerequisites:</p>
                      <p className="text-sm text-muted-foreground">Complete Smart Contract Development</p>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button size="sm" variant="outline" className="w-full">
                    Start Course
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="credentials" className="mt-6">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Blockchain Basics</CardTitle>
                  <CardDescription>Foundational knowledge of blockchain technology</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                        Verified
                      </Badge>
                      <span className="text-xs text-muted-foreground">Issued: March 15, 2023</span>
                    </div>
                    <div className="rounded-md bg-muted p-3">
                      <p className="text-xs font-mono break-all">0x8a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b</p>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button size="sm" variant="outline">
                    View Certificate
                  </Button>
                  <Button size="sm" variant="outline">
                    Share
                  </Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Cryptography Fundamentals</CardTitle>
                  <CardDescription>Essential cryptographic concepts for blockchain</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                        Verified
                      </Badge>
                      <span className="text-xs text-muted-foreground">Issued: April 22, 2023</span>
                    </div>
                    <div className="rounded-md bg-muted p-3">
                      <p className="text-xs font-mono break-all">0x7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d</p>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button size="sm" variant="outline">
                    View Certificate
                  </Button>
                  <Button size="sm" variant="outline">
                    Share
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="internships" className="mt-6">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Blockchain Developer Intern</CardTitle>
                  <CardDescription>TechFusion Labs</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Badge>Remote</Badge>
                      <Badge variant="outline">3 Months</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Work on developing and testing smart contracts for decentralized finance applications. Gain
                      hands-on experience with Solidity, Truffle, and Web3.js.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="secondary">Solidity</Badge>
                      <Badge variant="secondary">Smart Contracts</Badge>
                      <Badge variant="secondary">DeFi</Badge>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button className="w-full">Apply Now</Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Web3 Frontend Developer</CardTitle>
                  <CardDescription>MetaVerse Innovations</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Badge>Hybrid</Badge>
                      <Badge variant="outline">6 Months</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Design and implement user interfaces for decentralized applications. Work with React, Next.js, and
                      Ethers.js to create seamless Web3 experiences.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="secondary">React</Badge>
                      <Badge variant="secondary">Next.js</Badge>
                      <Badge variant="secondary">Ethers.js</Badge>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button className="w-full">Apply Now</Button>
                </CardFooter>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

