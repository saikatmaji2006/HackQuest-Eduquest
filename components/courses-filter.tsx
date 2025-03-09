"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Slider } from "@/components/ui/slider"

export function CoursesFilter() {
  const [priceRange, setPriceRange] = useState([0, 100])

  return (
    <div className="w-full md:w-72 space-y-4">
      <div className="bg-card rounded-md border shadow-sm p-4">
        <h3 className="font-medium mb-3">Filters</h3>
        <div className="space-y-5">
          <Accordion type="multiple" defaultValue={["categories", "level"]}>
            <AccordionItem value="categories">
              <AccordionTrigger>Categories</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Checkbox id="web3" />
                    <Label htmlFor="web3">Web3 Basics</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox id="blockchain" />
                    <Label htmlFor="blockchain">Blockchain</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox id="smart-contracts" />
                    <Label htmlFor="smart-contracts">Smart Contracts</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox id="defi" />
                    <Label htmlFor="defi">DeFi</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox id="nft" />
                    <Label htmlFor="nft">NFTs</Label>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="level">
              <AccordionTrigger>Level</AccordionTrigger>
              <AccordionContent>
                <RadioGroup defaultValue="all">
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="all" id="all" />
                    <Label htmlFor="all">All Levels</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="beginner" id="beginner" />
                    <Label htmlFor="beginner">Beginner</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="intermediate" id="intermediate" />
                    <Label htmlFor="intermediate">Intermediate</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="advanced" id="advanced" />
                    <Label htmlFor="advanced">Advanced</Label>
                  </div>
                </RadioGroup>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="duration">
              <AccordionTrigger>Duration</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Checkbox id="short" />
                    <Label htmlFor="short">0-4 weeks</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox id="medium" />
                    <Label htmlFor="medium">5-8 weeks</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox id="long" />
                    <Label htmlFor="long">9+ weeks</Label>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="price">
              <AccordionTrigger>Price Range</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-4">
                  <Slider defaultValue={[0, 100]} max={100} step={1} value={priceRange} onValueChange={setPriceRange} />
                  <div className="flex items-center justify-between">
                    <span className="text-sm">${priceRange[0]}</span>
                    <span className="text-sm">${priceRange[1]}</span>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="rating">
              <AccordionTrigger>Rating</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Checkbox id="rating-4" />
                    <Label htmlFor="rating-4" className="flex">
                      <span>4.5 & up</span>
                      <div className="ml-2 flex">
                        {[...Array(5)].map((_, i) => (
                          <svg
                            key={i}
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill={i < 4 ? "currentColor" : "none"}
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
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox id="rating-3" />
                    <Label htmlFor="rating-3" className="flex">
                      <span>3.5 & up</span>
                      <div className="ml-2 flex">
                        {[...Array(5)].map((_, i) => (
                          <svg
                            key={i}
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill={i < 3 ? "currentColor" : "none"}
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
                    </Label>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <Button className="w-full">Apply Filters</Button>
          <Button variant="outline" className="w-full">
            Reset
          </Button>
        </div>
      </div>
    </div>
  )
}

