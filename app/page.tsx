import { HeroSection } from "@/components/hero-section"
import { FeatureSection } from "@/components/feature-section"
import { TestimonialSection } from "@/components/testimonial-section"
import { RoadmapSection } from "@/components/roadmap-section"
import { CallToAction } from "@/components/call-to-action"

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <main className="flex-1">
        <HeroSection />
        <FeatureSection />
        <RoadmapSection />
        <TestimonialSection />
        <CallToAction />
      </main>
    </div>
  )
}

