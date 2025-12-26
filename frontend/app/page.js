import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import Link from "next/link"

export default function LandingPage() {


  return (
    <div className="min-h-screen bg-white">
      <Navbar />

      {/* Hero Section */ }
      <main className="pt-32 pb-20">
        <div className="container mx-auto px-6">
          <div className="max-w-[800px] mx-auto text-center">
            <h1 className="text-[64px] leading-[1.1] font-semibold tracking-[-0.02em] mb-8">
              Detect Plagiarism,
              <br />
              Protect Originality
            </h1>
            <div className="space-y-4 mb-12">
              <p className="text-2xl text-gray-600">
                Advanced AI-powered plagiarism detection for academic integrity.
              </p>
              <p className="text-2xl text-gray-600">
                Compare your work against billions of sources instantly.
              </p>
            </div>
            <div className="flex items-center justify-center space-x-4">
              <Button asChild
                size="lg"
                className="h-14 px-8 text-lg bg-black hover:bg-black/90 rounded-md"
              >
                <Link href="/pages/submission">Check Document</Link>
              </Button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
