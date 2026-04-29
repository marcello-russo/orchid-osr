// frontend/src/pages/AboutPage.tsx
import { 
  BookOpen, 
  Cpu, 
  BarChart3, 
  GraduationCap,
  Zap,
  ShieldCheck,
  Smartphone,
  Battery
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function AboutPage() {
  return (
    <div className="space-y-16 animate-in fade-in duration-700 pb-24 max-w-5xl mx-auto">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-[2rem] bg-linear-to-br from-primary/15 via-primary/5 to-background p-10 md:p-16 border border-primary/10 shadow-2xl shadow-primary/5">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold mb-8 tracking-widest uppercase">
            <GraduationCap className="size-4" /> Master's Thesis Project
          </div>
          <h1 className="text-5xl md:text-6xl font-black text-primary tracking-tighter mb-6 leading-none">
            OrchID <span className="text-foreground/90">NCD</span>
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground leading-relaxed font-light">
            An end-to-end research ecosystem built at the 
            <span className="text-foreground font-medium"> University of Naples Federico II</span>, 
            bridging the gap between field botany and state-of-the-art Computer Vision.
          </p>
        </div>
        {/* Background decoration */}
        <div className="absolute top-0 right-0 -translate-y-1/2 translate-x-1/4 size-[30rem] rounded-full bg-primary/5 blur-[100px]" />
      </div>

      {/* Chapter 1: The Vision */}
      <section className="space-y-6 px-4 md:px-0">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary">
            <BookOpen className="size-6" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">The Vision & Scientific Context</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 text-muted-foreground leading-relaxed">
          <p>
            The identification of <strong>Ophrys</strong> orchids is a notorious challenge in Mediterranean botany. Due to pollinator-driven evolution, these species exhibit extreme morphological plasticity and frequent hybridization. Traditional classification methods often struggle with the subtle variations in labellum patterns and floral structure.
          </p>
          <p>
            This project, developed as a Master's Thesis, approaches the problem through the lens of <strong>New Class Discovery (NCD)</strong>. We don't just want a closed-set classifier; we are building a framework capable of handling the inherent uncertainty of biodiversity data, ensuring that every prediction is backed by a rigorous, reproducible pipeline.
          </p>
        </div>
      </section>

      {/* Chapter 2: Engineering for Performance */}
      <section className="space-y-8 p-8 md:p-12 rounded-[2rem] bg-muted/30 border border-border">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <h2 className="text-3xl font-bold flex items-center gap-3">
              <Zap className="size-8 text-primary" /> Engineering for Performance
            </h2>
            <p className="text-muted-foreground">How we turned a complex DL model into a lightning-fast web experience.</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <FeatureCard 
            icon={<Cpu className="size-5" />}
            title="Docker & Build-Time Caching"
            description="Instead of downloading models at startup, we optimized our Dockerfile to fetch and cache all weights during the build phase. This reduced Hugging Face Space startup time from minutes to seconds."
          />
          <FeatureCard 
            icon={<Smartphone className="size-5" />}
            title="Dynamic WebP Optimization"
            description="To handle high-resolution field photos on mobile, we implemented a custom backend resizing engine. Images are dynamically scaled and converted to WebP, reducing bandwidth usage by over 90%."
          />
          <FeatureCard 
            icon={<BarChart3 className="size-5" />}
            title="Responsive Visualization"
            description="Our custom-built Distribution Charts adapt dynamically to screen size, ensuring that complex species statistics remain readable even on small smartphone viewports."
          />
        </div>
      </section>

      {/* Chapter 3: Data Integrity */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center px-4 md:px-0">
        <div className="space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-green-500/10 text-green-600 text-xs font-bold uppercase tracking-wider">
            <ShieldCheck className="size-3.5" /> Zero Data Leakage
          </div>
          <h2 className="text-3xl font-bold">The "Clean Split" Methodology</h2>
          <div className="space-y-4 text-muted-foreground leading-relaxed">
            <p>
              In computer vision, <strong>data leakage</strong> is the silent killer of model generalization. We discovered that several internet sources contained the same orchid photographs under different filenames.
            </p>
            <p>
              To solve this, we implemented a systematic <strong>MD5 hash deduplication</strong> across the entire dataset. This led to the creation of <em>Experiment 6 (Clean Split)</em>: a robust dataset where training and test sets are strictly disjoint at the pixel level, ensuring that our results are genuinely earned.
            </p>
          </div>
        </div>
        <div className="bg-muted/50 rounded-3xl p-8 border border-border">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Dataset Purity</span>
              <span className="text-xs font-bold text-green-500">100%</span>
            </div>
            <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-green-500 w-full" />
            </div>
            <ul className="space-y-3 pt-4">
              <li className="flex items-center gap-3 text-sm text-muted-foreground">
                <div className="size-1.5 rounded-full bg-green-500" />
                Deduplicated via MD5 Hashes
              </li>
              <li className="flex items-center gap-3 text-sm text-muted-foreground">
                <div className="size-1.5 rounded-full bg-green-500" />
                Balanced 50-image test sets per class
              </li>
              <li className="flex items-center gap-3 text-sm text-muted-foreground">
                <div className="size-1.5 rounded-full bg-green-500" />
                Stratified 5-Fold Cross Validation
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Chapter 4: Smart Training */}
      <section className="space-y-8 px-4 md:px-0">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-orange-500/10 flex items-center justify-center text-orange-500">
            <Battery className="size-6" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">Smart Training Infrastructure</h2>
        </div>
        <div className="bg-linear-to-r from-muted/50 to-transparent p-8 rounded-3xl border-l-4 border-orange-500">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div className="space-y-4">
              <p className="text-muted-foreground leading-relaxed">
                Training deep models like ConvNeXt and DINOv2 requires massive computational power. To manage this on portable hardware, we developed <code>smart-train.sh</code>: a battery-aware management script.
              </p>
              <p className="text-muted-foreground leading-relaxed text-sm italic">
                "The script monitors the laptop's power state in real-time. If the battery drops below 40%, it gracefully pauses the training at the next epoch boundary to avoid data corruption. It automatically resumes once the charge reaches 80%."
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-background border border-border rounded-md text-[10px] font-mono uppercase">Battery Monitor v2.1</span>
              <span className="px-3 py-1 bg-background border border-border rounded-md text-[10px] font-mono uppercase">Auto-Resume Logic</span>
              <span className="px-3 py-1 bg-background border border-border rounded-md text-[10px] font-mono uppercase">Epoch Boundary Safety</span>
              <span className="px-3 py-1 bg-background border border-border rounded-md text-[10px] font-mono uppercase">Multi-GPU Ready</span>
            </div>
          </div>
        </div>
      </section>

      {/* Footer / Attribution */}
      <footer className="pt-16 border-t border-border text-center space-y-6 px-4 md:px-0">
        <div className="flex flex-col md:flex-row justify-center gap-4">
          <div className="px-6 py-3 rounded-full bg-muted/50 border border-border text-xs font-medium uppercase tracking-wider">
            University of Naples Federico II
          </div>
          <div className="px-6 py-3 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold uppercase tracking-widest">
            Computer Science Department
          </div>
        </div>
        <p className="text-muted-foreground text-sm max-w-lg mx-auto leading-relaxed">
          Developed as a Master's Thesis project for the Master's Degree in Computer Science, 
          at the DIETI Department (Dipartimento di Ingegneria Elettrica e delle Tecnologie dell'Informazione).
        </p>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <Card className="shadow-none border-border/50 bg-background/50 hover:border-primary/30 transition-colors">
      <CardHeader className="pb-3 text-left">
        <div className="size-10 rounded-lg bg-primary/5 flex items-center justify-center text-primary mb-2">
          {icon}
        </div>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-left">
        <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>
      </CardContent>
    </Card>
  );
}
