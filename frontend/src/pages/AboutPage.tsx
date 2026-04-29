// frontend/src/pages/AboutPage.tsx
import { 
  Cpu, 
  BarChart3, 
  GraduationCap,
  Zap,
  ShieldCheck,
  Smartphone,
  Battery,
  Target,
  FlaskConical,
  Binary,
  Microscope
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function AboutPage() {
  return (
    <div className="space-y-20 animate-in fade-in duration-700 pb-32 max-w-5xl mx-auto px-4 md:px-0">
      {/* 1. Hero / Academic Intro */}
      <div className="relative overflow-hidden rounded-[2.5rem] bg-linear-to-br from-primary/20 via-primary/5 to-background p-10 md:p-16 border border-primary/10 shadow-2xl shadow-primary/5">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold mb-8 tracking-widest uppercase">
            <GraduationCap className="size-4" /> Master's Thesis Project
          </div>
          <h1 className="text-5xl md:text-7xl font-black text-primary tracking-tighter mb-6 leading-none">
            OrchID <span className="text-foreground/90">NCD</span>
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground leading-relaxed font-light">
            A comprehensive Deep Learning framework for biodiversity monitoring, developed at the 
            <span className="text-foreground font-medium"> University of Naples Federico II</span>.
          </p>
        </div>
        <div className="absolute top-0 right-0 -translate-y-1/2 translate-x-1/4 size-[40rem] rounded-full bg-primary/5 blur-[120px]" />
      </div>

      {/* 2. Objectives: Embedding Space & NCD */}
      <section className="space-y-8">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-blue-500/10 flex items-center justify-center text-blue-500">
            <Target className="size-6" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">Core Objectives</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 text-muted-foreground leading-relaxed">
          <div className="space-y-4 text-sm md:text-base">
            <h3 className="text-foreground font-bold flex items-center gap-2">
              <Binary className="size-4 text-primary" /> Latent Embedding Space
            </h3>
            <p>
              One of the primary goals of this research is the construction of a high-dimensional 
              <strong> vector space</strong> (embedding space) where orchid images are mapped based on their 
              morphological features. In this space, images of the same species should cluster together, 
              while different species should be linearly separable. This serves as the foundation for 
              all downstream tasks, including similarity search and outlier detection.
            </p>
          </div>
          <div className="space-y-4 text-sm md:text-base">
            <h3 className="text-foreground font-bold flex items-center gap-2">
              <Microscope className="size-4 text-primary" /> Novel Class Discovery (NCD)
            </h3>
            <p>
              Traditional classifiers assume a "closed world" where all possible classes are known. 
              In biodiversity, this is rarely true. Our research focuses on <strong>Novel Class Discovery</strong>, 
              the ability of the model to recognize that a sample belongs to a species it has never seen 
              during training, effectively identifying "novel" entities in the wild without human intervention.
            </p>
          </div>
        </div>
      </section>

      {/* 3. Experimental Setup & Architecture Choices */}
      <section className="space-y-8 p-10 md:p-14 rounded-[3rem] bg-muted/20 border border-border/50">
        <div className="space-y-4 max-w-3xl">
          <div className="flex items-center gap-4">
            <div className="size-12 rounded-2xl bg-purple-500/10 flex items-center justify-center text-purple-500">
              <FlaskConical className="size-6" />
            </div>
            <h2 className="text-3xl font-bold tracking-tight">Experimental Setup</h2>
          </div>
          <p className="text-muted-foreground text-sm md:text-base leading-relaxed">
            Building a robust orchid classifier requires more than just high-quality images. We designed a 
            rigorous experimental pipeline to compare different inductive biases:
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <FeatureCard 
            icon={<Cpu className="size-5" />}
            title="Model Selection"
            description="We benchmarked ResNet (standard CNN), ConvNeXt (modernized CNN), and DINOv2 (Vision Transformer). This allows us to understand if local spatial hierarchies (CNNs) or global attention mechanisms (ViT) are better suited for subtle floral features."
          />
          <FeatureCard 
            icon={<Zap className="size-5" />}
            title="Optimization Strategy"
            description="While ResNet thrives on standard Adam, ConvNeXt requires AdamW with decoupled weight decay to avoid catastrophic overfitting. DINOv2 is fine-tuned using LoRA (Low-Rank Adaptation), preserving its massive pre-trained knowledge while specializing on orchids."
          />
          <FeatureCard 
            icon={<ShieldCheck className="size-5" />}
            title="Purity-First Splitting"
            description="We implemented an MD5-based deduplication system to prevent data leakage. The 'Clean Split' (Experiment 6) ensures that no photograph—even if renamed or sourced from multiple sites—exists in both training and testing sets."
          />
        </div>
      </section>

      {/* 4. Results & Deep Dive into Exp 6 */}
      <section className="space-y-8 px-4 md:px-0">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-green-500/10 flex items-center justify-center text-green-500">
            <BarChart3 className="size-6" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">Experiment Analysis: Serie 6</h2>
        </div>
        <div className="bg-muted/30 border border-border rounded-[2.5rem] p-8 md:p-12 overflow-hidden relative">
          <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 gap-12">
            <div className="space-y-6">
              <p className="text-muted-foreground leading-relaxed text-sm">
                <strong>Experiment 6</strong> is the cornerstone of our current results. After identifying 
                leakage in previous series (where duplicates artificially boosted test scores), Exp 6 
                provides a realistic assessment of model generalization.
              </p>
              <div className="space-y-4">
                <div className="flex items-start gap-4 p-4 rounded-2xl bg-background/50 border border-border/50">
                  <div className="text-primary font-bold text-lg">0.90</div>
                  <div className="text-[10px] text-muted-foreground">
                    <span className="text-foreground font-bold block mb-1">ResNet50 Peak Performance</span>
                    Our ResNet50 model reached a peak Val F1-Score of 0.905, demonstrating that with proper 5-fold validation, 
                    standard architectures are still highly competitive.
                  </div>
                </div>
                <div className="flex items-start gap-4 p-4 rounded-2xl bg-background/50 border border-border/50">
                  <div className="text-primary font-bold text-lg">ViT</div>
                  <div className="text-[10px] text-muted-foreground">
                    <span className="text-foreground font-bold block mb-1">Transformers vs CNNs</span>
                    DINOv2 (ViT) shows exceptional robustness to occlusions (leaves, hands) in field photos, 
                    outperforming CNNs in more cluttered environments.
                  </div>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-center bg-background/40 rounded-3xl border border-border/50 p-6">
              <div className="text-center space-y-2">
                <div className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">Current Training State</div>
                <div className="text-4xl font-black text-primary">DINOv2-Base</div>
                <div className="text-[10px] text-muted-foreground">Fold 5/5 in progress...</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Engineering Excellence */}
      <section className="space-y-8 px-4 md:px-0">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-orange-500/10 flex items-center justify-center text-orange-500">
            <Battery className="size-6" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">Engineering & Optimization</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <Card className="shadow-none border-border/60 bg-linear-to-b from-muted/50 to-background">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Smartphone className="size-4 text-primary" /> Mobile-First Deployment
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs text-muted-foreground space-y-4">
              <p>
                To allow botanists to use this app in the field, we implemented extreme backend optimizations. 
                Images are dynamically resized and converted to <strong>WebP</strong> format, reducing data 
                transfer by 90% while maintaining the visual clarity needed for classification.
              </p>
            </CardContent>
          </Card>
          <Card className="shadow-none border-border/60 bg-linear-to-b from-muted/50 to-background">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Battery className="size-4 text-orange-500" /> Battery-Aware Training
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs text-muted-foreground space-y-4">
              <p>
                The <code>smart-train.sh</code> system was developed to manage the heavy GPU workloads of 
                Vision Transformers. It monitors the host laptop's battery state, automatically pausing 
                training during power dips and resuming when safely plugged in.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 6. Footer / Academic Footer */}
      <footer className="pt-20 border-t border-border text-center space-y-8 px-4">
        <div className="flex flex-col md:flex-row justify-center gap-4">
          <div className="px-6 py-3 rounded-full bg-muted/50 border border-border text-[10px] font-bold uppercase tracking-wider">
            University of Naples Federico II
          </div>
          <div className="px-6 py-3 rounded-full bg-primary/10 border border-primary/20 text-primary text-[10px] font-bold uppercase tracking-widest">
            M.Sc. Computer Science
          </div>
        </div>
        <div className="max-w-2xl mx-auto space-y-4">
          <p className="text-muted-foreground text-sm leading-relaxed">
            This project is part of a Master's Thesis focused on the application of 
            Self-Supervised Learning and NCD to biodiversity data. 
            All experiments were conducted at the <strong>DIETI</strong> Department.
          </p>
          <div className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">
            Author: Marcello Russo
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <Card className="shadow-none border-border/50 bg-background/50 hover:border-primary/30 transition-all hover:-translate-y-1">
      <CardHeader className="pb-3 text-left">
        <div className="size-10 rounded-lg bg-primary/5 flex items-center justify-center text-primary mb-2">
          {icon}
        </div>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-left">
        <p className="text-[10px] text-muted-foreground leading-relaxed">{description}</p>
      </CardContent>
    </Card>
  );
}
