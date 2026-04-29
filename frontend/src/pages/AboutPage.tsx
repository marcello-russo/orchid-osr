// frontend/src/pages/AboutPage.tsx
import { Link } from "react-router-dom";
import { 
  BookOpen, 
  Database, 
  Cpu, 
  BarChart3, 
  Lightbulb, 
  Info,
  GraduationCap
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function AboutPage() {
  return (
    <div className="space-y-12 animate-in fade-in duration-500 pb-20">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-linear-to-br from-primary/10 via-primary/5 to-background p-8 md:p-12 border border-primary/10">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-bold mb-6 tracking-wider uppercase">
            <GraduationCap className="size-3.5" /> Master's Thesis Project
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-primary tracking-tight mb-4">
            OrchID <span className="text-foreground/80">NCD</span>
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground leading-relaxed">
            A Deep Learning infrastructure for the taxonomic classification of the 
            <span className="italic font-medium text-foreground"> Ophrys</span> genus and the study of 
            New Class Discovery (NCD) in the context of Mediterranean biodiversity.
          </p>
        </div>
        {/* Background decoration */}
        <div className="absolute top-0 right-0 -translate-y-1/2 translate-x-1/4 size-96 rounded-full bg-primary/5 blur-3xl" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Main Content Column */}
        <div className="md:col-span-2 space-y-8">
          
          <Section 
            icon={<BookOpen className="size-5" />} 
            title="Scientific Context"
          >
            <p>
              The <strong>Ophrys</strong> genus represents one of the most complex and fascinating cases of biological evolution driven by pollinators. These orchids utilize "sexual deception," emitting pheromones and mimicking the morphology of insects to attract males of specific species.
            </p>
            <p>
              This extreme specialization leads to incredible morphological variability and frequent hybridization, making field identification extremely difficult even for expert botanists. The <strong>OrchID</strong> project was born to provide objective computational tools capable of supporting biodiversity monitoring.
            </p>
          </Section>

          <Section 
            icon={<Database className="size-5" />} 
            title="Dataset & Data Integrity"
          >
            <p>
              The reference dataset (Exp 6) was curated to ensure total absence of <strong>data leakage</strong>. Through a systematic analysis of MD5 hashes, we identified and removed cross-source duplicates that would have contaminated the test metrics.
            </p>
            <div className="grid grid-cols-2 gap-4 my-4">
              <div className="p-4 rounded-xl bg-muted/50 border border-border">
                <div className="text-2xl font-bold text-primary">2,317</div>
                <div className="text-xs text-muted-foreground uppercase text-[10px]">Total Images</div>
              </div>
              <div className="p-4 rounded-xl bg-muted/50 border border-border">
                <div className="text-2xl font-bold text-primary">6</div>
                <div className="text-xs text-muted-foreground uppercase text-[10px]">Target Species</div>
              </div>
            </div>
            <p>
              The current split includes 2,017 images for training and a balanced test set of 300 images (50 per class), allowing for an unbiased evaluation across all treated species.
            </p>
            <Link to="/dataset" className="inline-flex items-center text-primary font-medium hover:gap-2 transition-all">
              Explore the dataset <span className="ml-1">→</span>
            </Link>
          </Section>

          <Section 
            icon={<Cpu className="size-5" />} 
            title="Architecture Comparison"
          >
            <p>
              We compared three dominant philosophies of modern Computer Vision:
            </p>
            <ul className="space-y-4 mt-4 text-sm">
              <li className="flex gap-3">
                <div className="size-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                <div>
                  <span className="font-bold">ResNet (Classic CNN):</span> The industry standard based on residual connections. Excellent balance between speed and accuracy.
                </div>
              </li>
              <li className="flex gap-3">
                <div className="size-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                <div>
                  <span className="font-bold">ConvNeXt (Modern CNN):</span> An evolution of convolutional networks that adopts Transformer structural improvements (LayerNorm, larger kernels) without sacrificing spatial efficiency.
                </div>
              </li>
              <li className="flex gap-3">
                <div className="size-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                <div>
                  <span className="font-bold">DINOv2 (Vision Transformers):</span> Models pre-trained with self-supervised learning on 142 million images. We use <strong>LoRA</strong> (Low-Rank Adaptation) adapters for specific fine-tuning on orchid species.
                </div>
              </li>
            </ul>
          </Section>

          <Section 
            icon={<BarChart3 className="size-5" />} 
            title="Validation Methodology"
          >
            <p>
              Result robustness is guaranteed by a <strong>5-Fold Stratified Cross-Validation</strong>. This approach reduces the variance associated with random sample choice, producing a mean metric accompanied by its standard deviation (±).
            </p>
            <p>
              For training, we utilize a <strong>Gradient Accumulation</strong> pipeline (Effective Batch Size: 32) that maintains high numerical precision even on hardware with limited memory (Quadro/Laptop GPUs).
            </p>
            <Link to="/experiments" className="inline-flex items-center text-primary font-medium hover:gap-2 transition-all">
              Analyze experiments <span className="ml-1">→</span>
            </Link>
          </Section>
        </div>

        {/* Sidebar Info Column */}
        <div className="space-y-6">
          <Card className="border-primary/20 bg-primary/5 shadow-none">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Lightbulb className="size-4 text-primary" /> Roadmap
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-4 text-muted-foreground">
              <div>
                <div className="font-bold text-foreground mb-1">Series 1: Benchmarking</div>
                <p>Comparison of baseline performance on clean data (Current State).</p>
              </div>
              <div className="opacity-60">
                <div className="font-bold text-foreground mb-1">Series 2: Contrastive Learning</div>
                <p>Implementation of SupCon to improve latent space separation.</p>
              </div>
              <div className="opacity-60">
                <div className="font-bold text-foreground mb-1">Series 3: Explainability</div>
                <p>Integration of Grad-CAM and Integrated Gradients to interpret model focus.</p>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-none bg-muted/30">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Info className="size-4" /> Glossary
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-3 font-mono">
              <div><span className="text-primary">NCD:</span> New Class Discovery</div>
              <div><span className="text-primary">LoRA:</span> Low-Rank Adaptation</div>
              <div><span className="text-primary">F1-Score:</span> Harmonic mean of Precision & Recall</div>
              <div><span className="text-primary">Stochastic:</span> Controlled random process (seed=57)</div>
            </CardContent>
          </Card>

          <div className="px-4 py-8 rounded-2xl border border-border bg-muted/10 text-center space-y-2">
            <p className="text-[10px] text-muted-foreground uppercase tracking-widest">Master's Thesis in Computer Science</p>
            <div className="font-bold text-sm text-foreground leading-tight">
              University of Naples Federico II
            </div>
            <div className="text-[10px] text-primary/80 font-medium">Department of Electrical Engineering and Information Technology</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Section({ 
  icon, 
  title, 
  children 
}: { 
  icon: React.ReactNode; 
  title: string; 
  children: React.ReactNode 
}) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2.5">
        <div className="p-2 rounded-lg bg-primary/10 text-primary">
          {icon}
        </div>
        <h2 className="text-xl font-bold tracking-tight">{title}</h2>
      </div>
      <div className="space-y-4 text-sm text-muted-foreground leading-relaxed">
        {children}
      </div>
    </div>
  );
}
