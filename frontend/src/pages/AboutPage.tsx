import { useTranslation } from "react-i18next";
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
  Microscope,
  BookOpen
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function AboutPage() {
  const { t } = useTranslation();

  return (
    <div className="space-y-20 animate-in fade-in duration-700 pb-32 max-w-5xl mx-auto px-4 md:px-0">
      {/* 1. Hero / Academic Intro */}
      <div className="relative overflow-hidden rounded-[2.5rem] bg-linear-to-br from-primary/20 via-primary/5 to-background p-10 md:p-16 border border-primary/10 shadow-2xl shadow-primary/5">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold mb-8 tracking-widest uppercase">
            <GraduationCap className="size-4" /> {t('about.hero.tag')}
          </div>
          <h1 className="text-5xl md:text-7xl font-black text-primary tracking-tighter mb-6 leading-none">
            {t('about.hero.title')} <span className="text-foreground/90">{t('about.hero.subtitle')}</span>
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground leading-relaxed font-light">
            {t('about.hero.description')}
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
          <h2 className="text-3xl font-bold tracking-tight">{t('about.objectives.title')}</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 text-muted-foreground leading-relaxed">
          <div className="space-y-4 text-sm md:text-base">
            <h3 className="text-foreground font-bold flex items-center gap-2">
              <Binary className="size-4 text-primary" /> {t('about.objectives.vectorSpace.title')}
            </h3>
            <p>
              {t('about.objectives.vectorSpace.description')}
            </p>
          </div>
          <div className="space-y-4 text-sm md:text-base">
            <h3 className="text-foreground font-bold flex items-center gap-2">
              <Microscope className="size-4 text-primary" /> {t('about.objectives.ncd.title')}
            </h3>
            <p>
              {t('about.objectives.ncd.description')}
            </p>
          </div>
        </div>
      </section>

      {/* 3. Scientific Context */}
      <section className="space-y-6">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary">
            <BookOpen className="size-6" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">{t('about.scientific.title')}</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 text-muted-foreground leading-relaxed text-sm">
          <p>{t('about.scientific.content1')}</p>
          <p>{t('about.scientific.content2')}</p>
        </div>
      </section>

      {/* 4. Experimental Setup */}
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
            description="We implemented an MD5-based deduplication system to prevent data leakage. The 'Clean Split' (Experiment 6) ensures that no photograph exists in both training and testing sets."
          />
        </div>
      </section>

      {/* 5. Results & Deep Dive into Exp 6 */}
      <section className="space-y-8 px-4 md:px-0">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-green-500/10 flex items-center justify-center text-green-500">
            <BarChart3 className="size-6" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">{t('about.experiment6.title')}</h2>
        </div>
        <div className="bg-muted/30 border border-border rounded-[2.5rem] p-8 md:p-12 overflow-hidden relative">
          <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 gap-12">
            <div className="space-y-6">
              <p className="text-muted-foreground leading-relaxed text-sm">
                {t('about.experiment6.description')}
              </p>
              <div className="space-y-4">
                <div className="flex items-start gap-4 p-4 rounded-2xl bg-background/50 border border-border/50">
                  <div className="text-primary font-bold text-lg">0.91</div>
                  <div className="text-[10px] text-muted-foreground">
                    <span className="text-foreground font-bold block mb-1">{t('about.experiment6.vit')}</span>
                    {t('about.experiment6.vitDesc')}
                  </div>
                </div>
                <div className="flex items-start gap-4 p-4 rounded-2xl bg-background/50 border border-border/50">
                  <div className="text-primary font-bold text-lg">0.90</div>
                  <div className="text-[10px] text-muted-foreground">
                    <span className="text-foreground font-bold block mb-1">{t('about.experiment6.resnet')}</span>
                    {t('about.experiment6.resnetDesc')}
                  </div>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-center bg-background/40 rounded-3xl border border-border/50 p-6">
              <div className="text-center space-y-2">
                <div className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">Latest Completed Model</div>
                <div className="text-4xl font-black text-primary uppercase">DINOv2-Base</div>
                <div className="text-[10px] text-muted-foreground">All 5 Folds Complete</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 6. Engineering Excellence */}
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

      {/* 7. Footer / Academic Footer */}
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
