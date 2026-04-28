// frontend/src/pages/AboutPage.tsx
import { Link } from "react-router-dom";
import { BookOpen } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function AboutPage() {
  return (
    <div className="space-y-8 animate-in fade-in duration-300 max-w-3xl">
      <h1 className="text-2xl font-bold text-primary flex items-center gap-2">
        <BookOpen className="size-6" />
        About This Project
      </h1>

      <Section title="1. Introduction">
        <p>
          OrchID NCD is a research project for <strong>Ophrys</strong> orchid species
          classification, developed as part of the National Center for Biodiversity (NCD)
          initiative. The goal is to build an accurate, reproducible classifier that can
          distinguish 6 Ophrys species from field photographs.
        </p>
        <p>
          Accurate orchid classification supports biodiversity monitoring and conservation
          efforts. Ophrys species are morphologically similar and often confused, making
          automated classification a valuable tool for field botanists.
        </p>
        <p>
          <Link to="/dataset" className="text-primary hover:underline">Explore the dataset &rarr;</Link>
        </p>
      </Section>

      <Section title="2. The Dataset">
        <p>
          The dataset contains 2,317 images across 6 Ophrys species, sourced from internet
          repositories and private field collections. The key challenge was ensuring{" "}
          <strong>zero data leakage</strong> between training and test sets.
        </p>
        <p>
          An MD5 hash comparison revealed that 15 duplicate images existed under different
          filenames across sources. These were removed from training. The final clean split
          (exp6) uses 2,017 training images and 300 test images (50 per class, balanced).
        </p>
        <p>
          <Link to="/experiments" className="text-primary hover:underline">View all experiments &rarr;</Link>
        </p>
      </Section>

      <Section title="3. Methodology">
        <p>
          <strong>K-fold cross-validation.</strong> All models are evaluated with 5-fold
          stratified cross-validation (seed=57). This provides robust evaluation rather than
          relying on a single train/val split, and produces mean &plusmn; std metrics that
          reflect true generalization.
        </p>
        <p>
          <strong>Patience-based early stopping.</strong> Training stops after 15 consecutive
          epochs without validation loss improvement. We initially tested the Prechelt
          Generalization Loss (GL) criterion, but found it unreliable on this dataset —
          val_loss fluctuations in early epochs triggered premature stops. Patience-only is
          sufficient: if the model diverges, val_loss won't improve for 15 epochs and patience
          catches it.
        </p>
        <p>
          <strong>Per-architecture training recipes.</strong> Not all architectures respond
          equally to the same optimizer settings. ResNet works well with standard Adam
          (lr=0.001, no weight decay). ConvNeXt severely overfits with that recipe — its
          LayerNorm layers are sensitive to large gradients, and without decoupled weight
          decay, train_loss drops to 0.01 while val_loss rises above 2.0. The ConvNeXt paper
          recipe (AdamW, lr=0.0001, wd=0.05) resolved this. DINOv2 uses AdamW with low
          weight decay on its LoRA adapters.
        </p>
        <p>
          <strong>Gradient accumulation.</strong> Physical batch size is 8 (fits 4GB VRAM on
          a Quadro T2000). Gradient accumulation over 4 steps gives an effective batch size
          of 32, which is mathematically identical to a single batch=32 forward pass for
          cross-entropy loss. No mixed precision — we previously found FP16 distorted training
          results.
        </p>
      </Section>

      <Section title="4. Results">
        <p>
          Serie 1 compares 6 architectures with cross-entropy loss on the clean split.
          Results so far:
        </p>
        <ul className="list-disc list-inside space-y-1 text-sm">
          <li><strong>ResNet18</strong> — Val F1: 0.868 &plusmn; 0.016, Test F1: 0.850</li>
          <li><strong>ResNet50</strong> — Val F1: 0.905 &plusmn; 0.027, Test F1: 0.880</li>
          <li><strong>ConvNeXt-Tiny</strong> — Training in progress</li>
          <li><strong>ConvNeXt-Small, DINOv2-Small, DINOv2-Base</strong> — Pending</li>
        </ul>
        <p>
          <Link to="/experiments" className="text-primary hover:underline">See detailed results &rarr;</Link>
        </p>
      </Section>

      <Section title="5. Next Steps">
        <ul className="list-disc list-inside space-y-1 text-sm">
          <li><strong>Serie 2:</strong> Supervised Contrastive Learning (SupCon) — trains same-class proximity and different-class separation in embedding space</li>
          <li><strong>Serie 3:</strong> Explainability — occlusion sensitivity maps, IntegratedGradients attribution, per-class saliency analysis</li>
        </ul>
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader className="pb-2 border-b border-border/50">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="pt-4 space-y-3 text-sm text-muted-foreground leading-relaxed">
        {children}
      </CardContent>
    </Card>
  );
}
