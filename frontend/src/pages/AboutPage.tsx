// frontend/src/pages/AboutPage.tsx
import { Link } from "react-router-dom";
import { 
  BookOpen, 
  Database, 
  Cpu, 
  BarChart3, 
  Lightbulb, 
  Microscope,
  Info
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function AboutPage() {
  return (
    <div className="space-y-12 animate-in fade-in duration-500 pb-20">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-linear-to-br from-primary/10 via-primary/5 to-background p-8 md:p-12 border border-primary/10">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-bold mb-6 tracking-wider uppercase">
            <Microscope className="size-3.5" /> Research Project
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-primary tracking-tight mb-4">
            OrchID <span className="text-foreground/80">NCD</span>
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground leading-relaxed">
            Un'infrastruttura di Deep Learning per la classificazione tassonomica del genere 
            <span className="italic font-medium text-foreground"> Ophrys</span> e lo studio della 
            New Class Discovery (NCD) nel contesto della biodiversità mediterranea.
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
            title="Contesto Scientifico"
          >
            <p>
              Il genere <strong>Ophrys</strong> rappresenta uno dei casi più complessi e affascinanti di evoluzione biologica guidata dagli impollinatori. Queste orchidee utilizzano il "mimetismo sessuale", emettendo feromoni e imitando la morfologia degli insetti per attirare i maschi di specifiche specie.
            </p>
            <p>
              Questa estrema specializzazione porta a una variabilità morfologica incredibile e a fenomeni di ibridazione frequenti, rendendo l'identificazione sul campo estremamente ardua anche per esperti botanici. Il progetto <strong>OrchID</strong> nasce per fornire strumenti computazionali oggettivi in grado di supportare il monitoraggio della biodiversità.
            </p>
          </Section>

          <Section 
            icon={<Database className="size-5" />} 
            title="Dataset & Data Integrity"
          >
            <p>
              Il dataset di riferimento (Exp 6) è stato curato per garantire l'assenza totale di <strong>data leakage</strong>. Attraverso un'analisi sistematica degli hash MD5, abbiamo identificato e rimosso duplicati cross-source che avrebbero inquinato le metriche di test.
            </p>
            <div className="grid grid-cols-2 gap-4 my-4">
              <div className="p-4 rounded-xl bg-muted/50 border border-border">
                <div className="text-2xl font-bold text-primary">2,317</div>
                <div className="text-xs text-muted-foreground uppercase">Immagini Totali</div>
              </div>
              <div className="p-4 rounded-xl bg-muted/50 border border-border">
                <div className="text-2xl font-bold text-primary">6</div>
                <div className="text-xs text-muted-foreground uppercase">Specie Target</div>
              </div>
            </div>
            <p>
              La suddivisione attuale prevede 2,017 immagini per il training e un test set bilanciato di 300 immagini (50 per classe), permettendo una valutazione imparziale su tutte le specie trattate.
            </p>
            <Link to="/dataset" className="inline-flex items-center text-primary font-medium hover:gap-2 transition-all">
              Esplora il dataset <span className="ml-1">→</span>
            </Link>
          </Section>

          <Section 
            icon={<Cpu className="size-5" />} 
            title="Architetture a Confronto"
          >
            <p>
              Non ci siamo limitati a un singolo modello, ma abbiamo messo a confronto le tre filosofie dominanti della Computer Vision moderna:
            </p>
            <ul className="space-y-4 mt-4">
              <li className="flex gap-3">
                <div className="size-2 rounded-full bg-primary mt-1.5 shrink-0" />
                <div>
                  <span className="font-bold">ResNet (CNN Classiche):</span> Lo standard industriale basato su connessioni residue. Ottimo equilibrio tra velocità e precisione.
                </div>
              </li>
              <li className="flex gap-3">
                <div className="size-2 rounded-full bg-primary mt-1.5 shrink-0" />
                <div>
                  <span className="font-bold">ConvNeXt (Modern CNN):</span> Un'evoluzione delle reti convoluzionali che adotta i miglioramenti strutturali dei Transformers (LayerNorm, Kernel più ampi) senza rinunciare all'efficienza spaziale.
                </div>
              </li>
              <li className="flex gap-3">
                <div className="size-2 rounded-full bg-primary mt-1.5 shrink-0" />
                <div>
                  <span className="font-bold">DINOv2 (Vision Transformers):</span> Modelli pre-addestrati con apprendimento auto-supervisionato su 142 milioni di immagini. Utilizziamo adattatori <strong>LoRA</strong> (Low-Rank Adaptation) per il fine-tuning specifico sulle orchidee.
                </div>
              </li>
            </ul>
          </Section>

          <Section 
            icon={<BarChart3 className="size-5" />} 
            title="Metodologia di Validazione"
          >
            <p>
              La robustezza dei risultati è garantita da una <strong>5-Fold Stratified Cross-Validation</strong>. Questo approccio riduce la varianza legata alla scelta casuale dei campioni, producendo una metrica media accompagnata dalla deviazione standard (±).
            </p>
            <p>
              Per l'addestramento utilizziamo una pipeline di <strong>Gradient Accumulation</strong> (Effective Batch Size: 32) che permette di mantenere un'alta precisione numerica anche su hardware con memoria limitata (GPU Quadro/Laptop).
            </p>
            <Link to="/experiments" className="inline-flex items-center text-primary font-medium hover:gap-2 transition-all">
              Analizza gli esperimenti <span className="ml-1">→</span>
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
                <div className="font-bold text-foreground mb-1">Serie 1: Benchmarking</div>
                <p>Confronto delle prestazioni base su dati puliti (Stato attuale).</p>
              </div>
              <div className="opacity-60">
                <div className="font-bold text-foreground mb-1">Serie 2: Contrastive Learning</div>
                <p>Implementazione di SupCon per migliorare la separazione nello spazio latente.</p>
              </div>
              <div className="opacity-60">
                <div className="font-bold text-foreground mb-1">Serie 3: Explainability</div>
                <p>Integrazione di mappe di calore Grad-CAM e Integrated Gradients per interpretare il focus dei modelli.</p>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-none bg-muted/30">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Info className="size-4" /> Glossario
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-3 font-mono">
              <div>
                <span className="text-primary">NCD:</span> New Class Discovery
              </div>
              <div>
                <span className="text-primary">LoRA:</span> Low-Rank Adaptation
              </div>
              <div>
                <span className="text-primary">F1-Score:</span> Media armonica tra Precision e Recall
              </div>
              <div>
                <span className="text-primary">Stochastic:</span> Processo casuale controllato (seed=57)
              </div>
            </CardContent>
          </Card>

          <div className="px-4 py-6 rounded-2xl border border-border bg-muted/10 text-center">
            <p className="text-xs text-muted-foreground mb-4">Progetto sviluppato nell'ambito del</p>
            <div className="font-bold text-sm tracking-widest uppercase text-foreground">
              National Center for Biodiversity
            </div>
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
      <div className="space-y-4 text-muted-foreground leading-relaxed">
        {children}
      </div>
    </div>
  );
}
