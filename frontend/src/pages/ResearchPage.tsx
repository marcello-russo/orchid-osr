import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Callout } from "@/components/ui/Callout";
import { Lightbox } from "@/components/ui/Lightbox";
import { Cite } from "@/features/research/components/Citation";
import { DataTable } from "@/features/research/components/DataTable";
import { PAPERS } from "@/features/research/data/papers";
import { cn } from "@/lib/utils";
import { CiteProvider } from "@/features/research/components/CiteContext";

/* ── Sections index ── */

const SECTIONS = [
  { id: "intro", label: "Introduzione" },
  { id: "contesto", label: "Contesto Botanico" },
  { id: "dataset", label: "Dataset" },
  { id: "metodologia", label: "Metodologia" },
  { id: "exp6", label: "Exp 6 — CE" },
  { id: "exp7", label: "Exp 7 — SupCon" },
  { id: "exp8", label: "Exp 8 — SpHOR" },
  { id: "exp10", label: "Exp 10 — MGH-NCD" },
  { id: "benchmark", label: "Benchmark" },
  { id: "conclusioni", label: "Conclusioni" },
  { id: "bibliografia", label: "Bibliografia" },
];

/* ── Hooks ── */

function useActiveSection() {
  const [active, setActive] = useState("");
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) { setActive(e.target.id); break; }
        }
      },
      { rootMargin: "-80px 0px -60% 0px" },
    );
    for (const { id } of SECTIONS) {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, []);
  return active;
}

/* ── Shared components ── */

function Figure({ src, alt, caption, className }: { src: string; alt: string; caption?: string; className?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <figure className="my-8">
      <div className="flex justify-center">
        <img
          src={src} alt={alt}
          className={cn("rounded-xl border border-border cursor-pointer hover:opacity-90 transition-opacity", className || "max-w-full")}
          onClick={() => setOpen(true)}
        />
      </div>
      {caption && <figcaption className="text-sm text-muted-foreground text-center mt-2 italic">{caption}</figcaption>}
      {open && <Lightbox src={src} alt={alt} caption={caption} onClose={() => setOpen(false)} />}
    </figure>
  );
}

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 mb-16">
      <h2 className="text-2xl font-bold tracking-tight mb-6 pb-2 border-b border-border/50">{title}</h2>
      {children}
    </section>
  );
}

function SubSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-8">
      <h3 className="text-lg font-semibold tracking-tight mb-3">{title}</h3>
      {children}
    </div>
  );
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="text-muted-foreground leading-relaxed mb-3">{children}</p>;
}

/* ── Main page ── */

export function ResearchPage() {
  const active = useActiveSection();

  return (
    <CiteProvider>
    <div className="relative max-w-7xl mx-auto px-4 md:px-8">
      {/* Sidebar TOC */}
      <aside className="fixed left-0 top-24 bottom-0 w-64 hidden xl:block overflow-y-auto border-r border-border/50 pr-4 pl-6">
        <nav className="space-y-1 pt-6">
          <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground px-3 pb-2">Indice</p>
          {SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className={cn(
                "block px-3 py-1.5 text-sm rounded-md transition-colors",
                active === s.id
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
              )}
            >
              {s.label}
            </a>
          ))}
        </nav>
      </aside>

      {/* Main */}
      <main className="research-content relative xl:ml-64 py-12 max-w-4xl">

        {/* ── HERO ── */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-primary/10 via-primary/5 to-background border border-border/50 mb-12">
          <div className="p-8 md:p-12">
            <Badge variant="outline" className="mb-4 text-xs">Tesi Magistrale — 2024/2025</Badge>
            <h1 className="text-3xl md:text-4xl font-black tracking-tight leading-tight">
              Ultra-Fine-Grained Classification of Ophrys Orchids
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground mt-1">
              via Deep Learning and Supervised Contrastive Learning
            </p>
            <div className="mt-6 text-sm text-muted-foreground space-y-1">
              <p><span className="font-medium text-foreground">Autore:</span> Marcello Russo</p>
              <p><span className="font-medium text-foreground">Relatore:</span> Luigi Libero Lucio Starace</p>
              <p><span className="font-medium text-foreground">Correlatori:</span> Mara Sangiovanni, Alessandro De Luca</p>
              <p className="mt-3 italic max-w-2xl text-xs">
                Università degli Studi di Napoli Federico II — DIETI — Laurea Magistrale in Informatica
              </p>
            </div>
          </div>
        </div>

        {/* ── INTRODUZIONE ── */}
        <Section id="intro" title="Introduzione">
          <P>
            Il genere <em>Ophrys</em> comprende oltre 300 specie di orchidee europee caratterizzate
            da una straordinaria somiglianza morfologica, che rende la loro identificazione
            tassonomica estremamente complessa persino per botanici esperti. Questa difficoltà,
            nota come classificazione <em>ultra-fine-grained</em> (UFGVC)<Cite id="yu2021benchmark" />,
            rappresenta una sfida significativa per i sistemi di visione artificiale.
          </P>
          <P>
            Questa tesi affronta il problema confrontando sistematicamente sei architetture
            (ResNet, ConvNeXt, DINOv2) su tre paradigmi di apprendimento: entropia incrociata,
            apprendimento contrastivo supervisionato (SupCon)<Cite id="khosla2020supervised" />
            e prototipi sferici ortogonali (SpHOR)<Cite id="bahavan2026sphor" />.
          </P>
          <SubSection title="Contributi">
            <ul className="list-disc list-inside text-muted-foreground space-y-1 text-sm">
              <li>Benchmark multi-architettura su tre paradigmi di apprendimento</li>
              <li>Dataset curato di 3.040 immagini con split train/test privi di data leakage</li>
              <li>Analisi dello spazio latente tramite proiezioni t-SNE</li>
              <li>Valutazione open-set con tecniche Mahalanobis, KNN ed Energy<Cite id="liu2020energy" /></li>
            </ul>
          </SubSection>
        </Section>

        {/* ── 1. CONTESTO BOTANICO ── */}
        <Section id="contesto" title="Contesto Botanico">
          <SubSection title="Il Genere Ophrys">
            <P>
              Il genere <em>Ophrys</em> (Orchidaceae) comprende orchidee terrestri diffuse in tutto
              il bacino del Mediterraneo. Le piante sono perenni, con foglie basali a rosetta e uno
              stelo fiorale che produce da 2 a 12 fiori. Ogni fiore è caratterizzato da un grande
              labello vellutato che costituisce l&apos;elemento morfologico più vistoso e diagnostico.
            </P>
          </SubSection>
          <SubSection title="Mimetismo Sessuale">
            <P>
              Le orchidee del genere <em>Ophrys</em> sono note per il loro sistema di impollinazione
              basato sul mimetismo sessuale (pseudocopulazione)<Cite id="schiestl2005swindle" />.
              Ogni specie produce una miscela unica di allomoni che imita il feromone sessuale di
              una specifica specie di impollinatore. Il labello, per forma, colore e consistenza,
              mimetizza visivamente il corpo dell&apos;insetto femmina; lo speculum — una macchia
              riflettente sul labello — simula la luce riflessa dalle ali. Solo circa il 10% dei
              fiori viene effettivamente impollinato, ma ciascuno produce fino a 12.000 semi.
            </P>
          </SubSection>
          <SubSection title="Il Problema Tassonomico">
            <P>
              Il genere <em>Ophrys</em> è uno dei gruppi tassonomicamente più controversi del regno
              vegetale<Cite id="devey2008ophrys" />. Le stime sul numero di specie variano
              drasticamente: da circa 20 (<em>Flora Europaea</em>, 1980) a oltre 130 (Delforge, 1995).
              L&apos;ibridazione frequente produce oltre 250 ibridi naturali descritti, e lo studio
              molecolare ha evidenziato copie eterogenee del DNA ribosomiale ITS in singoli individui.
              Il complesso <em>O. sphegodes</em> (le &ldquo;forretterelle&rdquo;) è uno dei gruppi
              più variabili, con almeno 18 taxa subspecifici.
            </P>
            <P>
              Il nostro dataset adotta una classificazione intermedia, trattando sei entità — <em>O. exaltata</em>,
              <em>O. garganica</em>, <em>O. incubacea</em>, <em>O. majellensis</em>,
              <em>O. sphegodes</em> e <em>O. sphegodes</em> Palena — come specie distinte,
              coerentemente con la tradizione tassonomica italiana. La penisola del Gargano costituisce
              un hotspot di diversità del complesso.
            </P>
          </SubSection>
          <Figure
            src="/figures/angular_separability.webp"
            alt="Separabilità angolare"
            caption="Separabilità angolare nello spazio degli embedding attraverso i tre paradigmi di apprendimento."
          />
        </Section>

        {/* ── 2. DATASET ── */}
        <Section id="dataset" title="Dataset">
          <P>
            Il dataset comprende immagini di sei entità tassonomiche del complesso <em>O. sphegodes</em>,
            tutte fotografate nel territorio italiano, principalmente in Puglia e sul Gargano.
            Le immagini provengono da due fonti: fotografie da campo scattate dai ricercatori
            (fonte primaria per <em>O. majellensis</em> e <em>O. sphegodes</em> Palena) e
            immagini raccolte da banche dati internet aperte per le altre specie.
          </P>

          <DataTable
            columns={[
              { key: "species", label: "Specie" },
              { key: "train", label: "Train" },
              { key: "test", label: "Test" },
            ]}
            rows={[
              { species: "O. exaltata", train: "490", test: "50" },
              { species: "O. garganica", train: "563", test: "50" },
              { species: "O. incubacea", train: "352", test: "50" },
              { species: "O. majellensis", train: "152", test: "50" },
              { species: "O. sphegodes", train: "425", test: "50" },
              { species: "O. sphegodes Palena", train: "210", test: "50" },
              { species: "O. sphegodes_Napoli (novel)", train: "0", test: "548" },
              { species: "Totale", train: "2.192", test: "848" },
            ]}
            caption="Distribuzione delle immagini per specie negli split di training e test. La specie novel è presente solo nel test set (548 immagini)."
          />

          <Callout variant="warning" title="Data Leakage">
            Nelle fasi iniziali (Exp 1–5), il dataset presentava un problema critico di data leakage:
            immagini identiche a livello di contenuto apparivano sia in training che in test con nomi
            diversi. La scoperta è avvenuta tramite hashing MD5. A partire da Exp 6, il dataset è stato
            ripulito con un protocollo in tre fasi: (1) deduplicazione MD5, (2) estrazione di un
            hold-out bilanciato di 50 immagini per classe, (3) cross-validation stratificata a 5 fold
            con seed 57. Questo clean split costituisce la base di tutti gli esperimenti.
          </Callout>

          <Callout variant="info" title="Specie Novel: O. sphegodes_Napoli">
            Una settima specie, <em>O. sphegodes_Napoli</em>, è stata volutamente esclusa dal training
            set e riservata esclusivamente al test per gli esperimenti di Novel Class Discovery e
            Open-Set Recognition (Exp 8, 10). Morfologicamente affine a <em>O. sphegodes</em> ma con
            caratteri diagnostici distinti a livello di popolazione locale, costituisce il test ideale
            per valutare la capacità del sistema di rilevare entità tassonomiche non precedentemente
            osservate.
          </Callout>

          <SubSection title="Preprocessing">
            <P>
              Le immagini grezze vengono processate da un modulo di object detection basato su Faster
              R-CNN pre-addestrato, che ritaglia automaticamente il fiore rimuovendo lo sfondo.
              Il letterbox resize preserva l&apos;aspect ratio originale con padding nero.
              Le dimensioni di input sono native per architettura: 256×512 per ResNet, 224×224 per
              ConvNeXt e DINOv2. Durante la cross-validation, le classi minoritarie vengono
              sovracampionate tramite augmentations (rotazioni, flip, color jitter, cutout).
            </P>
          </SubSection>
        </Section>

        {/* ── 3. METODOLOGIA ── */}
        <Section id="metodologia" title="Metodologia Sperimentale">
          <SubSection title="Architetture">
            <P>
              Sei architetture appartenenti a tre famiglie sono state selezionate per coprire lo stato
              dell&apos;arte della classificazione visiva. Le reti convolutive classiche
              <strong> ResNet-18</strong> (11M parametri) e <strong>ResNet-50</strong> (25M)<Cite id="he2016deep" />
              sono state scelte come baseline per efficienza computazionale. Le CNN modernizzate
              <strong>ConvNeXt-Tiny</strong> (28M) e <strong>ConvNeXt-Small</strong> (50M)<Cite id="liu2022convnet" />
              incorporano scelte progettuali dei Transformer (kernel 7×7, Layer Normalization, GELU).
              I Vision Transformer <strong>DINOv2-Small</strong> (21M, ViT-S/14) e
              <strong>DINOv2-Base</strong> (86M, ViT-B/14)<Cite id="oquab2024dinov2" /> sono
              pre-addestrati in modo auto-supervisionato su 142 milioni di immagini.
            </P>
          </SubSection>

          <SubSection title="Recipe di Addestramento">
            <DataTable
              columns={[
                { key: "model", label: "Modello" },
                { key: "input", label: "Input" },
                { key: "lr", label: "LR" },
                { key: "wd", label: "WD" },
                { key: "optim", label: "Optimizer" },
              ]}
              rows={[
                { model: "ResNet-18", input: "256×512", lr: "10⁻³", wd: "0,0", optim: "Adam" },
                { model: "ResNet-50", input: "256×512", lr: "10⁻³", wd: "0,0", optim: "Adam" },
                { model: "ConvNeXt-T", input: "224×224", lr: "3·10⁻⁴", wd: "5·10⁻²", optim: "AdamW" },
                { model: "ConvNeXt-S", input: "224×224", lr: "3·10⁻⁴", wd: "5·10⁻²", optim: "AdamW" },
                { model: "DINOv2-S", input: "224×224", lr: "2·10⁻⁴", wd: "10⁻²", optim: "AdamW" },
                { model: "DINOv2-B", input: "224×224", lr: "2·10⁻⁴", wd: "10⁻²", optim: "AdamW" },
              ]}
              caption="Recipe di addestramento per architettura."
            />
          </SubSection>

          <SubSection title="Disegno Sperimentale">
            <P>
              Gli esperimenti sono organizzati in quattro serie, ciascuna delle quali aggiunge
              complessità metodologica alla precedente, in un percorso che va dalla classificazione
              chiusa all&apos;open-world:
            </P>
            <ul className="list-disc list-inside text-muted-foreground space-y-2 text-sm mb-4">
              <li><strong>Exp 6 — Cross-Entropy Baseline:</strong> tutte le architetture vengono addestrate con loss CE per 150 epoche con early stopping (patience 15). Stabilisce la baseline di performance closed-set.</li>
              <li><strong>Exp 7 — Supervised Contrastive Learning:</strong> introduce il training in due fasi. Fase 1: backbone + projection head (128-dim) addestrati con SupConLoss (τ=0,07) per 100 epoche, validati con accuratezza 1-NN. Fase 2: backbone congelato, classificatore lineare addestrato con CE per 50 epoche<Cite id="khosla2020supervised" />.</li>
              <li><strong>Exp 8 — SpHOR:</strong> come Exp 7 ma con prototipi apprendibili L2-normalizzati sulla sfera unitaria e loss di repulsione ortogonale (λ=0,01)<Cite id="bahavan2026sphor" />. Al test, quattro rivelatori OSR (Mahalanobis<Cite id="lee2018simple" />, KNN, Energy<Cite id="liu2020energy" />, Ensemble) determinano se un campione è noto o novel.</li>
              <li><strong>Exp 10 — MGH-NCD:</strong> adatta MGCE<Cite id="zheng2025mgce" /> al contesto ultra-FG. Tre esperti di clustering a risoluzioni KNN multiple (k=5, 15, 50) armonizzati tramite divergenza JS.</li>
            </ul>
          </SubSection>

          <SubSection title="Protocollo di Validazione">
            <P>
              Tutti gli esperimenti utilizzano cross-validation stratificata a 5 fold con seed 57.
              La metrica principale per la classificazione closed-set è l&apos;F1 macro, robusta allo
              squilibrio naturale del dataset (152 immagini per <em>O. majellensis</em> contro 570
              per <em>O. garganica</em>). Per l&apos;open-set si utilizza il tasso di rilevamento
              al 10% di FPR<Cite id="bendale2015openworld" />: la soglia di rifiuto viene calibrata
              in modo che solo il 10% delle immagini novel venga erroneamente accettato, e si misura
              quanti dei campioni noti vengono correttamente riconosciuti. Un valore di 0,907 significa
              273/300 note riconosciute quando solo 5/50 novel sfuggono.
            </P>
          </SubSection>

          <Figure
            src="/figures/architecture.webp"
            alt="Architettura del sistema"
            className="max-h-[720px] w-auto"
            caption="Diagramma della pipeline di addestramento e valutazione: dalla raccolta immagini alla pubblicazione dei risultati."
          />

          <SubSection title="Pipeline Open-Set Recognition">
            <P>
              Per la valutazione OSR, il modello addestrato su 6 classi viene testato su un set
              che include <em>O. sphegodes_Napoli</em> come classe sconosciuta. La procedura si
              compone di tre fasi: (1) estrazione degli embedding dal training e test set tramite
              backbone congelato; (2) addestramento di quattro rivelatori (Mahalanobis, KNN, Energy,
              Ensemble) sui soli embedding noti; (3) calcolo del tasso di rilevamento al 10% FPR.
            </P>
          </SubSection>

          <Figure
            src="/figures/osr-flow.webp"
            alt="Pipeline OSR"
            caption="Pipeline di Open-Set Recognition: estrazione embedding, addestramento rivelatori, rilevamento novità. I rivelatori valutano la distanza/probabilità che un campione appartenga a una classe nota."
          />
        </Section>

        {/* ── 4. EXP 6: CE ── */}
        <Section id="exp6" title="Exp 6 — Baseline Cross-Entropy">
          <P>
            L&apos;Exp 6 stabilisce la baseline di classificazione closed-set con loss CE standard.
            L&apos;obiettivo softmax separa le classi con iperpiani di decisione ma non struttura
            lo spazio latente: due campioni della stessa specie possono proiettarsi in regioni
            arbitrarie dello spazio, purché il classificatore li assegni alla classe corretta.
            Questa mancanza di struttura è accettabile per la classificazione closed-set ma diventa
            critica in scenari open-world<Cite id="geng2020openset" />.
          </P>

          <DataTable
            columns={[
              { key: "model", label: "Modello" },
              { key: "cv", label: "F1 CV" },
              { key: "test", label: "F1 Test" },
              { key: "params", label: "Parametri" },
              { key: "gap", label: "Gap" },
            ]}
            rows={[
              { model: "ResNet-18", cv: "0,860", test: "0,880", params: "11M", gap: "+0,020" },
              { model: "ResNet-50", cv: "0,858", test: "0,882", params: "25M", gap: "+0,024" },
              { model: "ConvNeXt-Tiny", cv: "0,892", test: "0,890", params: "28M", gap: "−0,002" },
              { model: "ConvNeXt-Small", cv: "0,883", test: "0,863", params: "50M", gap: "−0,020" },
              { model: "DINOv2-Small", cv: "0,906", test: "0,888", params: "21M", gap: "−0,018" },
              { model: "DINOv2-Base", cv: "0,916", test: "0,927", params: "86M", gap: "+0,011" },
            ]}
            highlightRow="DINOv2-Base"
            caption="Exp 6 — F1 macro di cross-validation e test per architettura."
          />

          <Callout variant="success" title="DINOv2-Base domina la classificazione closed-set">
            Con 0,927 F1 test, DINOv2-Base è il miglior modello closed-set dell&apos;intero studio.
            Il gap CV-Test positivo (+0,011) è sorprendente per un modello da 86M parametri su un
            dataset di 2.232 immagini. Il pre-training su 142 milioni di immagini di DINOv2<Cite id="oquab2024dinov2" />
            fornisce una rappresentazione di partenza sufficientemente ricca da garantire la generalizzazione.
          </Callout>

          <P>
            Il pattern del gap CV-Test rivela il fenomeno della <em>double descent</em><Cite id="belkin2019double" />:
            le ResNet (11–25M) mostrano gap positivo, operando nel regime sovra-parametrizzato con
            la regolarizzazione implicita della batch normalization<Cite id="ioffe2015batchnorm" />.
            ConvNeXt-Small (50M, il più grande tra le CNN) mostra overfitting più pronunciato perché
            la layer normalization non fornisce la stessa regolarizzazione. DINOv2-Base, nonostante
            i 86M parametri, generalizza grazie al pre-training su larga scala.
          </P>

          <Figure
            src="/figures/loss_curves_exp6_dinov2b.webp"
            alt="Loss curves DINOv2-B Exp 6"
            caption="Curve di training e validazione per DINOv2-Base in Exp 6 (CE). Convergenza stabile intorno a 60 epoche, early stopping a 75."
          />
        </Section>

        {/* ── 5. EXP 7: SUPCON ── */}
        <Section id="exp7" title="Exp 7 — Supervised Contrastive Learning">
          <P>
            Per affrontare la limitazione strutturale della CE, si adotta l&apos;Apprendimento
            Contrastivo Supervisionato (SupCon)<Cite id="khosla2020supervised" />. La loss SupCon
            attira esplicitamente i campioni della stessa classe e respinge quelli di classi diverse
            nello spazio normalizzato L2. Il risultato è uno spazio metrico dove la distanza coseno
            riflette la similarità semantica — proprietà essenziale per l&apos;open-set recognition
            e il novel class discovery<Cite id="liu2024novel" /><Cite id="vaze2022gcd" />.
          </P>

          <DataTable
            columns={[
              { key: "model", label: "Modello" },
              { key: "cv", label: "F1 CV" },
              { key: "test", label: "F1 Test" },
              { key: "gap", label: "Gap" },
            ]}
            rows={[
              { model: "ResNet-18", cv: "0,612", test: "0,736", gap: "+0,123" },
              { model: "ResNet-50", cv: "0,736", test: "0,799", gap: "+0,063" },
              { model: "ConvNeXt-Tiny", cv: "0,876", test: "0,681", gap: "−0,195" },
              { model: "ConvNeXt-Small", cv: "0,882", test: "0,689", gap: "−0,194" },
              { model: "DINOv2-Small", cv: "0,906", test: "0,896", gap: "−0,010" },
              { model: "DINOv2-Base", cv: "0,914", test: "0,687", gap: "−0,226" },
            ]}
            highlightRow="DINOv2-Small"
            caption="Exp 7 — F1 macro di cross-validation e test."
          />

          <Callout variant="success" title="DINOv2-Small: il miglior compromesso dello studio">
            Con F1 test 0,896 e gap CV-Test trascurabile (−0,01), DINOv2-Small è il modello più
            bilanciato dell&apos;intero studio. A livello di specie, eccelle su
            <em> O. sphegodes Palena</em> (F1 = 0,990) e <em>O. majellensis</em> (F1 = 0,980).
          </Callout>

          <Callout variant="danger" title="OSR: Fallimento completo — accuratezza classi note = 0,0">
            Nonostante embedding qualitativamente migliori, il test OSR rivela una limitazione
            fondamentale: l&apos;accuratezza sulle classi note è <strong>0,0</strong> a ogni soglia.
            Tutte le immagini vengono classificate come sconosciute. L&apos;AUROC medio è 0,485,
            indistinguibile dal caso. Nello spazio normalizzato L2 di SupCon, le distribuzioni
            degli embedding &ldquo;noti&rdquo; e &ldquo;novel&rdquo; sono completamente sovrapposte:
            la loss contrastiva separa le classi tra loro ma non crea una regione per
            &ldquo;ciò che non è nessuna classe nota&rdquo;. Serve un margine angolare esplicito.
          </Callout>

          <P>
            Il pattern di overfitting è marcato: DINOv2-Base mostra il gap più ampio (−0,226),
            mentre le ResNet beneficiano della batch normalization (gap positivo). ConvNeXt ha
            gap ≈ −0,19. La t-SNE (Fig. 1b) mostra cluster marcatamente più coesi rispetto alla CE.
          </P>

          <Figure
            src="/figures/cv_test_gap.webp"
            alt="Generalization gap"
            caption="Gap di generalizzazione (F1 Test − F1 CV) in funzione dei parametri. Le architetture moderne mostrano gap negativi maggiori (overfitting); le ResNet beneficiano della regolarizzazione implicita della batch normalization."
          />

          <Figure
            src="/figures/per_species_f1.webp"
            alt="F1 per specie"
            caption="F1 per specie attraverso architetture ed esperimenti. O. sphegodes è sistematicamente la specie più difficile; O. exaltata e O. sphegodes Palena le più facili."
          />

          <Figure
            src="/figures/metric_curves_dinov2s.webp"
            alt="Metric curves DINOv2-S"
            caption="Curve di training e validazione per DINOv2-Small durante l'addestramento SupCon."
          />
        </Section>

        {/* ── 6. EXP 8: SPHOR ── */}
        <Section id="exp8" title="Exp 8 — Prototipi Sferici Ortogonali (SpHOR)">
          <P>
            SpHOR<Cite id="bahavan2026sphor" /> (CVPR 2026 Findings) introduce prototipi apprendibili
            L2-normalizzati sulla sfera unitaria — uno per classe. La loss combina SupCon con una
            loss di repulsione ortogonale che forza i prototipi a occupare direzioni distinte,
            massimizzando la copertura angolare. Al test, quattro rivelatori valutano la novità:
            distanza di Mahalanobis<Cite id="lee2018simple" /> dal prototipo più vicino, KNN (k=5),
            Energy score<Cite id="liu2020energy" />, ed Ensemble normalizzato.
          </P>

          <Callout variant="info" title="Trade-off ineludibile">
            SpHOR migliora la separabilità angolare ma consuma capacità discriminativa: il F1
            closed-set cala mediamente di 0,12 rispetto a Exp 7. La scelta dell&apos;architettura
            diventa critica: ConvNeXt-Tiny emerge come punto di equilibrio ottimale.
          </Callout>

          <DataTable
            columns={[
              { key: "model", label: "Modello" },
              { key: "f1t", label: "F1 Test" },
              { key: "f1cv", label: "F1 CV" },
              { key: "det", label: "Detection@10%" },
              { key: "detector", label: "Miglior Detector" },
            ]}
            rows={[
              { model: "ResNet-18", f1t: "0,520", f1cv: "0,398", det: "0,093", detector: "Mahalanobis" },
              { model: "ResNet-50", f1t: "0,609", f1cv: "0,736", det: "0,538", detector: "Ensemble" },
              { model: "ConvNeXt-Tiny", f1t: "0,759", f1cv: "0,794", det: "0,907", detector: "Ensemble" },
              { model: "ConvNeXt-Small", f1t: "0,668", f1cv: "0,735", det: "0,480", detector: "Energy" },
              { model: "DINOv2-Small", f1t: "0,747", f1cv: "0,906", det: "0,670", detector: "Ensemble" },
              { model: "DINOv2-Base", f1t: "0,654", f1cv: "0,870", det: "0,746", detector: "Energy" },
            ]}
            highlightRow="ConvNeXt-Tiny"
            caption="Exp 8 — F1 closed-set e Detection Rate al 10% FPR."
          />

          <Callout variant="success" title="ConvNeXt-Tiny: detection rate 0,907 — campione open-set">
            273 immagini note su 300 correttamente identificate, 45 campioni novel su 50 respinti.
            È l&apos;unica configurazione che dimostra di apprendere la differenza tra
            &ldquo;immagine nota mai vista&rdquo; e &ldquo;specie mai vista&rdquo; a granularità
            ultra-fine. Tre fattori spiegano il successo: (1) equilibrio capacità-regularizzazione
            (28M parametri — né troppo pochi come ResNet-18 né troppi come ConvNeXt-Small);
            (2) località convoluzionale dei kernel 7×7 su caratteri morfologici di 30–50 pixel;
            (3) compatibilità della layer normalization con la loss di repulsione.
          </Callout>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/spotlight_convnext_osr.webp"
              alt="Detection rate per rivelatore"
              caption="Tasso di rilevamento per rivelatore. ConvNeXt-Tiny (arancione) raggiunge 0,907 con Ensemble."
            />
            <Figure
              src="/figures/exp8_per_species.webp"
              alt="F1 per specie Exp 8"
              caption="F1 per specie in Exp 8: O. sphegodes è la più penalizzata; Palena e Exaltata le meno colpite."
            />
          </div>

          <P>
            L&apos;analisi del clustering sugli embedding SpHOR conferma che la separazione
            non supervisionata rimane aperta: l&apos;NMI al k reale (k=6) è 0,164 per ConvNeXt-Tiny
            e 0,187 per DINOv2-Base. K-means converge a k=2 (separazione binaria tra i due cluster
            principali). DBSCAN classifica il 96% dei punti come rumore.
          </P>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/osr_detection_rate.webp"
              alt="OSR detection rate Exp 8"
              caption="Detection rate al variare della soglia FPR per ogni rivelatore in Exp 8. Ensemble e KNN sono i più robusti."
            />
            <Figure
              src="/figures/clustering_comparison.webp"
              alt="Clustering comparison"
              caption="Confronto del clustering (K-means, Spectral, DBSCAN) su embedding SpHOR vs MGH-NCD. Nessun metodo recupera le sei specie."
            />
          </div>

          <Figure
            src="/figures/tsne_exp8_multimodel.webp"
            alt="t-SNE multi-modello Exp 8"
            caption="t-SNE per tre architetture in Exp 8: (a) ConvNeXt-Tiny — embedding ben separati, outlier presenti; (b) DINOv2-Small — cluster più coesi ma meno separati; (c) DINOv2-Base — struttura intermedia."
          />
        </Section>

        {/* ── 7. EXP 10: MGH-NCD ── */}
        <Section id="exp10" title="Exp 10 — Multi-Granularity NCD (MGH-NCD)">
          <P>
            Sulla base dei risultati SpHOR, si è tentato di migliorare la scoperta di classi nascoste
            adottando MGCE<Cite id="zheng2025mgce" />. Tre esperti di clustering operano a risoluzioni
            KNN multiple (k=5, 15, 50), ciascuno genera pseudo-etichette tramite InfoMap, e un memory
            bank contrastivo allinea le rappresentazioni tra le scale. L&apos;ipotesi era che la
            multi-granularità potesse catturare la struttura gerarchica della tassonomia botanica.
          </P>

          <Callout variant="danger" title="Collasso a O. sphegodes Palena">
            Il 74% delle classificazioni totali cade su Palena, indipendentemente dalla vera specie.
            <em> O. sphegodes</em> non viene riconosciuta nemmeno una volta su 50 immagini.
            Palena agisce come un attrattore nello spazio degli embedding. Il miglior F1 test è
            0,369 (DINOv2-Base), contro 0,927 in Exp 6.
          </Callout>

          <DataTable
            columns={[
              { key: "model", label: "Modello" },
              { key: "f1t", label: "F1 Test" },
              { key: "det", label: "Detect@10%" },
              { key: "nmi", label: "NMI" },
              { key: "acc", label: "ACC Clust." },
              { key: "palena", label: "Fraz. Palena" },
            ]}
            rows={[
              { model: "ResNet-18", f1t: "0,226", det: "0,257", nmi: "0,034", acc: "0,253", palena: "64%" },
              { model: "ResNet-50", f1t: "0,174", det: "0,321", nmi: "0,023", acc: "0,206", palena: "68%" },
              { model: "ConvNeXt-Tiny", f1t: "0,349", det: "0,566", nmi: "0,142", acc: "0,386", palena: "55%" },
              { model: "ConvNeXt-Small", f1t: "0,247", det: "0,536", nmi: "0,075", acc: "0,330", palena: "60%" },
              { model: "DINOv2-Small", f1t: "0,270", det: "0,391", nmi: "0,239", acc: "0,592", palena: "61%" },
              { model: "DINOv2-Base", f1t: "0,369", det: "0,688", nmi: "0,485", acc: "0,511", palena: "74%" },
            ]}
            caption="Exp 10 — MGH-NCD: metriche complete."
          />

          <Figure
            src="/figures/spotlight_exp10_collapse.webp"
            alt="Collasso Exp 10"
            caption="Collasso di Exp 10 per DINOv2-Base. (a) Matrice di confusione: la colonna Palena cattura la maggior parte delle predizioni. (b) Per ogni specie, la frazione Palena supera l'accuratezza corretta."
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/confusion_matrices.webp"
              alt="Confusion matrices"
              caption="Matrici di confusione per Exp 7 (SupCon), Exp 8 (SpHOR) e Exp 10 (MGH-NCD). Il degrado progressivo è evidente: da una diagonale pulita (SupCon) al collasso su Palena (MGH-NCD)."
            />
            <Figure
              src="/figures/tsne_novel_exp10.webp"
              alt="t-SNE with novel Exp 10"
              caption="Proiezione t-SNE degli embedding di Exp 10 con la classe novel O. sphegodes_Napoli (cerchi rossi). La novel si sovrappone completamente alle classi note."
            />
          </div>

          <Figure
            src="/figures/exp10_per_species.webp"
            alt="F1 per specie Exp 10"
            caption="F1 per specie in Exp 10. Nessuna specie raggiunge 0,5 F1. Il collasso è uniforme: tutte le specie soffrono allo stesso modo."
          />

          <SubSection title="Analisi del Fallimento">
            <P>
              Il fallimento si articola su tre livelli. (1) <strong>Consensus spuria</strong>:
              l&apos;allineamento degli esperti forza il consenso tra tre risoluzioni KNN; in
              ultra-FGVC, dove la risoluzione &ldquo;grossolana&rdquo; (k=50) cattura già solo
              differenze finissime, il consenso collassa sulla classe più distintiva (Palena).
              (2) <strong>Full FT impatta negativamente</strong>: a differenza di Exp 8, dove
              LoRA regolarizzava, in Exp 10 la maggiore capacità ha amplificato il collasso.
              (3) <strong>Complessità eccessiva</strong>: tre esperti, memory bank, tre loss per
              2.232 immagini — il modello &ldquo;spende&rdquo; la sua capacità per soddisfare i
              vincoli di allineamento.
            </P>
          </SubSection>

          <Callout variant="info" title="Lezione: la semplicità supera la complessità">
            SpHOR, con un singolo obiettivo contrastivo e prototipi ortogonali, supera MGCE
            nonostante la minore sofisticazione. La scelta dell&apos;approccio deve essere guidata
            dalla scala intrinseca del problema, non dalla sofisticazione della tecnica. Questo è
            un risultato scientifico valido: dimostra che l&apos;aggiunta di complessità non porta
            necessariamente a miglioramenti in ultra-FGVC.
          </Callout>
        </Section>

        {/* ── 8. BENCHMARK ── */}
        <Section id="benchmark" title="Benchmark Consolidato">
          <P>
            La tabella seguente consolida le prestazioni di tutte le architetture attraverso i
            quattro esperimenti, per classificazione closed-set (F1 test) e open-set (Detection
            Rate al 10% FPR).
          </P>

          <DataTable
            columns={[
              { key: "model", label: "Modello" },
              { key: "e6", label: "Exp 6 (CE)" },
              { key: "e7", label: "Exp 7 (SupCon)" },
              { key: "e8", label: "Exp 8 (SpHOR)" },
              { key: "e10", label: "Exp 10 (MGH)" },
              { key: "osr8", label: "OSR@10% (8)" },
              { key: "osr10", label: "OSR@10% (10)" },
            ]}
            rows={[
              { model: "ResNet-18", e6: "0,880", e7: "0,736", e8: "0,520", e10: "0,226", osr8: "0,093", osr10: "0,257" },
              { model: "ResNet-50", e6: "0,882", e7: "0,799", e8: "0,609", e10: "0,174", osr8: "0,538", osr10: "0,321" },
              { model: "ConvNeXt-Tiny", e6: "0,890", e7: "0,681", e8: "0,759", e10: "0,349", osr8: "0,907", osr10: "0,566" },
              { model: "ConvNeXt-Small", e6: "0,863", e7: "0,689", e8: "0,668", e10: "0,247", osr8: "0,480", osr10: "0,536" },
              { model: "DINOv2-Small", e6: "0,888", e7: "0,896", e8: "0,747", e10: "0,270", osr8: "0,670", osr10: "0,391" },
              { model: "DINOv2-Base", e6: "0,927", e7: "0,687", e8: "0,654", e10: "0,369", osr8: "0,746", osr10: "0,688" },
            ]}
            bests={{ e6: "DINOv2-Base", e7: "DINOv2-Small", osr8: "ConvNeXt-Tiny" }}
            caption="Benchmark completo: F1 test e Detection Rate al 10% FPR. ★ = miglior valore per colonna."
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <Figure
              src="/figures/benchmark_heatmap.webp"
              alt="Benchmark heatmap"
              caption="Heatmap: F1 test per modello ed esperimento."
            />
            <Figure
              src="/figures/closed_vs_open_scatter.webp"
              alt="Closed vs open"
              caption="F1 test vs Detection Rate: il trade-off closed-set / open-set."
            />
          </div>

          <SubSection title="Risultati Chiave">
            <ol className="list-decimal list-inside text-muted-foreground space-y-2">
              <li><strong>Campione closed-set:</strong> DINOv2-Small + SupCon (F1 test 0,896, gap CV-Test −0,01) — il miglior compromesso tra accuratezza e generalizzazione.</li>
              <li><strong>Campione open-set:</strong> ConvNeXt-Tiny + SpHOR (detection rate 0,907) — prima dimostrazione di OSR funzionante in ultra-FGVC.</li>
              <li><strong>Compromesso SpHOR:</strong> calo medio F1 closed-set di 0,12, ma detection rate da 0,0 (SupCon) a 0,907 (SpHOR). Trade-off accettabile per ambienti aperti.</li>
              <li><strong>Fallimento MGH-NCD:</strong> collasso a Palena. La multi-granularità è controproducente quando la granularità intrinseca è già al massimo.</li>
              <li><strong>Risultato negativo:</strong> la scoperta non supervisionata di nuove specie in ultra-FGVC non è risolvibile con i metodi attuali (NMI = 0,16–0,19).</li>
            </ol>
          </SubSection>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Figure
              src="/figures/benchmark_bar.webp"
              alt="Benchmark bar chart"
              caption="Confronto a barre dell'F1 test attraverso esperimenti e architetture."
            />
            <Figure
              src="/figures/summary_dual_heatmap.webp"
              alt="Dual heatmap"
              caption="Heatmap duale: F1 test (sinistra) e Detection Rate (destra) affiancati per confronto immediato."
            />
          </div>

          <Figure
            src="/figures/osr_detection_comparison.webp"
            alt="OSR detection comparison"
            caption="Detection rate al 10% FPR per tutti i rivelatori: Exp 8 (sinistra) vs Exp 10 (destra)."
          />

          <Figure
            src="/figures/spotlight_species_evolution.webp"
            alt="Species evolution"
            caption="Evoluzione dell'F1 per specie attraverso i quattro esperimenti. Il degrado è generalizzato, ma alcune specie (O. sphegodes, O. garganica) soffrono più di altre."
          />

          <Figure
            src="/figures/loss_curves_aggregated.webp"
            alt="Loss curves aggregated"
            caption="Curve di loss aggregate per esperimento. La convergenza è più rapida in Exp 6 (CE) rispetto a Exp 7–10 (contrastive)."
          />
        </Section>

        {/* ── 9. CONCLUSIONI ── */}
        <Section id="conclusioni" title="Conclusioni">
          <P>
            Questa tesi ha presentato uno studio sistematico della classificazione ultra-fine-grained
            del genere <em>Ophrys</em> mediante deep learning, confrontando sei architetture su tre
            paradigmi di apprendimento con un protocollo rigoroso basato su clean split MD5 e
            cross-validation stratificata.
          </P>

          <SubSection title="Contributi">
            <ul className="list-disc list-inside text-muted-foreground space-y-1 text-sm">
              <li>Primo benchmark sistematico di architetture deep learning per la classificazione ultra-FGVC delle <em>Ophrys</em>.</li>
              <li>Dimostrazione che SpHOR + ConvNeXt-Tiny risolve l&apos;OSR in ultra-FGVC (detection rate 0,907).</li>
              <li>Risultato negativo pubblicabile: NCD in ultra-FGVC rimane un problema aperto.</li>
              <li>Dataset curato e deduplicato per sei specie di <em>Ophrys</em>.</li>
            </ul>
          </SubSection>

          <SubSection title="Limiti">
            <ul className="list-disc list-inside text-muted-foreground space-y-1 text-sm">
              <li>Copertura geografica ristretta (Puglia e Gargano).</li>
              <li>Squilibrio tra fonti: specie da campo vs internet.</li>
              <li>Risorse computazionali limitate (batch size 8, LoRA vs full FT discontinuo).</li>
              <li>Assenza di validazione esterna con botanici o DNA barcoding.</li>
            </ul>
          </SubSection>

          <SubSection title="Lavori Futuri">
            <ul className="list-disc list-inside text-muted-foreground space-y-1 text-sm">
              <li>Espansione del dataset tramite citizen science e generative AI.</li>
              <li>Metodi part-based (APL<Cite id="dai2025apl" />, PartGCD<Cite id="wang2025partgcd" />) per NCD.</li>
              <li>Integrazione sistematica di XAI con validazione da parte di esperti botanici.</li>
              <li>Validazione sul campo tramite deploy dell&apos;applicazione OrchID.</li>
            </ul>
          </SubSection>
        </Section>

        {/* ── 10. BIBLIOGRAFIA ── */}
        <Section id="bibliografia" title="Bibliografia">
          <div className="space-y-3 text-sm">
            {PAPERS.map((p) => (
              <div key={p.id} className="flex gap-2">
                <span className="text-primary font-mono text-xs shrink-0 mt-0.5">[{p.id}]</span>
                <div>
                  <span className="font-medium text-foreground">{p.authors} ({p.year}).</span>{" "}
                  <em>{p.title}.</em>{" "}
                  {p.venue && <span>{p.venue}. </span>}
                  {p.url && (
                    <a href={p.url} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline text-xs">
                      [Link]
                    </a>
                  )}
                  {p.doi && (
                    <a href={`https://doi.org/${p.doi}`} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline text-xs">
                      [DOI]
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Section>

      </main>
    </div>
    </CiteProvider>
  );
}
