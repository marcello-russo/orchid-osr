import { useTranslation } from "react-i18next";
import {
  BarChart3,
  GraduationCap,
  Target,
  FlaskConical,
  Cpu,
  Zap,
  ShieldCheck,
  Binary,
  Microscope,
  BookOpen,
  ArrowRight,
  BookMarked,
  Route,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const REFERENCES = [
  {
    id: 1,
    citation:
      "Paulus, H.F. & Gack, C. (1990). Pollinators as prepotinating mating factors in Ophrys. Israel Journal of Botany, 39, 43–79.",
    url: "https://www.researchgate.net/publication/280605510_Paulus_Gack_1990_Pollinators_as_prepollinating_isolation_factors_Evolution_and_speciation_in_Ophrys_-_Israel_J_Bot_39_43-79",
  },
  {
    id: 2,
    citation:
      "Devey, D.S., Bateman, R.M., Fay, M.F. & Hawkins, J.A. (2008). Friends or relatives? Phylogenetics and species delimitation in the controversial European orchid genus Ophrys. Annals of Botany, 101(3), 385–403.",
    url: "https://doi.org/10.1093/aob/mcm302",
  },
  {
    id: 3,
    citation:
      "Bateman, R.M., Sramkó, G. & Paun, O. (2018). Integrating phylogenomics, population genetics and species distribution modelling to resolve the taxonomy of the Ophrys sphegodes complex. Botanical Journal of the Linnean Society, 186(3), 267–291.",
    url: "https://doi.org/10.1093/botlinnean/box094",
  },
  {
    id: 4,
    citation:
      "He, K., Zhang, X., Ren, S. & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.",
    url: "https://doi.org/10.1109/CVPR.2016.90",
  },
  {
    id: 5,
    citation:
      "Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T. & Xie, S. (2022). A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11976–11986.",
    url: "https://doi.org/10.1109/CVPR52688.2022.01167",
  },
  {
    id: 6,
    citation:
      "Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Xu, S., Edunov, V., Jegou, H., Mairal, J., Labatut, P. & Joulin, A. (2024). DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research.",
    url: "https://arxiv.org/abs/2304.07193",
  },
  {
    id: 7,
    citation:
      "Vaze, S., Han, K., Vedaldi, A. & Zisserman, A. (2022). Generalized Category Discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 7492–7501.",
    url: "https://doi.org/10.1109/CVPR52688.2022.00737",
  },
  {
    id: 8,
    citation:
      "Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C. & Krishnan, D. (2020). Supervised Contrastive Learning. In Advances in Neural Information Processing Systems (NeurIPS), 33, 18661–18673.",
    url: "https://papers.nips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html",
  },
  {
    id: 9,
    citation:
      "Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR).",
    url: "https://openreview.net/forum?id=nZeVKeeFYf9",
  },
  {
    id: 10,
    citation:
      "Kingma, D.P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations (ICLR).",
    url: "https://arxiv.org/abs/1412.6980",
  },
  {
    id: 11,
    citation:
      "Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. In International Conference on Learning Representations (ICLR).",
    url: "https://openreview.net/forum?id=Bkg6RiCqY7",
  },
  {
    id: 12,
    citation:
      "Prechelt, L. (1998). Early Stopping — But When? In Neural Networks: Tricks of the Trade, Lecture Notes in Computer Science, vol 1524. Springer, 55–69.",
    url: "https://doi.org/10.1007/3-540-49430-8_3",
  },
];

function renderCitations(text: string) {
  const parts = text.split(/\[(\d+)\]/g);
  return parts.map((part, i) => {
    const num = parseInt(part, 10);
    if (!isNaN(num) && part === String(num)) {
      return (
        <a
          key={i}
          href={`#ref-${num}`}
          className="text-primary text-xs font-bold align-super no-underline hover:underline cursor-pointer"
        >
          [{num}]
        </a>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export function AboutPage() {
  const { t } = useTranslation();

  return (
    <div className="space-y-12 md:space-y-20 animate-in fade-in duration-700 pb-20 md:pb-32 max-w-7xl mx-auto px-4 md:px-12">
      {/* 1. Hero */}
      <div className="relative overflow-hidden rounded-2xl md:rounded-[2.5rem] bg-linear-to-br from-primary/20 via-primary/5 to-background p-6 md:p-16 border border-primary/10 shadow-2xl shadow-primary/5">
        <div className="relative z-10 max-w-4xl">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold mb-8 tracking-widest uppercase">
            <GraduationCap className="size-4" /> {t("about.hero.tag")}
          </div>
          <h1 className="text-4xl md:text-7xl font-black text-primary tracking-tighter mb-4 md:mb-6 leading-none">
            {t("about.hero.title")}{" "}
            <span className="text-foreground/90">{t("about.hero.subtitle")}</span>
          </h1>
          <p className="text-base md:text-xl text-muted-foreground leading-relaxed font-light">
            {t("about.hero.description")}
          </p>
        </div>
        <div className="absolute top-0 right-0 -translate-y-1/2 translate-x-1/4 size-[40rem] rounded-full bg-primary/5 blur-[120px]" />
      </div>

      {/* 2. Research Objectives */}
      <section className="space-y-6 md:space-y-8">
        <div className="flex items-center gap-3 md:gap-4">
          <div className="size-10 md:size-12 rounded-2xl bg-blue-500/10 flex items-center justify-center text-blue-500">
            <Target className="size-5 md:size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{t("about.objectives.title")}</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 text-muted-foreground leading-relaxed text-base">
          <div className="space-y-4">
            <h3 className="text-foreground font-bold text-lg flex items-center gap-2">
              <Binary className="size-5 text-primary" /> {t("about.objectives.vectorSpace.title")}
            </h3>
            <p>{renderCitations(t("about.objectives.vectorSpace.description"))}</p>
          </div>
          <div className="space-y-4">
            <h3 className="text-foreground font-bold text-lg flex items-center gap-2">
              <Microscope className="size-5 text-primary" /> {t("about.objectives.ncd.title")}
            </h3>
            <p>{renderCitations(t("about.objectives.ncd.description"))}</p>
          </div>
        </div>
      </section>

      {/* 3. Scientific Context */}
      <section className="space-y-4 md:space-y-6">
        <div className="flex items-center gap-3 md:gap-4">
          <div className="size-10 md:size-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary">
            <BookOpen className="size-5 md:size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{t("about.scientific.title")}</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 text-muted-foreground leading-relaxed text-base">
          <p>{renderCitations(t("about.scientific.content1"))}</p>
          <p>{renderCitations(t("about.scientific.content2"))}</p>
        </div>
      </section>

      {/* 3.5 Experimental Journey */}
      <section className="space-y-6 md:space-y-8">
        <div className="flex items-center gap-3 md:gap-4">
          <div className="size-10 md:size-12 rounded-2xl bg-amber-500/10 flex items-center justify-center text-amber-500">
            <Route className="size-5 md:size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{t("about.journey.title")}</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 text-muted-foreground leading-relaxed text-base">
          <div className="space-y-4">
            <h3 className="text-foreground font-bold text-lg">{t("about.journey.baselines_title")}</h3>
            <p>{t("about.journey.baselines")}</p>
            <h3 className="text-foreground font-bold text-lg pt-2">{t("about.journey.leakage_title")}</h3>
            <p>{t("about.journey.leakage")}</p>
          </div>
          <div className="space-y-4">
            <h3 className="text-foreground font-bold text-lg">{t("about.journey.preprocessing_title")}</h3>
            <p>{t("about.journey.preprocessing")}</p>
            <h3 className="text-foreground font-bold text-lg pt-2">{t("about.journey.clean_title")}</h3>
            <p>{t("about.journey.clean")}</p>
          </div>
        </div>
      </section>

      {/* 4. Methodology */}
      <section className="space-y-8 p-5 md:p-14 rounded-2xl md:rounded-[3rem] bg-muted/20 border border-border/50">
        <div className="space-y-4 max-w-3xl">
          <div className="flex items-center gap-3 md:gap-4">
            <div className="size-10 md:size-12 rounded-2xl bg-purple-500/10 flex items-center justify-center text-purple-500">
              <FlaskConical className="size-5 md:size-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{t("about.methodology.title")}</h2>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-8">
          <FeatureCard
            icon={<Cpu className="size-5" />}
            title={t("about.methodology.model_selection.title")}
            description={t("about.methodology.model_selection.description")}
          />
          <FeatureCard
            icon={<Zap className="size-5" />}
            title={t("about.methodology.optimization.title")}
            description={t("about.methodology.optimization.description")}
          />
          <FeatureCard
            icon={<ShieldCheck className="size-5" />}
            title={t("about.methodology.rigor.title")}
            description={t("about.methodology.rigor.description")}
          />
        </div>
      </section>

      {/* 5. Results */}
      <section className="space-y-6 md:space-y-8">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-green-500/10 flex items-center justify-center text-green-500">
            <BarChart3 className="size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{t("about.experiment6.title")}</h2>
        </div>
        <div className="bg-muted/30 border border-border rounded-2xl md:rounded-[2.5rem] p-5 md:p-12 overflow-hidden relative">
          <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-12">
            <div className="space-y-6">
              <p className="text-muted-foreground leading-relaxed text-base">
                {t("about.experiment6.description")}
              </p>
              <div className="space-y-3 md:space-y-4">
                <div className="flex items-start gap-3 p-4 md:p-5 rounded-xl md:rounded-2xl bg-background/50 border border-border/50">
                  <div className="text-primary font-bold text-xl md:text-2xl">0.91</div>
                  <div className="text-sm text-muted-foreground">
                    <span className="text-foreground font-bold block mb-1 text-base">
                      {t("about.experiment6.vit")}
                    </span>
                    {renderCitations(t("about.experiment6.vitDesc"))}
                  </div>
                </div>
                <div className="flex items-start gap-3 p-4 md:p-5 rounded-xl md:rounded-2xl bg-background/50 border border-border/50">
                  <div className="text-primary font-bold text-xl md:text-2xl">0.90</div>
                  <div className="text-sm text-muted-foreground">
                    <span className="text-foreground font-bold block mb-1 text-base">
                      {t("about.experiment6.resnet")}
                    </span>
                    {t("about.experiment6.resnetDesc")}
                  </div>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-center bg-background/40 rounded-xl md:rounded-3xl border border-border/50 p-5 md:p-8">
              <div className="text-center space-y-3">
                <div className="text-xs text-muted-foreground uppercase tracking-widest font-bold">
                  {t("about.experiment6.badge_model")}
                </div>
                <div className="text-5xl font-black text-primary uppercase">DINOv2-Base</div>
                <div className="text-xs text-muted-foreground">
                  {t("about.experiment6.badge_folds")}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 6. Next Steps */}
      <section className="space-y-6 md:space-y-8">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-orange-500/10 flex items-center justify-center text-orange-500">
            <ArrowRight className="size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{t("about.future.title")}</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8 text-muted-foreground text-base leading-relaxed">
          <p>{renderCitations(t("about.future.phase2"))}</p>
          <p>{renderCitations(t("about.future.phase3"))}</p>
        </div>
      </section>

      {/* 7. Bibliography */}
      <section className="space-y-6">
        <div className="flex items-center gap-3 md:gap-4">
          <div className="size-10 md:size-12 rounded-2xl bg-muted flex items-center justify-center text-muted-foreground">
            <BookMarked className="size-5 md:size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{t("about.bibliography.title")}</h2>
        </div>
        <ol className="space-y-3 list-decimal list-inside text-sm text-muted-foreground leading-relaxed">
          {REFERENCES.map((ref) => (
            <li key={ref.id} id={`ref-${ref.id}`} className="scroll-mt-8">
              <a
                href={ref.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground hover:text-primary transition-colors"
              >
                {ref.citation}
              </a>
            </li>
          ))}
        </ol>
      </section>

      {/* 8. Footer */}
      <footer className="pt-12 md:pt-20 border-t border-border text-center space-y-6 md:space-y-8">
        <div className="flex flex-col md:flex-row justify-center gap-4">
          <div className="px-6 py-3 rounded-full bg-muted/50 border border-border text-xs font-bold uppercase tracking-wider">
            {t("about.footer.university")}
          </div>
          <div className="px-6 py-3 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold uppercase tracking-widest">
            {t("about.footer.degree")}
          </div>
        </div>
        <div className="max-w-2xl mx-auto space-y-4">
          <p className="text-muted-foreground text-base leading-relaxed">
            {t("about.footer.description")}{" "}
            <strong>{t("about.footer.department")}</strong>.
          </p>
          <div className="text-xs text-muted-foreground uppercase tracking-widest font-bold">
            {t("about.footer.author")}
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <Card className="shadow-none border-border/50 bg-background/50 hover:border-primary/30 transition-all hover:-translate-y-1">
      <CardHeader className="pb-3 text-left">
        <div className="size-10 rounded-lg bg-primary/5 flex items-center justify-center text-primary mb-2">
          {icon}
        </div>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-left">
        <p className="text-sm text-muted-foreground leading-relaxed">
          {renderCitations(description)}
        </p>
      </CardContent>
    </Card>
  );
}
