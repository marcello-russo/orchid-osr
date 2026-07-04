import { useState, useCallback } from "react";
import { useTranslation } from "react-i18next";
import {
  FlaskConical,
  BarChart3,
  ChevronDown,
  ChevronRight,
  Target,
  Binary,
  BookOpen,
  Route,
  Zap,
  BookMarked,
  AlertTriangle,
  Rocket,
  GraduationCap,
  Cpu,
  ShieldCheck,
  Microscope,
  Layers,
  Brain,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useQuery } from "@tanstack/react-query";
import { fetchApi, fetchNcdComparison } from "@/lib/api";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { foldToEpochPoints } from "@/features/explorer/types";
import { useChartColors } from "@/features/explorer/hooks/useChartColors";
import type { ModelComparisonEntry, FoldData, NcdComparison, NcdComparisonEntry } from "@/lib/types";

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
  {
    id: 13,
    citation:
      "van den Oord, A., Li, Y. & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. arXiv preprint arXiv:1807.03748.",
    url: "https://arxiv.org/abs/1807.03748",
  },
  {
    id: 14,
    citation:
      "Bendale, A. & Boult, T. (2015). Towards Open World Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1893–1902.",
    url: "https://doi.org/10.1109/CVPR.2015.7298799",
  },
  {
    id: 15,
    citation:
      "Geng, C., Huang, S. & Chen, S. (2020). Recent Advances in Open Set Recognition: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(10), 3614–3631.",
    url: "https://doi.org/10.1109/TPAMI.2020.2981604",
  },
  {
    id: 16,
    citation:
      "Bahavan, N., Seneviratne, S. & Halgamuge, S. (2026). SphOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Learning Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings Track.",
    url: "https://arxiv.org/abs/2503.08049",
  },
  {
    id: 17,
    citation:
      "Zheng, H., Pu, N., Li, W., Sebe, N. & Zhong, Z. (2025). Generalized Fine-Grained Category Discovery with Multi-Granularity Conceptual Experts. arXiv preprint arXiv:2509.26227.",
    url: "https://arxiv.org/abs/2509.26227",
  },
  {
    id: 18,
    citation:
      "Belkin, M., Hsu, D., Ma, S. & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias-variance trade-off. Proceedings of the National Academy of Sciences, 116(32), 15849–15854.",
    url: "https://doi.org/10.1073/pnas.1903070116",
  },
  {
    id: 19,
    citation:
      "Ioffe, S. & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In International Conference on Machine Learning (ICML), 448–456.",
    url: "https://arxiv.org/abs/1502.03167",
  },
  {
    id: 20,
    citation:
      "Liu, W., Wang, X., Owens, J. & Li, Y. (2020). Energy-based Out-of-distribution Detection. In Advances in Neural Information Processing Systems (NeurIPS), 33, 21480–21491.",
    url: "https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html",
  },
  {
    id: 21,
    citation:
      "Lee, K., Lee, K., Lee, H. & Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. In Advances in Neural Information Processing Systems (NeurIPS), 31.",
    url: "https://arxiv.org/abs/1807.03888",
  },
  {
    id: 22,
    citation:
      "Chen, Y., Lang, N., Schmidt, B.C., Jain, A., Basset, Y., Beery, S., Larrivée, M. & Rolnick, D. (2025). Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring. In NeurIPS 2025 Dataset and Benchmark Track (Spotlight).",
    url: "https://arxiv.org/abs/2503.01691",
  },
  {
    id: 23,
    citation:
      "Bao, W., Yu, Q. & Kong, Y. (2023). Latent Space Energy-based Model for Fine-grained Open Set Recognition. arXiv preprint arXiv:2309.10711.",
    url: "https://arxiv.org/abs/2309.10711",
  },
  {
    id: 24,
    citation:
      "Kareinen, J., Skyttä, A., Eerola, T., Kraft, K., Lensu, L., Suikkanen, S., Lehtiniemi, M. & Kälviäinen, H. (2025). Open-Set Plankton Recognition. In ECCV 2024 OOD-CV Workshop. arXiv preprint arXiv:2503.11318.",
    url: "https://arxiv.org/abs/2503.11318",
  },
  {
    id: 25,
    citation:
      "Sun, Y., Ming, Y., Zhu, X. & Li, Y. (2022). Out-of-Distribution Detection with Deep Nearest Neighbors. In Proceedings of the International Conference on Machine Learning (ICML), 20827–20840.",
    url: "https://proceedings.mlr.press/v162/sun22d.html",
  },
  {
    id: 26,
    citation:
      "Shah, D. (2026). Beyond Toy Benchmarks: A Systematic Evaluation of OOD Detection Methods For Plant Pathology Classification. arXiv preprint arXiv:2605.08618.",
    url: "https://arxiv.org/abs/2605.08618",
  },
  {
    id: 27,
    citation:
      "Low, C.Y., Koo, H., Park, J. & Cha, M. (2025). Dual-Margin Embedding for Fine-Grained Long-Tailed Plant Taxonomy. arXiv preprint arXiv:2512.18994.",
    url: "https://arxiv.org/abs/2512.18994",
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

export function ResearchPage() {
  const { t } = useTranslation();

  const expQuery = useQuery({
    queryKey: ["experiments"],
    queryFn: () => fetchApi<{ exp6: ModelComparisonEntry[]; exp7: ModelComparisonEntry[]; exp8: ModelComparisonEntry[] }>("/api/comparison/experiments"),
    staleTime: 5 * 60 * 1000,
  });

  return (
    <div className="space-y-12 md:space-y-20 animate-in fade-in duration-700 pb-20 md:pb-32 max-w-7xl mx-auto px-4 md:px-12">
      {/* 1. Hero */}
      <div className="relative overflow-hidden rounded-2xl md:rounded-[2.5rem] bg-linear-to-br from-primary/20 via-primary/5 to-background p-6 md:p-16 border border-primary/10 shadow-2xl shadow-primary/5">
        <div className="relative z-10 max-w-4xl">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold mb-8 tracking-widest uppercase">
            <GraduationCap className="size-4" /> {t("about.hero.tag")}
          </div>
          <h1 className="text-4xl md:text-7xl font-black text-primary tracking-tighter mb-4 md:mb-6 leading-none">
            {t("about.hero.title")}
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

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-8">
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
          <FeatureCard
            icon={<Layers className="size-5" />}
            title={t("about.methodology.augmentation.title")}
            description={t("about.methodology.augmentation.description")}
          />
        </div>
      </section>

      {/* 5 ============================================================
            EXP 6: CROSS-ENTROPY BASELINE
      ============================================================ */}
      <section className="space-y-6 md:space-y-8 pt-4">
        <div className="flex items-center gap-3 md:gap-4">
          <div className="size-10 md:size-12 rounded-2xl bg-green-500/10 flex items-center justify-center text-green-500">
            <BarChart3 className="size-5 md:size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">Exp 6 — Cross-Entropy Baseline</h2>
        </div>
        <p className="text-muted-foreground leading-relaxed">{t("about.exp6.description")}</p>
        {expQuery.isLoading ? (
          <div className="flex justify-center py-8"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>
        ) : expQuery.error ? (
          <p className="text-sm text-muted-foreground text-center py-8">{t("about.results.unavailable")}</p>
        ) : (
          <ExpTable exp={expQuery.data?.exp6 ?? []} />
        )}
      </section>

      {/* 6 ============================================================
            EXP 7: SUPERVISED CONTRASTIVE LEARNING
      ============================================================ */}
      <section className="space-y-6 md:space-y-8 pt-4">
        <div className="flex items-center gap-3 md:gap-4">
          <div className="size-10 md:size-12 rounded-2xl bg-orange-500/10 flex items-center justify-center text-orange-500">
            <Brain className="size-5 md:size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">Exp 7 — Supervised Contrastive Learning</h2>
        </div>

        <div className="p-6 rounded-xl bg-amber-500/5 border border-amber-500/20 text-muted-foreground leading-relaxed">
          <p className="font-bold text-foreground mb-2">{t("about.exp7.motivation_title")}</p>
          <p>{renderCitations(t("about.exp7.motivation"))}</p>
        </div>

        {expQuery.isLoading ? (
          <div className="flex justify-center py-8"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>
        ) : expQuery.error ? (
          <p className="text-sm text-muted-foreground text-center py-8">{t("about.results.unavailable")}</p>
        ) : (
          <ExpTable exp={expQuery.data?.exp7 ?? []} />
        )}

        <div className="p-6 md:p-8 rounded-2xl bg-gradient-to-br from-amber-500/10 to-amber-500/5 border border-amber-500/20">
          <div className="flex items-center gap-3 mb-4">
            <Cpu className="size-6 text-amber-500" />
            <h3 className="text-xl font-bold">{t("about.exp7.winner_title")}</h3>
          </div>
          <p className="text-muted-foreground leading-relaxed">{t("about.exp7.winner")}</p>
        </div>

        <h3 className="text-lg font-bold flex items-center gap-2">
          <ShieldCheck className="size-5 text-red-500" />
          {t("about.exp7.overfitting_title")}
        </h3>
        <p className="text-muted-foreground leading-relaxed">{renderCitations(t("about.exp7.overfitting"))}</p>

        <h3 className="text-lg font-bold flex items-center gap-2">
          <Microscope className="size-5 text-blue-500" />
          {t("about.exp7.species_title")}
        </h3>
        <p className="text-muted-foreground leading-relaxed">{renderCitations(t("about.exp7.species"))}</p>

        <h3 className="text-lg font-bold flex items-center gap-2">
          <Layers className="size-5 text-purple-500" />
          {t("about.exp7.arch_title")}
        </h3>
        <p className="text-muted-foreground leading-relaxed">{renderCitations(t("about.exp7.arch"))}</p>
      </section>

      {/* 7 ============================================================
            EXP 8: OPEN-WORLD RECOGNITION
      ============================================================ */}
      <section className="space-y-6 md:space-y-8 pt-4">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-violet-500/10 flex items-center justify-center text-violet-500">
            <Brain className="size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">Exp 8 — Open-World Recognition</h2>
        </div>

        <p className="text-muted-foreground leading-relaxed">{renderCitations(t("about.future.phase3"))}</p>

        {expQuery.isLoading ? (
          <div className="flex justify-center py-8"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>
        ) : expQuery.error ? (
          <p className="text-sm text-muted-foreground text-center py-8">{t("about.results.unavailable")}</p>
        ) : (
          <ExpTable exp={expQuery.data?.exp8 ?? []} />
        )}

        <div className="p-6 rounded-xl bg-muted/20 border border-border/50 text-muted-foreground leading-relaxed">
          <p className="font-bold text-foreground mb-2">{t("about.future.phase1_embedding_title")}</p>
          <p className="text-sm">{renderCitations(t("about.future.phase1_embedding"))}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8 mt-6">
          <div className="space-y-4 p-6 rounded-xl bg-blue-500/5 border border-blue-500/20">
            <h3 className="text-foreground font-bold text-base flex items-center gap-2">
              <ShieldCheck className="size-4 text-blue-500" />
              {t("about.future.osr_title")}
            </h3>
            <p className="text-sm text-muted-foreground">{renderCitations(t("about.future.osr_description"))}</p>
          </div>
          <div className="space-y-4 p-6 rounded-xl bg-violet-500/5 border border-violet-500/20">
            <h3 className="text-foreground font-bold text-base flex items-center gap-2">
              <Binary className="size-4 text-violet-500" />
              {t("about.future.gcd_title")}
            </h3>
            <p className="text-sm text-muted-foreground">{renderCitations(t("about.future.gcd_description"))}</p>
          </div>
        </div>

        <div className="p-6 rounded-xl bg-amber-500/5 border border-amber-500/20 text-muted-foreground leading-relaxed">
          <p className="font-bold text-foreground mb-2">{t("about.future.phase3_incremental_title")}</p>
          <p className="text-sm">{t("about.future.phase3_incremental")}</p>
        </div>

        <div className="p-6 rounded-xl bg-primary/5 border border-primary/10 text-muted-foreground leading-relaxed">
          <p className="font-bold text-foreground mb-2">{t("about.future.phase3_ncd_title")}</p>
          <p className="text-sm">{renderCitations(t("about.future.phase3_ncd"))}</p>
        </div>

        <NcdResults experiment="Exp 8 (SpHOR)" />
      </section>

      {/* 8 ============================================================
            EXP 9: NOVEL CLASS DISCOVERY (hidden O. majellensis)
      ============================================================ */}
      <section className="space-y-6 md:space-y-8 pt-4">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-red-500/10 flex items-center justify-center text-red-500">
            <FlaskConical className="size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">Exp 9 — Novel Class Discovery</h2>
        </div>

        <p className="text-muted-foreground leading-relaxed">
          {t("about.exp9.description")}
        </p>

        <Exp9NcdResults />
      </section>

      {/* 9 ============================================================
            CROSS-EXPERIMENT ANALYSIS
      ============================================================ */}
      <section className="space-y-6 md:space-y-8 pt-4">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-red-500/10 flex items-center justify-center text-red-500">
            <AlertTriangle className="size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{t("about.analysis.title")}</h2>
        </div>

        <p className="text-muted-foreground leading-relaxed">
          {t("about.analysis.description")}
        </p>

        <div className="p-6 rounded-xl bg-red-500/5 border border-red-500/20 text-muted-foreground leading-relaxed">
          <p className="font-bold text-foreground mb-2">{t("about.analysis.key_finding")}</p>
          <p className="text-sm">
            {t("about.analysis.key_finding_detail")}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4 p-6 rounded-xl bg-blue-500/5 border border-blue-500/20">
            <h3 className="text-foreground font-bold text-base">{t("about.analysis.confusion_title")}</h3>
            <div className="text-sm text-muted-foreground space-y-2">
              <p><strong>1.</strong> {t("about.analysis.confusion_1")}</p>
              <p><strong>2.</strong> {t("about.analysis.confusion_2")}</p>
              <p><strong>3.</strong> {t("about.analysis.confusion_3")}</p>
              <p><strong>4.</strong> {t("about.analysis.confusion_4")}</p>
            </div>
          </div>
          <div className="space-y-4 p-6 rounded-xl bg-violet-500/5 border border-violet-500/20">
            <h3 className="text-foreground font-bold text-base">{t("about.analysis.root_cause_title")}</h3>
            <div className="text-sm text-muted-foreground space-y-2">
              <p>{t("about.analysis.root_cause_detail")}</p>
            </div>
          </div>
        </div>

        <div className="p-6 rounded-xl bg-amber-500/5 border border-amber-500/20 text-muted-foreground leading-relaxed">
          <p className="font-bold text-foreground mb-2">{t("about.analysis.literature_title")}</p>
          <p className="text-sm">
            {renderCitations(t("about.analysis.literature_detail"))}
          </p>
        </div>
      </section>

      {/* 10 ============================================================
            EXP 10: ULTIMATE EXPERIMENT
      ============================================================ */}
      <section className="space-y-6 md:space-y-8 pt-4">
        <div className="flex items-center gap-4">
          <div className="size-12 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center text-white shadow-lg">
            <Rocket className="size-6" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">{t("about.exp10.title")}</h2>
        </div>

        <p className="text-muted-foreground leading-relaxed">
          {t("about.exp10.description")}
        </p>

        {/* Innovation 1 */}
        <div className="p-6 rounded-xl bg-gradient-to-br from-blue-500/10 to-blue-500/5 border border-blue-500/20">
          <div className="flex items-center gap-3 mb-3">
            <div className="size-8 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-500 font-black">1</div>
            <h3 className="text-lg font-bold">{t("about.exp10.innovation1_title")}</h3>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {renderCitations(t("about.exp10.innovation1_detail"))}
          </p>
        </div>

        {/* Innovation 2 */}
        <div className="p-6 rounded-xl bg-gradient-to-br from-purple-500/10 to-purple-500/5 border border-purple-500/20">
          <div className="flex items-center gap-3 mb-3">
            <div className="size-8 rounded-lg bg-purple-500/10 flex items-center justify-center text-purple-500 font-black">2</div>
            <h3 className="text-lg font-bold">{t("about.exp10.innovation2_title")}</h3>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {renderCitations(t("about.exp10.innovation2_detail"))}
          </p>
        </div>

        {/* Innovation 3 */}
        <div className="p-6 rounded-xl bg-gradient-to-br from-green-500/10 to-green-500/5 border border-green-500/20">
          <div className="flex items-center gap-3 mb-3">
            <div className="size-8 rounded-lg bg-green-500/10 flex items-center justify-center text-green-500 font-black">3</div>
            <h3 className="text-lg font-bold">{t("about.exp10.innovation3_title")}</h3>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {renderCitations(t("about.exp10.innovation3_detail"))}
          </p>
        </div>

        {/* Innovation 4 */}
        <div className="p-6 rounded-xl bg-gradient-to-br from-amber-500/10 to-amber-500/5 border border-amber-500/20">
          <div className="flex items-center gap-3 mb-3">
            <div className="size-8 rounded-lg bg-amber-500/10 flex items-center justify-center text-amber-500 font-black">4</div>
            <h3 className="text-lg font-bold">{t("about.exp10.innovation4_title")}</h3>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {t("about.exp10.innovation4_detail")}
          </p>
          <ul className="text-sm text-muted-foreground space-y-1 mt-2 list-disc list-inside">
            <li>{renderCitations(t("about.exp10.innovation4_maha"))}</li>
            <li>{t("about.exp10.innovation4_knn")}</li>
            <li>{renderCitations(t("about.exp10.innovation4_energy"))}</li>
          </ul>
          <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
            {t("about.exp10.innovation4_ensemble")}
          </p>
        </div>

        {/* Expected outcomes */}
        <div className="p-6 rounded-xl bg-gradient-to-br from-primary/10 to-primary/5 border border-primary/20">
          <div className="flex items-center gap-3 mb-3">
            <Rocket className="size-5 text-primary" />
            <h3 className="text-lg font-bold">{t("about.exp10.outcome_title")}</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/50 text-xs uppercase tracking-widest text-muted-foreground font-bold">
                  <th className="text-left py-2 px-3">{t("about.exp10.outcome_title")}</th>
                  <th className="text-center py-2 px-3">{t("about.exp10.outcome_baseline")}</th>
                  <th className="text-center py-2 px-3">{t("about.exp10.outcome_target")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                <tr>
                  <td className="py-2 px-3 font-medium">{t("about.exp10.outcome_known_f1")}</td>
                  <td className="text-center py-2 px-3 font-mono">0.927 (CE)</td>
                  <td className="text-center py-2 px-3 font-mono text-green-600">{'>'}0.94</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 font-medium">{t("about.exp10.outcome_majellensis")}</td>
                  <td className="text-center py-2 px-3 font-mono">96\u2013100% (CE)</td>
                  <td className="text-center py-2 px-3 font-mono text-green-600">{'>'}98%</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 font-medium">{t("about.exp10.outcome_auroc")}</td>
                  <td className="text-center py-2 px-3 font-mono">~0.45</td>
                  <td className="text-center py-2 px-3 font-mono text-green-600">{'>'}0.55</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 font-medium">{t("about.exp10.outcome_best_auroc")}</td>
                  <td className="text-center py-2 px-3 font-mono">0.497</td>
                  <td className="text-center py-2 px-3 font-mono text-green-600">{'>'}0.60</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 font-medium">{t("about.exp10.outcome_clustering")}</td>
                  <td className="text-center py-2 px-3 font-mono">~0.46</td>
                  <td className="text-center py-2 px-3 font-mono text-green-600">{'>'}0.55</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <p className="text-muted-foreground text-sm leading-relaxed">
          {t("about.exp10.outcome_note")}
        </p>

        <NcdResults experiment="Exp 10 (MGH-NCD)" />
      </section>

      {/* 12. Bibliography */}
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
                className="block text-foreground hover:text-primary transition-colors"
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

const MEDAL = ["🥇", "🥈", "🥉"];

function MicrosoftLogo() {
  return <img src="/microsoft-logo.webp" alt="Microsoft" className="size-full object-scale-down p-1" />;
}
function MetaLogo() {
  return <img src="/meta-logo.webp" alt="Meta" className="h-full max-w-full object-scale-down p-1" />;
}

const FAMILY: Record<string, { label: string; logo: React.ReactNode }> = {
  resnet18:        { label: "ResNet / Microsoft", logo: <MicrosoftLogo /> },
  resnet50:        { label: "ResNet / Microsoft", logo: <MicrosoftLogo /> },
  convnext_tiny:   { label: "ConvNeXt / Meta",   logo: <MetaLogo /> },
  convnext_small:  { label: "ConvNeXt / Meta",   logo: <MetaLogo /> },
  dinov2_small:    { label: "DINOv2 / Meta",     logo: <MetaLogo /> },
  dinov2_base:     { label: "DINOv2 / Meta",     logo: <MetaLogo /> },
};

function formatPct(v: number | null | undefined, d = 2): string {
  if (v == null) return "—";
  return (v * 100).toFixed(d) + "%";
}

function formatF1WithStd(f1: number | null | undefined, std: number | null | undefined): string {
  if (f1 == null) return "—";
  const base = (f1 * 100).toFixed(2);
  if (std == null) return base;
  return `${base} ± ${(std * 100).toFixed(2)}`;
}

function ExpTable({ exp: entries }: { exp: ModelComparisonEntry[] }) {
  const sorted = [...entries].sort((a, b) => -(a.mean_f1 ?? 0) + (b.mean_f1 ?? 0));
  const [expanded, setExpanded] = useState<string | null>(null);
  const toggle = useCallback((v: string) => setExpanded(p => p === v ? null : v), []);

  if (sorted.length === 0) return null;

  const best = sorted[0];
  const gap = sorted.length >= 2 && sorted[0]?.mean_f1 != null && sorted[sorted.length - 1]?.mean_f1 != null
    ? (sorted[0].mean_f1! - sorted[sorted.length - 1].mean_f1!) * 100 : null;

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-xl border border-border/50 bg-background/50">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border/50 text-xs uppercase tracking-widest text-muted-foreground font-bold">
              <th className="text-left py-3 px-4 w-12">#</th>
              <th className="text-left py-3 px-3">Modello</th>
              <th className="text-center py-3 px-2">F1 (CV)</th>
              <th className="text-center py-3 px-2">Acc (CV)</th>
              <th className="text-center py-3 px-2">Test F1</th>
              <th className="text-center py-3 px-2">Test Acc</th>
              <th className="text-center py-3 pr-4 pl-2">Best Fold</th>
              <th className="w-6" />
            </tr>
          </thead>
          <tbody className="divide-y divide-border/30">
            {sorted.map((m, i) => (
              <ExpRow key={m.model_variant} model={m} rank={i + 1}
                isExpanded={expanded === m.model_variant}
                onToggle={() => toggle(m.model_variant)} />
            ))}
          </tbody>
        </table>
      </div>

      {best?.mean_f1 != null && (
        <div className="space-y-1 text-sm text-muted-foreground leading-relaxed">
          <CommentLine text={`Vincente: ${best.name} con F1 ${(best.mean_f1 * 100).toFixed(2)}%`} />
          {gap != null && gap > 1 && (
            <CommentLine text={`Gap primo-ultimo: ${gap.toFixed(1)} punti percentuali`} />
          )}
        </div>
      )}
    </div>
  );
}

function ExpRow({ model, rank, isExpanded, onToggle }: {
  model: ModelComparisonEntry; rank: number; isExpanded: boolean; onToggle: () => void;
}) {
  const foldQuery = useQuery({
    queryKey: ["fold", model.model_variant],
    queryFn: () => Promise.all(
      Array.from({ length: model.num_folds ?? 5 }, (_, i) =>
        fetchApi<FoldData>(`/api/comparison/${model.model_variant}/fold/${i}`))
    ),
    staleTime: 5 * 60 * 1000,
    enabled: isExpanded,
  });
  const colors = useChartColors();
  const fam = FAMILY[model.model_variant];
  const colorIdx = fam ? Object.keys(FAMILY).indexOf(model.model_variant) : 0;
  const color = colors[Math.max(0, colorIdx)];

  return (
    <>
      <tr onClick={onToggle} className={`cursor-pointer transition-colors hover:bg-muted/30 ${rank === 1 ? "bg-amber-500/5" : ""}`}>
        <td className="py-3 px-4"><span className="text-lg font-black tabular-nums">{rank <= 3 ? MEDAL[rank - 1] : `#${rank}`}</span></td>
        <td className="py-3 px-3">
          <div className="flex items-center gap-3">
            <span className="size-8 rounded-lg flex items-center justify-center shrink-0 bg-background/80 ring-1 ring-border/60">
              {fam?.logo}
            </span>
            <div>
              <div className="font-bold text-foreground">{model.name}</div>
              {fam && <div className="text-[11px] text-muted-foreground tracking-wider uppercase">{fam.label}</div>}
            </div>
          </div>
        </td>
        <td className="text-center py-3 px-2 font-bold tabular-nums">{formatF1WithStd(model.mean_f1, model.std_f1)}</td>
        <td className="text-center py-3 px-2 tabular-nums">{formatPct(model.mean_accuracy)}</td>
        <td className="text-center py-3 px-2 tabular-nums">{formatPct(model.test_f1)}</td>
        <td className="text-center py-3 px-2 tabular-nums text-xs text-muted-foreground">{formatPct(model.test_accuracy)}</td>
        <td className="text-center py-3 pr-4 pl-2 tabular-nums text-xs text-muted-foreground">{model.best_fold != null ? `Fold ${model.best_fold}` : "—"}</td>
        <td className="text-muted-foreground">{isExpanded ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}</td>
      </tr>
      {isExpanded && (
        <tr><td colSpan={7} className="p-0">
          <div className="bg-muted/10 border-t border-border/30 p-4">
            {foldQuery.isLoading ? (
              <div className="flex justify-center py-4"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>
            ) : foldQuery.error ? (
              <p className="text-xs text-muted-foreground text-center py-4">Fold details unavailable</p>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                {foldQuery.data?.map((fold, i) => (
                  <FoldCard key={i} fold={fold} index={i} color={color} />
                ))}
              </div>
            )}
          </div>
        </td></tr>
      )}
    </>
  );
}

function FoldCard({ fold, index, color }: { fold: FoldData; index: number; color: string }) {
  const points = foldToEpochPoints(fold);
  return (
    <div className="rounded-xl border border-border/40 bg-background p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-bold text-xs text-muted-foreground uppercase tracking-wider">Fold {index}</span>
        <span className="text-xs font-bold tabular-nums" style={{ color }}>F1: {(fold.best_f1 * 100).toFixed(1)}%</span>
      </div>
      <div className="h-14">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
            <YAxis domain={["auto", "auto"]} hide />
            <Tooltip contentStyle={{ fontSize: 11, padding: "4px 8px", borderRadius: 6 }}
              labelFormatter={(v) => `Epoch ${v}`}
              formatter={(v) => [Number(v) * 100 + "%", "F1"]} />
            <Line type="monotone" dataKey="val_f1" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-between text-[11px] text-muted-foreground tabular-nums">
        <span>Best ep: {fold.best_epoch}</span>
        <span>Acc: {(fold.best_accuracy * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

function CommentLine({ text }: { text: string }) {
  return <p className="pl-4 border-l-2 border-primary/30">{text}</p>;
}

function Exp9NcdResults() {
  const { t } = useTranslation();

  const ncdCompQuery = useQuery<NcdComparison>({
    queryKey: ["ncd-comparison"],
    queryFn: fetchNcdComparison,
    staleTime: 5 * 60 * 1000,
  });

  const exp9Models = ncdCompQuery.data?.models?.filter((m) => m.experiment === "Exp 9 (NCD)") ?? [];
  const sorted = [...exp9Models].sort((a, b) => b.mean_auroc - a.mean_auroc);
  const isLoading = ncdCompQuery.isLoading;

  if (isLoading) {
    return <div className="flex justify-center py-8"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>;
  }

  if (!exp9Models.length) {
    return (
      <div className="p-6 rounded-xl bg-amber-500/5 border border-amber-500/20 text-muted-foreground leading-relaxed">
        <p className="font-bold text-foreground mb-2">{t("about.exp9.training")}</p>
        <p className="text-sm">{t("about.exp9.waiting")}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Explanation */}
      <div className="p-6 rounded-xl bg-red-500/5 border border-red-500/20 text-muted-foreground leading-relaxed">
        <p className="font-bold text-foreground mb-2">{t("about.exp9.negative_result_title")}</p>
        <p className="text-sm">{t("about.exp9.negative_result")}</p>
      </div>

      {/* Results table */}
      <div className="overflow-x-auto rounded-xl border border-border/50 bg-background/50">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border/50 text-xs uppercase tracking-widest text-muted-foreground font-bold">
              <th className="text-left py-3 px-4 w-10">#</th>
              <th className="text-left py-3 px-3">{t("about.exp9.model")}</th>
              <th className="text-center py-3 px-3">{t("about.exp9.clust_acc")}</th>
              <th className="text-center py-3 px-3">{t("about.exp9.auroc")}</th>
              <th className="text-center py-3 px-3">{t("about.exp9.baseline")}</th>
              <th className="text-center py-3 pr-5 pl-3">{t("about.exp9.delta")}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/30">
            {(sorted as NcdComparisonEntry[]).map((m, i) => {
              const isBest = m.mean_auroc >= Math.max(...sorted.map(x => x.mean_auroc));
              return (
                <tr key={m.model} className={isBest ? "bg-amber-500/5" : "hover:bg-muted/30 transition-colors"}>
                  <td className="py-3 px-4"><span className="text-lg font-black">{i <= 2 ? ["🥇","🥈","🥉"][i] : `#${i+1}`}</span></td>
                  <td className="py-3 px-3 font-bold">{m.model}</td>
                  <td className="text-center py-3 px-3 tabular-nums">{(m.acc * 100).toFixed(1)}%</td>
                  <td className="text-center py-3 px-3 font-bold tabular-nums">{(m.mean_auroc * 100).toFixed(1)}%</td>
                  <td className="text-center py-3 px-3 text-muted-foreground tabular-nums">{(m.baseline_auroc * 100).toFixed(1)}%</td>
                  <td className={`text-center py-3 pr-5 pl-3 tabular-nums ${m.improvement >= 0 ? "text-green-600" : "text-red-500"}`}>
                    {m.improvement >= 0 ? "+" : ""}{(m.improvement * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const DETECTOR_COLORS: Record<string, string> = {
  Mahalanobis: "#6366f1",
  KNN: "#22c55e",
  Energy: "#f59e0b",
  Ensemble: "#8b5cf6",
};

function NcdResults({ experiment }: { experiment: string }) {
  const { t } = useTranslation();
  const ncdCompQuery = useQuery<NcdComparison>({
    queryKey: ["ncd-comparison"],
    queryFn: fetchNcdComparison,
    staleTime: 5 * 60 * 1000,
  });

  const models = ncdCompQuery.data?.models?.filter((m) => m.experiment === experiment) ?? [];
  const sorted = [...models].sort((a, b) => b.ensemble_auroc - a.ensemble_auroc);
  const isLoading = ncdCompQuery.isLoading;

  const chartData = sorted.map((m) => ({
    name: m.model.replace("_", " "),
    Mahalanobis: +(m.mahalanobis_auroc * 100).toFixed(1),
    KNN: +(m.knn_auroc * 100).toFixed(1),
    Energy: +(m.energy_auroc * 100).toFixed(1),
    Ensemble: +(m.ensemble_auroc * 100).toFixed(1),
  }));

  if (isLoading) {
    return <div className="flex justify-center py-8"><div className="size-5 rounded-full border-2 border-primary border-t-transparent animate-spin" /></div>;
  }

  if (models.length === 0) return null;

  return (
    <div className="space-y-8 mt-8">
      <h3 className="text-xl font-bold flex items-center gap-2">
        <Rocket className="size-5 text-primary" />
        {experiment === "Exp 10 (MGH-NCD)" ? t("about.exp10.ncd_title") : "NCD Results — Open-Set Detection per Detector"}
      </h3>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-border/50 bg-background/50">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border/50 text-xs uppercase tracking-widest text-muted-foreground font-bold">
              <th className="text-left py-3 px-4 w-10">#</th>
              <th className="text-left py-3 px-3">Modello</th>
              {["Ensemble", "Mahalanobis", "KNN", "Energy"].map((d) => (
                <th key={d} className="text-center py-3 px-2" style={{ color: DETECTOR_COLORS[d] }}>
                  {d}
                </th>
              ))}
              <th className="text-center py-3 px-2">Clust Acc</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/30">
            {sorted.map((m, i) => (
              <tr key={m.model} className={i === 0 ? "bg-amber-500/5" : "hover:bg-muted/30 transition-colors"}>
                <td className="py-3 px-4"><span className="text-lg font-black">{i <= 2 ? ["🥇","🥈","🥉"][i] : `#${i+1}`}</span></td>
                <td className="py-3 px-3 font-bold">{m.model}</td>
                {["Ensemble", "Mahalanobis", "KNN", "Energy"].map((d) => {
                  const key = `${d.toLowerCase()}_auroc` as keyof typeof m;
                  const val = m[key] as number;
                  return (
                    <td key={d} className="text-center py-3 px-2 font-bold tabular-nums"
                        style={{ color: d === "Ensemble" ? DETECTOR_COLORS.Ensemble : undefined }}>
                      {(val * 100).toFixed(1)}%
                    </td>
                  );
                })}
                <td className="text-center py-3 px-2 tabular-nums text-muted-foreground">
                  {(m.cluster_acc * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Bar chart */}
      <div className="rounded-xl border border-border/50 bg-background/50 p-4 md:p-6">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} tickFormatter={(v) => `${v}%`} />
            <Tooltip
              contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid hsl(var(--border))" }}
              formatter={(v, name) => [`${v ?? 0}%`, name]}
            />
            {["Mahalanobis", "KNN", "Energy", "Ensemble"].map((d) => (
              <Bar key={d} dataKey={d} fill={DETECTOR_COLORS[d]} name={d}
                   radius={[4, 4, 0, 0]} maxBarSize={24} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
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
