export interface Paper {
  id: string;
  authors: string;
  year: number;
  title: string;
  venue?: string;
  url?: string;
  doi?: string;
}

export const PAPERS: Paper[] = [
  { id: "yu2021benchmark", authors: "Yu et al.", year: 2021, title: "Ultra-Fine-Grained Visual Categorization Benchmark", venue: "CVPR", url: "https://arxiv.org/abs/2103.13074" },
  { id: "khosla2020supervised", authors: "Khosla et al.", year: 2020, title: "Supervised Contrastive Learning", venue: "NeurIPS", url: "https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html" },
  { id: "bahavan2026sphor", authors: "Bahavan et al.", year: 2026, title: "SpHOR: A Representation Learning Perspective on Open-Set Recognition", venue: "CVPR 2026 Findings", url: "https://arxiv.org/abs/2503.08049" },
  { id: "he2016deep", authors: "He et al.", year: 2016, title: "Deep Residual Learning for Image Recognition", venue: "CVPR", url: "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html" },
  { id: "liu2022convnet", authors: "Liu et al.", year: 2022, title: "A ConvNet for the 2020s", venue: "CVPR", url: "https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html" },
  { id: "oquab2024dinov2", authors: "Oquab et al.", year: 2024, title: "DINOv2: Learning Robust Visual Features without Supervision", venue: "arXiv", url: "https://arxiv.org/abs/2304.07193" },
  { id: "belkin2019double", authors: "Belkin et al.", year: 2019, title: "Reconciling modern machine-learning practice and the classical bias–variance trade-off", venue: "PNAS", doi: "10.1073/pnas.1903070116" },
  { id: "ioffe2015batchnorm", authors: "Ioffe & Szegedy", year: 2015, title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", venue: "ICML", url: "https://proceedings.mlr.press/v37/ioffe15.html" },
  { id: "bendale2015openworld", authors: "Bendale & Boult", year: 2015, title: "Towards Open World Recognition", venue: "IEEE TPAMI", url: "https://ieeexplore.ieee.org/document/7284977" },
  { id: "liu2020energy", authors: "Liu et al.", year: 2020, title: "Energy-based Out-of-distribution Detection", venue: "NeurIPS", url: "https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html" },
  { id: "geng2020openset", authors: "Geng et al.", year: 2020, title: "Open-Set Recognition: A Survey", venue: "IEEE TPAMI", url: "https://ieeexplore.ieee.org/document/9178490" },
  { id: "lee2018simple", authors: "Lee et al.", year: 2018, title: "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks", venue: "NeurIPS", url: "https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html" },
  { id: "vaze2022gcd", authors: "Vaze et al.", year: 2022, title: "Generalized Category Discovery", venue: "CVPR", url: "https://openaccess.thecvf.com/content/CVPR2022/html/Vaze_Generalized_Category_Discovery_CVPR_2022_paper.html" },
  { id: "zheng2025mgce", authors: "Zheng et al.", year: 2025, title: "Multi-Granularity Conceptual Experts for Generalized Category Discovery", venue: "arXiv", url: "https://arxiv.org/abs/2509.26227" },
  { id: "devey2008ophrys", authors: "Devey et al.", year: 2008, title: "Friends or relatives? Phylogenetics and species delimitation in the genus Ophrys", venue: "Annals of Botany", doi: "10.1093/aob/mcn075" },
  { id: "schiestl2005swindle", authors: "Schiestl", year: 2005, title: "On the success of a swindle: pollination by deception in orchids", venue: "Naturwissenschaften", url: "https://doi.org/10.1007/s00114-005-0636-y" },
  { id: "dai2025apl", authors: "Dai et al.", year: 2025, title: "Adaptive Part Learning for Fine-Grained Generalized Category Discovery", venue: "CVPR 2025", url: "https://arxiv.org/abs/2507.06928" },
  { id: "wang2025partgcd", authors: "Wang et al.", year: 2025, title: "PartGCD: Learning Part Knowledge to Facilitate Category Understanding for Fine-Grained GCD", venue: "arXiv", url: "https://arxiv.org/abs/2503.16782" },
  { id: "hu2022lora", authors: "Hu et al.", year: 2022, title: "LoRA: Low-Rank Adaptation of Large Language Models", venue: "ICLR", url: "https://openreview.net/forum?id=nZeVKeeFYf9" },
  { id: "liu2024novel", authors: "Liu et al.", year: 2024, title: "Novel Class Discovery for Ultra-Fine-Grained Visual Categorization", venue: "CVPR", url: "https://arxiv.org/abs/2405.06283" },
];

export const PAPER_MAP = Object.fromEntries(PAPERS.map((p) => [p.id, p]));
