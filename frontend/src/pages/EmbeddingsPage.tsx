import { useState, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { Box, RotateCw, ZoomIn, ZoomOut, Image as ImageIcon } from "lucide-react";
import { useEmbeddings } from "@/hooks/useEmbeddings";
import { Scatter3D } from "@/features/embeddings/Scatter3D";
import { Scatter2D } from "@/features/embeddings/Scatter2D";

const PALETTE = [
  "#3B82F6", "#D97706", "#10B981", "#8B5CF6",
  "#EF4444", "#EC4899", "#14B8A6", "#F97316",
];

const API_BASE = import.meta.env.VITE_API_URL ?? "";

const EXP_OPTIONS = [
  { group: "Exp 2 (SupCon)", prefix: "Exp 2", options: [
    { value: "exp2_supcon_resnet18", label: "ResNet-18" },
    { value: "exp2_supcon_resnet50", label: "ResNet-50" },
    { value: "exp2_supcon_convnext_tiny", label: "ConvNeXt-Tiny" },
    { value: "exp2_supcon_convnext_small", label: "ConvNeXt-Small" },
    { value: "exp2_supcon_dinov2_small", label: "DINOv2-Small" },
    { value: "exp2_supcon_dinov2_base", label: "DINOv2-Base" },
  ]},
  { group: "Exp 3 (SpHOR)", prefix: "Exp 3", options: [
    { value: "exp3_sphor_resnet18", label: "ResNet-18" },
    { value: "exp3_sphor_resnet50", label: "ResNet-50" },
    { value: "exp3_sphor_convnext_tiny", label: "ConvNeXt-Tiny" },
    { value: "exp3_sphor_convnext_small", label: "ConvNeXt-Small" },
    { value: "exp3_sphor_dinov2_small", label: "DINOv2-Small" },
    { value: "exp3_sphor_dinov2_base", label: "DINOv2-Base" },
  ]},

  { group: "Exp 4 (NCD, novel O. garganica)", prefix: "Exp 4", options: [
    { value: "exp4_sphor_ncd_resnet18", label: "ResNet-18" },
    { value: "exp4_sphor_ncd_resnet50", label: "ResNet-50" },
    { value: "exp4_sphor_ncd_convnext_tiny", label: "ConvNeXt-Tiny" },
    { value: "exp4_sphor_ncd_convnext_small", label: "ConvNeXt-Small" },
    { value: "exp4_sphor_ncd_dinov2_small", label: "DINOv2-Small" },
    { value: "exp4_sphor_ncd_dinov2_base", label: "DINOv2-Base" },
  ]},
];

export function EmbeddingsPage() {
  const { t } = useTranslation();
  const [experiment, setExperiment] = useState("exp4_sphor_ncd_dinov2_base");
  const [method, setMethod] = useState("tsne_3d");
  const [split, setSplit] = useState("all");
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [imgError, setImgError] = useState(false);

  const { data, isLoading, error } = useEmbeddings(experiment, method);

  const speciesColors = useMemo(() => {
    if (!data) return {} as Record<string, string>;
    const colors: Record<string, string> = {};
    data.species.forEach((s, i) => { colors[s] = PALETTE[i % PALETTE.length]; });
    return colors;
  }, [data]);

  const visiblePoints = useMemo(() => {
    if (!data) return [];
    return data.points
      .filter((p) => !hidden.has(p.species))
      .filter((p) => split === "all" || p.split === split)
      .map((p) => ({ x: p.x, y: p.y, z: p.z, species: p.species, idx: p.idx }));
  }, [data, hidden, split]);

  const selectedPoint = useMemo(() => {
    if (selectedIdx === null || !data) return null;
    return data.points[selectedIdx] ?? null;
  }, [selectedIdx, data]);

  const speciesStats = useMemo(() => {
    const pts = visiblePoints.length > 0 ? visiblePoints : [];
    const counts: Record<string, number> = {};
    for (const p of pts) counts[p.species] = (counts[p.species] || 0) + 1;
    const total = pts.length;
    return (data?.species ?? []).map((species) => ({ species, count: counts[species] || 0, total }))
      .sort((a, b) => b.count - a.count);
  }, [visiblePoints, data]);

  const toggleSpecies = (species: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(species)) next.delete(species); else next.add(species);
      return next;
    });
    setSelectedIdx(null);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 md:px-12 pb-20 animate-in fade-in duration-700">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="size-10 md:size-12 rounded-2xl bg-indigo-500/10 flex items-center justify-center shrink-0 text-indigo-500">
          <Box className="size-6" />
        </div>
        <div>
          <h1 className="text-2xl md:text-4xl font-black tracking-tight">{t("embeddings.title")}</h1>
          <p className="text-muted-foreground text-sm mt-1">{t("embeddings.description")}</p>
        </div>
      </div>

      {/* Selectors — always visible */}
      <div className="flex flex-wrap gap-2">
        <select value={experiment} onChange={(e) => setExperiment(e.target.value)}
          className="text-xs bg-card border border-border/50 rounded-lg px-2.5 py-1.5 text-foreground cursor-pointer appearance-auto">
          {EXP_OPTIONS.map((g) => (
            <optgroup key={g.group} label={g.group}>
              {g.options.map((o) => <option key={o.value} value={o.value} className="bg-card text-foreground">{g.prefix} — {o.label}</option>)}
            </optgroup>
          ))}
        </select>
        <select value={method} onChange={(e) => setMethod(e.target.value)}
          className="text-xs bg-card border border-border/50 rounded-lg px-2.5 py-1.5 text-foreground cursor-pointer appearance-auto">
          <option value="tsne_3d" className="bg-card text-foreground">{t("research.method_tsne3d")}</option>
          <option value="tsne_2d" className="bg-card text-foreground">{t("research.method_tsne2d")}</option>
          <option value="umap_2d" className="bg-card text-foreground">{t("research.method_umap2d")}</option>
          <option value="umap_3d" className="bg-card text-foreground">{t("research.method_umap3d")}</option>
        </select>
        <select value={split} onChange={(e) => setSplit(e.target.value)}
          className="text-xs bg-card border border-border/50 rounded-lg px-2.5 py-1.5 text-foreground cursor-pointer appearance-auto">
          <option value="all" className="bg-card text-foreground">{t("research.split_all")}</option>
          <option value="train" className="bg-card text-foreground">{t("research.split_train")}</option>
          <option value="test" className="bg-card text-foreground">{t("research.split_test")}</option>
          <option value="known" className="bg-card text-foreground">{t("research.split_known")}</option>
          <option value="novel" className="bg-card text-foreground">{t("research.split_novel")}</option>
        </select>
        <Link to={`/experiments/${experiment}`}
          className="text-xs text-primary hover:underline px-2 py-1.5 font-medium">
          {t("embeddings.view_experiment")} →
        </Link>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex items-center justify-center min-h-[40vh]">
          <div className="size-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
        </div>
      ) : error || !data ? (
        <div className="text-center py-20 space-y-4">
          <Box className="size-12 mx-auto text-muted-foreground/30" />
          <p className="text-muted-foreground">{t("embeddings.no_data")}</p>
          <p className="text-xs text-muted-foreground/50 max-w-md mx-auto leading-relaxed">{t("embeddings.hint")}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-6">
          {/* 3D Viewer */}
          <div className="rounded-2xl border border-border/50 bg-background/50 p-1 h-[500px] md:h-[600px]">
            {method.includes("3d") ? (
              <Scatter3D points={visiblePoints} species={data.species} speciesColors={speciesColors} onSelect={(i: number | null) => { setSelectedIdx(i); setImgError(false); }} />
            ) : (
              <Scatter2D points={visiblePoints.map((p) => ({ x: p.x, y: p.y, species: p.species, idx: p.idx }))}
                speciesColors={speciesColors} onSelect={(i: number | null) => { setSelectedIdx(i); setImgError(false); }} />
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            <div className="rounded-xl border border-border/40 bg-muted/20 p-3 space-y-2 text-sm text-muted-foreground">
              <div className="flex items-center gap-2"><RotateCw className="size-3.5" /><span>{t("embeddings.drag_rotate")}</span></div>
              <div className="flex items-center gap-2"><RotateCw className="size-3.5" /><span>{t("embeddings.ctrl_pan")}</span></div>
              <div className="flex items-center gap-2"><ZoomIn className="size-3.5" /><ZoomOut className="size-3.5" /><span>{t("embeddings.scroll_zoom")}</span></div>
            </div>

            {/* Species legend — always visible */}
            <div className="rounded-xl border border-border/40 bg-muted/20 p-3 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold uppercase tracking-wider text-muted-foreground">{t("embeddings.dataset_overview")}</span>
              </div>
              <div className="space-y-1.5">
                {speciesStats.map(({ species, count, total }) => {
                  const isHidden = hidden.has(species);
                  return (
                    <button key={species} onClick={() => toggleSpecies(species)}
                      className={`flex items-center gap-2 w-full text-left text-sm transition-all hover:opacity-80 ${isHidden ? "opacity-50" : ""}`}>
                      <span className={`size-2.5 rounded-full shrink-0 transition-all ${isHidden ? "opacity-30" : ""}`}
                        style={{ backgroundColor: speciesColors[species] || PALETTE[0] }} />
                      <span className={`flex-1 truncate transition-all ${isHidden ? "text-muted-foreground/40 line-through" : "text-muted-foreground"}`}>
                        {species}
                      </span>
                      <span className="font-bold tabular-nums text-foreground/80">{count}</span>
                      <span className="text-muted-foreground/50 text-[10px]">({total > 0 ? ((count / total) * 100).toFixed(0) : "0"}%)</span>
                    </button>
                  );
                })}
              </div>
              <p className="text-xs text-muted-foreground/50 pt-1 italic">{t("embeddings.click_hint")}</p>
            </div>

            {/* Selected point image — only when a point is selected */}
            {selectedPoint && (
              <div className="rounded-xl border border-border/40 bg-muted/20 overflow-hidden">
                <div className="aspect-square bg-muted/30 flex items-center justify-center relative">
                  {imgError ? (
                    <div className="flex flex-col items-center gap-2 p-4 text-center">
                      <ImageIcon className="size-8 text-muted-foreground/30" />
                      <p className="text-xs text-muted-foreground/60">{t("research.img_not_available")}</p>
                      <p className="text-[10px] text-muted-foreground/40">{t("research.img_download_hint")}</p>
                    </div>
                  ) : (
                    <img src={`${API_BASE}${selectedPoint.image_url}`} alt={selectedPoint.species}
                      className="w-full h-full object-contain"
                      onError={() => setImgError(true)}
                      onLoad={() => setImgError(false)}
                      loading="lazy" />
                  )}
                </div>
                <div className="p-3 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="size-2.5 rounded-full shrink-0" style={{ backgroundColor: speciesColors[selectedPoint.species] || PALETTE[0] }} />
                    <span className="font-bold text-sm">{selectedPoint.species}</span>
                  </div>
                  <p className="text-[11px] text-muted-foreground truncate">{selectedPoint.filename}</p>
                  <button onClick={() => setSelectedIdx(null)} className="text-[11px] text-primary hover:underline mt-1">{t("embeddings.deselect")}</button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
