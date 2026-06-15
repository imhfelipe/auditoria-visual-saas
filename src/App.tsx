import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Cpu,
  Download,
  FileVideo,
  Gauge,
  Pause,
  Play,
  RotateCcw,
  Square,
  Upload,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { CLASSES, CLASS_MAP, type ClassKey } from "@/lib/audit-classes";
import {
  VideoCanvas,
  type DetectedObject,
  type Roi,
} from "@/components/audit/VideoCanvas";

const CANVAS_W = 800;
const CANVAS_H = 450;
const TOTAL_FRAMES = 600; // ~50s @ 12fps
const FPS = 12;
const TICK_MS = 1000 / FPS;

interface LiveObject extends DetectedObject {
  vx: number;
  vy: number;
  enteredRoi: boolean;
}

interface HistoryPoint {
  frame: number;
  t: number; // seconds
  [k: string]: number;
}

const DEFAULT_SELECTED: ClassKey[] = ["person", "car", "cellphone"];

function fmtTime(sec: number) {
  const m = Math.floor(sec / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(sec % 60)
    .toString()
    .padStart(2, "0");
  const ms = Math.floor((sec - Math.floor(sec)) * 100)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}.${ms}`;
}

export default function App() {
  const [fileName, setFileName] = useState<string | null>(null);
  const [selected, setSelected] = useState<ClassKey[]>(DEFAULT_SELECTED);
  const [roi, setRoi] = useState<Roi>({ xMin: 15, xMax: 85, yMin: 25, yMax: 85 });
  const [running, setRunning] = useState(false);
  const [finished, setFinished] = useState(false);
  const [frame, setFrame] = useState(0);
  const [objects, setObjects] = useState<LiveObject[]>([]);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [modelLoaded, setModelLoaded] = useState(false);

  // refs for simulation
  const objectsRef = useRef<LiveObject[]>([]);
  const idRef = useRef(1);
  const uniqueInRoiRef = useRef<Record<ClassKey, Set<number>>>(
    {} as Record<ClassKey, Set<number>>,
  );
  const totalUniqueRef = useRef<Record<ClassKey, Set<number>>>(
    {} as Record<ClassKey, Set<number>>,
  );
  const peakRef = useRef<Record<ClassKey, { peak: number; frame: number }>>(
    {} as Record<ClassKey, { peak: number; frame: number }>,
  );
  const historyRef = useRef<HistoryPoint[]>([]);
  const frameRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef(0);

  const resetState = useCallback(() => {
    objectsRef.current = [];
    idRef.current = 1;
    uniqueInRoiRef.current = {} as Record<ClassKey, Set<number>>;
    totalUniqueRef.current = {} as Record<ClassKey, Set<number>>;
    peakRef.current = {} as Record<ClassKey, { peak: number; frame: number }>;
    historyRef.current = [];
    frameRef.current = 0;
    for (const c of CLASSES) {
      uniqueInRoiRef.current[c.key] = new Set();
      totalUniqueRef.current[c.key] = new Set();
      peakRef.current[c.key] = { peak: 0, frame: 0 };
    }
    setFrame(0);
    setObjects([]);
    setHistory([]);
    setFinished(false);
  }, []);

  useEffect(() => {
    resetState();
  }, [resetState]);

  const handleFile = (f: File | null) => {
    if (!f) return;
    setFileName(f.name);
    resetState();
  };

  const toggleClass = (k: ClassKey) => {
    setSelected((prev) =>
      prev.includes(k) ? prev.filter((x) => x !== k) : [...prev, k],
    );
  };

  const start = useCallback(() => {
    if (!fileName) return;
    if (finished) resetState();
    if (!modelLoaded) {
      // simulate model load delay
      setTimeout(() => {
        setModelLoaded(true);
        setRunning(true);
      }, 700);
      return;
    }
    setRunning(true);
  }, [fileName, finished, modelLoaded, resetState]);

  const stop = () => setRunning(false);

  // simulation tick
  useEffect(() => {
    if (!running) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }
    const loop = (ts: number) => {
      if (ts - lastTickRef.current >= TICK_MS) {
        lastTickRef.current = ts;
        tick();
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, roi, selected]);

  const tick = () => {
    const f = frameRef.current + 1;
    frameRef.current = f;
    const dt = 1 / FPS;

    // ROI in px
    const rx = (roi.xMin / 100) * CANVAS_W;
    const ry = (roi.yMin / 100) * CANVAS_H;
    const rw = ((roi.xMax - roi.xMin) / 100) * CANVAS_W;
    const rh = ((roi.yMax - roi.yMin) / 100) * CANVAS_H;

    // move & cull
    const alive: LiveObject[] = [];
    for (const o of objectsRef.current) {
      o.x += o.vx * dt;
      o.y += o.vy * dt;
      // wander a bit
      if (Math.random() < 0.04) o.vy += (Math.random() - 0.5) * 6;
      const inside =
        o.x + o.w / 2 >= rx &&
        o.x + o.w / 2 <= rx + rw &&
        o.y + o.h / 2 >= ry &&
        o.y + o.h / 2 <= ry + rh;
      o.inRoi = inside;
      if (inside && !o.enteredRoi) {
        o.enteredRoi = true;
        uniqueInRoiRef.current[o.cls].add(o.id);
      }
      if (
        o.x > CANVAS_W + 30 ||
        o.x + o.w < -30 ||
        o.y > CANVAS_H + 30 ||
        o.y + o.h < -30
      ) {
        continue;
      }
      alive.push(o);
    }

    // spawn
    for (const c of CLASSES) {
      const lambda = c.spawnRate * dt;
      if (Math.random() < lambda) {
        const [smin, smax] = c.size;
        const w = smin + Math.random() * (smax - smin);
        const h = c.size[0] === smin ? w * (smax / smin) * (0.6 + Math.random() * 0.4) : w;
        const fromLeft = Math.random() < 0.5;
        const speedRange = c.speed;
        const sp =
          (speedRange[0] + Math.random() * (speedRange[1] - speedRange[0])) *
          (CANVAS_W / 100);
        const vx = fromLeft ? sp : -sp;
        const vy = (Math.random() - 0.5) * 10;
        const y = ry + Math.random() * (rh + 60) - 30;
        const x = fromLeft ? -w - 5 : CANVAS_W + 5;
        const id = idRef.current++;
        totalUniqueRef.current[c.key].add(id);
        alive.push({
          id,
          cls: c.key,
          x,
          y,
          w,
          h,
          vx,
          vy,
          inRoi: false,
          enteredRoi: false,
        });
      }
    }

    objectsRef.current = alive;

    // counts in ROI now per class
    const counts: Partial<Record<ClassKey, number>> = {};
    for (const o of alive) {
      if (!o.inRoi) continue;
      counts[o.cls] = (counts[o.cls] ?? 0) + 1;
    }
    for (const c of CLASSES) {
      const v = counts[c.key] ?? 0;
      if (v > peakRef.current[c.key].peak) {
        peakRef.current[c.key] = { peak: v, frame: f };
      }
    }

    // history
    const point: HistoryPoint = { frame: f, t: +(f / FPS).toFixed(2) };
    for (const c of CLASSES) point[c.key] = counts[c.key] ?? 0;
    historyRef.current.push(point);

    // commit state (throttled)
    setFrame(f);
    setObjects(alive.slice());
    if (f % 4 === 0) setHistory(historyRef.current.slice());

    if (f >= TOTAL_FRAMES) {
      setRunning(false);
      setFinished(true);
      setHistory(historyRef.current.slice());
    }
  };

  // current counts in ROI from objects
  const liveCounts = useMemo(() => {
    const c: Partial<Record<ClassKey, number>> = {};
    for (const o of objects) if (o.inRoi) c[o.cls] = (c[o.cls] ?? 0) + 1;
    return c;
  }, [objects]);

  const progress = (frame / TOTAL_FRAMES) * 100;
  const tSeconds = frame / FPS;

  // --- Report ---
  const report = useMemo(() => {
    const per = selected.map((k) => {
      const def = CLASS_MAP[k];
      const uniqueRoi = uniqueInRoiRef.current[k]?.size ?? 0;
      const totalDetected = totalUniqueRef.current[k]?.size ?? 0;
      const peak = peakRef.current[k] ?? { peak: 0, frame: 0 };
      const entryRate = totalDetected ? (uniqueRoi / totalDetected) * 100 : 0;
      return {
        key: k,
        label: def.label,
        color: def.color,
        uniqueRoi,
        totalDetected,
        peak: peak.peak,
        peakAt: +(peak.frame / FPS).toFixed(1),
        entryRate: +entryRate.toFixed(1),
      };
    });
    return per;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished, selected, frame]);

  const insights = useMemo(() => {
    if (!finished) return null;
    const result: { kind: "info" | "warn" | "ok"; title: string; body: string }[] =
      [];
    // dominant
    const dom = [...report].sort((a, b) => b.uniqueRoi - a.uniqueRoi)[0];
    if (dom && dom.uniqueRoi > 0) {
      result.push({
        kind: "info",
        title: "Classe dominante",
        body: `${dom.label} liderou o tráfego na ROI com ${dom.uniqueRoi} indivíduos únicos.`,
      });
    }
    // person × cellphone correlation
    if (selected.includes("person") && selected.includes("cellphone")) {
      const ps = historyRef.current.map((p) => p.person ?? 0);
      const cs = historyRef.current.map((p) => p.cellphone ?? 0);
      const median = (a: number[]) => {
        const s = [...a].sort((x, y) => x - y);
        return s[Math.floor(s.length / 2)] ?? 0;
      };
      const pm = median(ps);
      const cm = median(cs);
      const both = ps.filter((v, i) => v > pm && cs[i] > cm).length;
      const ratio = ps.length ? (both / ps.length) * 100 : 0;
      if (ratio > 8) {
        result.push({
          kind: "warn",
          title: "Correlação Pessoa × Celular",
          body: `Em ${ratio.toFixed(1)}% dos frames houve pico simultâneo de pessoas e celulares — possível zona de espera ou uso intenso de dispositivos.`,
        });
      }
    }
    // baggage
    const bag = ["backpack", "handbag", "suitcase"].filter((k) =>
      selected.includes(k as ClassKey),
    ) as ClassKey[];
    if (bag.length && selected.includes("person")) {
      const personU = uniqueInRoiRef.current.person?.size ?? 0;
      const bagU = bag.reduce(
        (s, k) => s + (uniqueInRoiRef.current[k]?.size ?? 0),
        0,
      );
      const r = personU ? bagU / personU : 0;
      result.push({
        kind: r > 0.4 ? "warn" : "info",
        title: "Análise de bagagem",
        body: `Ratio bagagem/pessoa = ${r.toFixed(2)}. ${
          r > 0.4
            ? "Zona compatível com embarque/desembarque."
            : "Tráfego de pedestres sem alta carga."
        }`,
      });
    }
    // pedestrian × vehicle conflict
    const vehicles: ClassKey[] = ["car", "motorcycle", "bus", "truck", "bicycle"];
    const selVeh = vehicles.filter((v) => selected.includes(v));
    if (selected.includes("person") && selVeh.length) {
      const pU = uniqueInRoiRef.current.person?.size ?? 0;
      const vU = selVeh.reduce(
        (s, k) => s + (uniqueInRoiRef.current[k]?.size ?? 0),
        0,
      );
      const ratio = pU ? vU / pU : 0;
      const level =
        ratio > 0.7 ? "ALTO" : ratio > 0.25 ? "MODERADO" : "BAIXO";
      result.push({
        kind: ratio > 0.7 ? "warn" : ratio > 0.25 ? "info" : "ok",
        title: "Conflito Pedestre × Veículo",
        body: `Índice = ${ratio.toFixed(2)} → risco ${level}. ${
          ratio > 0.7
            ? "Avaliar sinalização, faixa segregada e redução de velocidade."
            : "Convivência dentro de parâmetros aceitáveis."
        }`,
      });
    }
    return result;
  }, [finished, report, selected]);

  const globalSummary = useMemo(() => {
    if (!finished) return null;
    const allCounts = historyRef.current.map((p) =>
      selected.reduce((s, k) => s + ((p[k] as number) ?? 0), 0),
    );
    const avg = allCounts.reduce((a, b) => a + b, 0) / Math.max(1, allCounts.length);
    let maxIdx = 0;
    for (let i = 1; i < allCounts.length; i++)
      if (allCounts[i] > allCounts[maxIdx]) maxIdx = i;
    return {
      avg: +avg.toFixed(2),
      peakSec: +(maxIdx / FPS).toFixed(1),
      peakVal: allCounts[maxIdx] ?? 0,
      durationSec: +(historyRef.current.length / FPS).toFixed(1),
    };
  }, [finished, selected]);

  const downloadCsv = () => {
    const header = [
      "frame",
      "second",
      ...selected.map((k) => `roi_${k}`),
      ...selected.map((k) => `cum_unique_${k}`),
    ];
    const totalByClass: Record<string, number> = {};
    for (const k of selected) totalByClass[k] = uniqueInRoiRef.current[k]?.size ?? 0;
    const n = historyRef.current.length;
    const lines = [header.join(",")];
    historyRef.current.forEach((p, i) => {
      const row = [
        p.frame,
        p.t,
        ...selected.map((k) => p[k] ?? 0),
        ...selected.map((k) =>
          Math.round(((i + 1) / n) * totalByClass[k]),
        ),
      ];
      lines.push(row.join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `visionaudit_${(fileName ?? "video").replace(/\..+$/, "")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen text-foreground">
      {/* Top bar */}
      <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-md bg-primary/15 text-primary glow-cyan">
              <Activity className="h-5 w-5" />
            </div>
            <div className="leading-tight">
              <div className="font-mono text-[11px] uppercase tracking-[0.25em] text-muted-foreground">
                IA · Auditoria Contínua
              </div>
              <h1 className="text-lg font-semibold tracking-tight">
                Vision<span className="text-primary text-glow">Audit</span>
              </h1>
            </div>
          </div>
          <div className="hidden items-center gap-2 md:flex">
            <Badge variant="outline" className="font-mono">
              <Cpu className="mr-1 h-3 w-3" /> YOLOv8 · Nano
            </Badge>
            <Badge variant="outline" className="font-mono">
              <Gauge className="mr-1 h-3 w-3" /> {FPS} fps
            </Badge>
            <Badge
              variant="outline"
              className={`font-mono ${running ? "border-primary text-primary" : ""}`}
            >
              ● {running ? "PROCESSANDO" : finished ? "CONCLUÍDO" : "IDLE"}
            </Badge>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1500px] grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-[320px_1fr]">
        {/* Sidebar */}
        <aside className="space-y-4">
          <SidebarCard title="1 · Upload do Vídeo" icon={<Upload className="h-4 w-4" />}>
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-border bg-secondary/40 px-3 py-6 text-center text-sm transition hover:border-primary hover:bg-secondary/60">
              <FileVideo className="mb-2 h-6 w-6 text-primary" />
              <span className="font-medium">Selecionar arquivo</span>
              <span className="text-xs text-muted-foreground">MP4 · AVI · MOV</span>
              <input
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
              />
            </label>
            {fileName && (
              <div className="mt-2 truncate rounded bg-secondary/60 px-2 py-1 font-mono text-xs text-muted-foreground">
                {fileName}
              </div>
            )}
          </SidebarCard>

          <SidebarCard title="2 · Alvos de Auditoria">
            <div className="grid grid-cols-2 gap-1.5">
              {CLASSES.map((c) => {
                const on = selected.includes(c.key);
                return (
                  <button
                    key={c.key}
                    onClick={() => toggleClass(c.key)}
                    className={`flex items-center gap-2 rounded-md border px-2 py-1.5 text-left text-xs transition ${
                      on
                        ? "border-primary/60 bg-primary/10"
                        : "border-border bg-secondary/30 hover:border-border/80"
                    }`}
                  >
                    <span
                      className="h-2.5 w-2.5 rounded-sm"
                      style={{ backgroundColor: c.color }}
                    />
                    <span className="truncate">{c.label}</span>
                  </button>
                );
              })}
            </div>
            <div className="mt-2 font-mono text-[10px] text-muted-foreground">
              {selected.length}/12 classes ativas
            </div>
          </SidebarCard>

          <SidebarCard title="3 · Zona de Auditoria (ROI)">
            <RoiSlider label="X mín" value={roi.xMin} onChange={(v) =>
              setRoi((r) => ({ ...r, xMin: Math.min(v, r.xMax - 5) }))} />
            <RoiSlider label="X máx" value={roi.xMax} onChange={(v) =>
              setRoi((r) => ({ ...r, xMax: Math.max(v, r.xMin + 5) }))} />
            <RoiSlider label="Y mín" value={roi.yMin} onChange={(v) =>
              setRoi((r) => ({ ...r, yMin: Math.min(v, r.yMax - 5) }))} />
            <RoiSlider label="Y máx" value={roi.yMax} onChange={(v) =>
              setRoi((r) => ({ ...r, yMax: Math.max(v, r.yMin + 5) }))} />
          </SidebarCard>

          <div className="flex gap-2">
            {!running ? (
              <Button
                className="flex-1 font-semibold"
                onClick={start}
                disabled={!fileName || selected.length === 0}
              >
                <Play className="mr-2 h-4 w-4" />
                {finished ? "Reanalisar" : "Iniciar Auditoria"}
              </Button>
            ) : (
              <Button variant="secondary" className="flex-1" onClick={stop}>
                <Pause className="mr-2 h-4 w-4" /> Pausar
              </Button>
            )}
            <Button
              variant="outline"
              onClick={() => {
                setRunning(false);
                resetState();
              }}
              title="Reiniciar"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
          {!modelLoaded && !running && (
            <p className="font-mono text-[11px] text-muted-foreground">
              Modelo YOLOv8n será carregado e mantido em cache na primeira execução.
            </p>
          )}
        </aside>

        {/* Main */}
        <main className="space-y-4">
          {!fileName && !running && !finished && <Welcome />}

          {(fileName || running || finished) && (
            <>
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_320px]">
                <Panel>
                  <PanelHeader
                    eyebrow="Feed ao vivo"
                    title="Detecção & Rastreamento"
                    right={
                      <span className="font-mono text-xs text-muted-foreground">
                        frame {frame.toString().padStart(4, "0")} / {TOTAL_FRAMES} ·{" "}
                        {fmtTime(tSeconds)}
                      </span>
                    }
                  />
                  <VideoCanvas
                    width={CANVAS_W}
                    height={CANVAS_H}
                    objects={objects}
                    roi={roi}
                    timestamp={fmtTime(tSeconds)}
                    counts={liveCounts}
                    selectedClasses={selected}
                    running={running}
                  />
                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-secondary">
                    <div
                      className="h-full bg-gradient-to-r from-primary to-accent transition-[width] duration-200"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <div className="mt-1 flex justify-between font-mono text-[11px] text-muted-foreground">
                    <span>{progress.toFixed(1)}% processado</span>
                    <span>{Math.round(TOTAL_FRAMES - frame)} frames restantes</span>
                  </div>
                </Panel>

                <Panel>
                  <PanelHeader eyebrow="Métricas" title="ROI · em tempo real" />
                  <div className="space-y-2">
                    {selected.map((k) => {
                      const d = CLASS_MAP[k];
                      const now = liveCounts[k] ?? 0;
                      const unique = uniqueInRoiRef.current[k]?.size ?? 0;
                      return (
                        <div
                          key={k}
                          className="rounded-md border border-border bg-secondary/30 p-3"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span
                                className="h-2.5 w-2.5 rounded-sm"
                                style={{ backgroundColor: d.color }}
                              />
                              <span className="text-sm font-medium">{d.label}</span>
                            </div>
                            <span className="font-mono text-[10px] uppercase text-muted-foreground">
                              IDs únicos
                            </span>
                          </div>
                          <div className="mt-1 flex items-end justify-between gap-3 font-mono">
                            <div>
                              <div className="text-[10px] text-muted-foreground">agora</div>
                              <div
                                className="text-2xl font-bold"
                                style={{ color: d.color }}
                              >
                                {now}
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-[10px] text-muted-foreground">
                                acumulado
                              </div>
                              <div className="text-2xl font-bold text-foreground">
                                {unique}
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Panel>
              </div>

              <Panel>
                <PanelHeader
                  eyebrow="Série temporal"
                  title="Ocupação da ROI ao longo do vídeo"
                />
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid stroke="rgba(148,163,184,0.12)" />
                      <XAxis
                        dataKey="t"
                        stroke="#94a3b8"
                        fontSize={11}
                        tickFormatter={(v) => `${v}s`}
                      />
                      <YAxis stroke="#94a3b8" fontSize={11} allowDecimals={false} />
                      <Tooltip
                        contentStyle={{
                          background: "#0b1322",
                          border: "1px solid #1f2a44",
                          fontFamily: "JetBrains Mono",
                          fontSize: 11,
                        }}
                        labelFormatter={(v) => `t = ${v}s`}
                      />
                      <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                      {selected.map((k) => (
                        <Line
                          key={k}
                          type="monotone"
                          dataKey={k}
                          name={CLASS_MAP[k].label}
                          stroke={CLASS_MAP[k].color}
                          dot={false}
                          strokeWidth={2}
                          isAnimationActive={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Panel>

              {finished && (
                <>
                  <Panel>
                    <PanelHeader
                      eyebrow="Relatório"
                      title="Métricas por classe"
                      right={
                        <Button size="sm" variant="outline" onClick={downloadCsv}>
                          <Download className="mr-2 h-4 w-4" /> Exportar CSV
                        </Button>
                      }
                    />
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                      {report.map((r) => (
                        <div
                          key={r.key}
                          className="rounded-lg border border-border bg-secondary/30 p-4"
                          style={{ borderTop: `3px solid ${r.color}` }}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-semibold">{r.label}</span>
                            <span className="font-mono text-[10px] text-muted-foreground">
                              {r.entryRate}% de entrada na ROI
                            </span>
                          </div>
                          <div className="mt-3 grid grid-cols-3 gap-2 font-mono">
                            <Stat label="únicos ROI" value={r.uniqueRoi} accent={r.color} />
                            <Stat label="pico" value={r.peak} />
                            <Stat label="pico em" value={`${r.peakAt}s`} />
                          </div>
                          <div className="mt-2 text-[11px] text-muted-foreground">
                            Total detectado no vídeo: <span className="text-foreground">{r.totalDetected}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Panel>

                  <Panel>
                    <PanelHeader
                      eyebrow="IA Heurística"
                      title="Correlações & Insights"
                    />
                    <div className="grid gap-3 md:grid-cols-2">
                      {insights?.map((ins, i) => (
                        <div
                          key={i}
                          className={`flex gap-3 rounded-md border p-3 ${
                            ins.kind === "warn"
                              ? "border-destructive/40 bg-destructive/10"
                              : ins.kind === "ok"
                                ? "border-primary/30 bg-primary/5"
                                : "border-border bg-secondary/30"
                          }`}
                        >
                          <div className="mt-0.5">
                            {ins.kind === "warn" ? (
                              <AlertTriangle className="h-4 w-4 text-destructive" />
                            ) : ins.kind === "ok" ? (
                              <Activity className="h-4 w-4 text-primary" />
                            ) : (
                              <Square className="h-4 w-4 text-muted-foreground" />
                            )}
                          </div>
                          <div>
                            <div className="text-sm font-semibold">{ins.title}</div>
                            <div className="mt-1 text-xs text-muted-foreground">
                              {ins.body}
                            </div>
                          </div>
                        </div>
                      ))}
                      {!insights?.length && (
                        <div className="text-sm text-muted-foreground">
                          Sem correlações relevantes para as classes selecionadas.
                        </div>
                      )}
                    </div>
                  </Panel>

                  {globalSummary && (
                    <Panel>
                      <PanelHeader
                        eyebrow="Resumo Global"
                        title="Visão consolidada da auditoria"
                      />
                      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 font-mono">
                        <Stat label="ocupação média" value={globalSummary.avg} />
                        <Stat
                          label="pico simultâneo"
                          value={globalSummary.peakVal}
                          accent="var(--accent)"
                        />
                        <Stat label="momento de pico" value={`${globalSummary.peakSec}s`} />
                        <Stat
                          label="duração auditada"
                          value={`${globalSummary.durationSec}s`}
                        />
                      </div>
                    </Panel>
                  )}
                </>
              )}
            </>
          )}
        </main>
      </div>

      <footer className="border-t border-border py-4">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-6 font-mono text-[11px] text-muted-foreground">
          <span>VisionAudit · processamento 100% local · sem upload externo</span>
          <span>v0.1 · demo</span>
        </div>
      </footer>
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────────── */

function SidebarCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-card/60 p-4 backdrop-blur">
      <div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}

function RoiSlider({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-3 last:mb-0">
      <div className="mb-1 flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>{label}</span>
        <span className="text-foreground">{value}%</span>
      </div>
      <Slider
        value={[value]}
        min={0}
        max={100}
        step={1}
        onValueChange={(v) => onChange(v[0])}
      />
    </div>
  );
}

function Panel({ children }: { children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-border bg-card/60 p-4 backdrop-blur">
      {children}
    </section>
  );
}

function PanelHeader({
  eyebrow,
  title,
  right,
}: {
  eyebrow: string;
  title: string;
  right?: React.ReactNode;
}) {
  return (
    <div className="mb-3 flex items-end justify-between gap-3">
      <div>
        <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-primary">
          {eyebrow}
        </div>
        <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      </div>
      {right}
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className="rounded-md bg-background/40 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="text-lg font-bold" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
    </div>
  );
}

function Welcome() {
  const features = [
    { t: "Detecção multi-classe", d: "12 categorias COCO simultâneas" },
    { t: "Rastreamento persistente", d: "IDs únicos frame a frame" },
    { t: "ROI configurável", d: "Delimite a zona exata da auditoria" },
    { t: "Dashboard ao vivo", d: "Métricas e gráfico em tempo real" },
    { t: "Heurísticas automáticas", d: "Correlações cruzadas entre classes" },
    { t: "Exportação CSV", d: "Pronto para Power BI / Tableau" },
  ];
  return (
    <section className="relative overflow-hidden rounded-xl border border-border bg-card/60 p-8 backdrop-blur">
      <div className="absolute inset-0 grid-bg opacity-30" />
      <div className="relative">
        <div className="font-mono text-[11px] uppercase tracking-[0.3em] text-primary">
          IA · Auditoria Visual Contínua
        </div>
        <h2 className="mt-2 max-w-2xl text-3xl font-bold leading-tight tracking-tight md:text-4xl">
          Mapeie processos em vídeo com IA{" "}
          <span className="text-primary text-glow">multi-classe</span>, frame a frame.
        </h2>
        <p className="mt-3 max-w-xl text-sm text-muted-foreground">
          Carregue um vídeo, defina a Região de Interesse e veja a plataforma
          detectar, rastrear e correlacionar pessoas, veículos, dispositivos e
          bagagens — sem enviar nada para a nuvem.
        </p>
        <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div
              key={f.t}
              className="rounded-md border border-border bg-secondary/30 p-3"
            >
              <div className="font-mono text-[10px] uppercase tracking-wider text-primary">
                ▸ feature
              </div>
              <div className="mt-1 text-sm font-semibold">{f.t}</div>
              <div className="text-xs text-muted-foreground">{f.d}</div>
            </div>
          ))}
        </div>
        <div className="mt-6 inline-flex items-center gap-2 rounded-md border border-dashed border-primary/40 bg-primary/5 px-3 py-2 font-mono text-xs text-primary">
          ← Selecione um vídeo no painel lateral para começar
        </div>
      </div>
    </section>
  );
}
