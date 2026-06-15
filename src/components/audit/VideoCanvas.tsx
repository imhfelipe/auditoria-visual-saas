import { useEffect, useRef } from "react";
import type { ClassKey } from "@/lib/audit-classes";
import { CLASS_MAP } from "@/lib/audit-classes";

export interface DetectedObject {
  id: number;
  cls: ClassKey;
  x: number;
  y: number;
  w: number;
  h: number;
  inRoi: boolean;
}

export interface Roi {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

interface Props {
  width: number;
  height: number;
  objects: DetectedObject[];
  roi: Roi;
  timestamp: string;
  counts: Partial<Record<ClassKey, number>>;
  selectedClasses: ClassKey[];
  running: boolean;
}

export function VideoCanvas({
  width,
  height,
  objects,
  roi,
  timestamp,
  counts,
  selectedClasses,
  running,
}: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const cvs = ref.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;

    // Background: simulated surveillance scene
    const grad = ctx.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, "#0b1322");
    grad.addColorStop(1, "#040810");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    // ground perspective lines
    ctx.strokeStyle = "rgba(34,211,238,0.08)";
    ctx.lineWidth = 1;
    const vpX = width / 2;
    const vpY = height * 0.45;
    for (let i = 0; i < 14; i++) {
      const x = (i / 13) * width;
      ctx.beginPath();
      ctx.moveTo(x, height);
      ctx.lineTo(vpX, vpY);
      ctx.stroke();
    }
    for (let i = 1; i < 8; i++) {
      const y = vpY + ((height - vpY) * i) / 7;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // ROI rect in px
    const rx = (roi.xMin / 100) * width;
    const ry = (roi.yMin / 100) * height;
    const rw = ((roi.xMax - roi.xMin) / 100) * width;
    const rh = ((roi.yMax - roi.yMin) / 100) * height;

    // darken outside ROI
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(0, 0, width, ry);
    ctx.fillRect(0, ry + rh, width, height - (ry + rh));
    ctx.fillRect(0, ry, rx, rh);
    ctx.fillRect(rx + rw, ry, width - (rx + rw), rh);

    // ROI border
    ctx.strokeStyle = "#22d3ee";
    ctx.setLineDash([6, 6]);
    ctx.lineWidth = 2;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
    ctx.fillStyle = "#22d3ee";
    ctx.font = "600 11px JetBrains Mono, monospace";
    ctx.fillText("ROI", rx + 6, ry + 14);

    // Detections
    for (const o of objects) {
      const def = CLASS_MAP[o.cls];
      ctx.lineWidth = 2;
      ctx.strokeStyle = def.color;
      ctx.strokeRect(o.x, o.y, o.w, o.h);
      // corner accents
      const c = 6;
      ctx.beginPath();
      ctx.moveTo(o.x, o.y + c);
      ctx.lineTo(o.x, o.y);
      ctx.lineTo(o.x + c, o.y);
      ctx.moveTo(o.x + o.w - c, o.y);
      ctx.lineTo(o.x + o.w, o.y);
      ctx.lineTo(o.x + o.w, o.y + c);
      ctx.stroke();

      // label
      const tag = `${def.label} #${o.id}${o.inRoi ? " [ROI]" : ""}`;
      ctx.font = "600 10px JetBrains Mono, monospace";
      const tw = ctx.measureText(tag).width + 8;
      ctx.fillStyle = def.color;
      ctx.fillRect(o.x, o.y - 14, tw, 14);
      ctx.fillStyle = "#0a0e1a";
      ctx.fillText(tag, o.x + 4, o.y - 4);
    }

    // HUD top bar
    ctx.fillStyle = "rgba(2,6,15,0.85)";
    ctx.fillRect(0, 0, width, 28);
    ctx.strokeStyle = "rgba(34,211,238,0.4)";
    ctx.beginPath();
    ctx.moveTo(0, 28);
    ctx.lineTo(width, 28);
    ctx.stroke();
    ctx.fillStyle = "#22d3ee";
    ctx.font = "600 12px JetBrains Mono, monospace";
    ctx.fillText(`● AUDIT  ${timestamp}`, 10, 18);
    let cx = 160;
    for (const k of selectedClasses) {
      const d = CLASS_MAP[k];
      const v = counts[k] ?? 0;
      ctx.fillStyle = d.color;
      ctx.fillRect(cx, 9, 8, 10);
      ctx.fillStyle = "#e2e8f0";
      const txt = `${d.label}: ${v}`;
      ctx.fillText(txt, cx + 12, 18);
      cx += ctx.measureText(txt).width + 32;
    }

    // recording dot
    if (running) {
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(width - 18, 14, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fecaca";
      ctx.font = "600 11px JetBrains Mono, monospace";
      ctx.fillText("REC", width - 60, 18);
    }
  }, [width, height, objects, roi, timestamp, counts, selectedClasses, running]);

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      className="w-full h-auto rounded-lg border border-border bg-black"
    />
  );
}
