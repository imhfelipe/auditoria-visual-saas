export type ClassKey =
  | "person"
  | "car"
  | "motorcycle"
  | "bus"
  | "truck"
  | "bicycle"
  | "backpack"
  | "handbag"
  | "suitcase"
  | "cellphone"
  | "laptop"
  | "bottle";

export interface ClassDef {
  key: ClassKey;
  label: string;
  color: string; // hex
  // base spawn rate per second
  spawnRate: number;
  // approx speed in % of canvas / second
  speed: [number, number];
  // size in canvas px (assume 800x450)
  size: [number, number];
}

export const CLASSES: ClassDef[] = [
  { key: "person", label: "Pessoa", color: "#22d3ee", spawnRate: 1.6, speed: [6, 14], size: [34, 70] },
  { key: "car", label: "Carro", color: "#f59e0b", spawnRate: 0.9, speed: [18, 32], size: [90, 50] },
  { key: "motorcycle", label: "Moto", color: "#fb923c", spawnRate: 0.5, speed: [22, 36], size: [60, 44] },
  { key: "bus", label: "Ônibus", color: "#facc15", spawnRate: 0.25, speed: [14, 22], size: [130, 70] },
  { key: "truck", label: "Caminhão", color: "#ef4444", spawnRate: 0.3, speed: [12, 20], size: [140, 72] },
  { key: "bicycle", label: "Bicicleta", color: "#a3e635", spawnRate: 0.45, speed: [10, 20], size: [60, 50] },
  { key: "backpack", label: "Mochila", color: "#a78bfa", spawnRate: 0.6, speed: [6, 14], size: [26, 32] },
  { key: "handbag", label: "Bolsa", color: "#f472b6", spawnRate: 0.4, speed: [6, 14], size: [22, 26] },
  { key: "suitcase", label: "Mala", color: "#fb7185", spawnRate: 0.2, speed: [4, 10], size: [30, 36] },
  { key: "cellphone", label: "Celular", color: "#34d399", spawnRate: 1.1, speed: [6, 14], size: [14, 22] },
  { key: "laptop", label: "Laptop", color: "#38bdf8", spawnRate: 0.3, speed: [4, 10], size: [38, 26] },
  { key: "bottle", label: "Garrafa", color: "#14b8a6", spawnRate: 0.4, speed: [4, 10], size: [14, 28] },
];

export const CLASS_MAP: Record<ClassKey, ClassDef> = Object.fromEntries(
  CLASSES.map((c) => [c.key, c]),
) as Record<ClassKey, ClassDef>;
