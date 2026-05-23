# ============================================================================
# 🏭 Plataforma SaaS de Auditoria Visual Contínua e Mapeamento de Processos
# Motor de IA Multi-Classe com Rastreamento de Artefatos
# ============================================================================
#
# 📦 INSTALAÇÃO (execute no terminal):
#   pip install streamlit ultralytics opencv-python pandas numpy
#
# 🚀 EXECUÇÃO:
#   python -m streamlit run app.py
#
# ============================================================================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import time
import gc
from datetime import timedelta
from collections import defaultdict


# ── Configuração da Página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Auditoria Visual Contínua • SaaS Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════
# 🧠 DICIONÁRIO DE CLASSES — O Cérebro da Auditoria
# Mapeia IDs COCO → Nome PT-BR + Cor RGB (para desenho OpenCV em BGR)
# ════════════════════════════════════════════════════════════════════════════
MAPA_CLASSES: dict[int, dict] = {
    0:  {"nome": "Pessoa",      "cor_rgb": (34, 197, 94),   "cor_bgr": (94, 197, 34),   "icone": "🚶"},
    1:  {"nome": "Bicicleta",   "cor_rgb": (14, 165, 233),  "cor_bgr": (233, 165, 14),  "icone": "🚲"},
    2:  {"nome": "Carro",       "cor_rgb": (59, 130, 246),  "cor_bgr": (246, 130, 59),  "icone": "🚗"},
    3:  {"nome": "Moto",        "cor_rgb": (249, 115, 22),  "cor_bgr": (22, 115, 249),  "icone": "🏍️"},
    5:  {"nome": "Ônibus",      "cor_rgb": (168, 85, 247),  "cor_bgr": (247, 85, 168),  "icone": "🚌"},
    7:  {"nome": "Caminhão",    "cor_rgb": (236, 72, 153),  "cor_bgr": (153, 72, 236),  "icone": "🚛"},
    16: {"nome": "Cachorro",    "cor_rgb": (245, 158, 11),  "cor_bgr": (11, 158, 245),  "icone": "🐕"},
    24: {"nome": "Mochila",     "cor_rgb": (6, 182, 212),   "cor_bgr": (212, 182, 6),   "icone": "🎒"},
    26: {"nome": "Bolsa",       "cor_rgb": (20, 184, 166),  "cor_bgr": (166, 184, 20),  "icone": "👜"},
    28: {"nome": "Mala",        "cor_rgb": (132, 204, 22),  "cor_bgr": (22, 204, 132),  "icone": "🧳"},
    63: {"nome": "Notebook",    "cor_rgb": (234, 179, 8),   "cor_bgr": (8, 179, 234),   "icone": "💻"},
    67: {"nome": "Celular",     "cor_rgb": (239, 68, 68),   "cor_bgr": (68, 68, 239),   "icone": "📱"},
}

# Mapeamento reverso: nome → id
NOME_PARA_ID: dict[str, int] = {v["nome"]: k for k, v in MAPA_CLASSES.items()}


# ── CSS Premium ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base dark theme */
    .stApp {
        background: linear-gradient(160deg, #080c14 0%, #0f172a 40%, #1e1b4b 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1222 0%, #111827 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    /* Metric glassmorphism cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.07), rgba(168,85,247,0.07));
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 14px;
        padding: 18px 22px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-weight: 700 !important;
    }

    /* Gradient text headings */
    h1 { background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 800 !important; }
    h2 { background: linear-gradient(90deg, #60a5fa, #818cf8, #a78bfa);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 700 !important; }
    h3, h4 { color: #c4b5fd !important; font-weight: 600 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: #fff !important; border: none !important;
        border-radius: 12px !important; padding: 0.65rem 1.6rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 18px rgba(79,70,229,0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(79,70,229,0.5) !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669, #10b981) !important;
        color: #fff !important; border: none !important;
        border-radius: 12px !important; font-weight: 600 !important;
        box-shadow: 0 4px 18px rgba(5,150,105,0.3) !important;
    }

    div[data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: rgba(99, 102, 241, 0.12) !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 🤖 CARREGAMENTO DO MODELO (com cache)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def carregar_modelo():
    """Carrega o YOLOv8 Nano com cache persistente para performance."""
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


# ════════════════════════════════════════════════════════════════════════════
# 🎨 FUNÇÕES DE DESENHO E PROCESSAMENTO
# ════════════════════════════════════════════════════════════════════════════

def desenhar_roi(frame: np.ndarray, x1: int, y1: int,
                 x2: int, y2: int) -> np.ndarray:
    """Desenha a zona de auditoria com overlay translúcido e bordas estilizadas."""
    h, w = frame.shape[:2]
    # Overlay escurecido FORA da ROI para destaque
    mask = np.zeros_like(frame, dtype=np.uint8)
    mask[:] = (0, 0, 0)
    mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    # Escurecer exterior
    exterior = cv2.addWeighted(frame, 0.35, np.zeros_like(frame), 0.65, 0)
    exterior[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    frame[:] = exterior

    # Bordas da ROI com cantos arredondados simulados
    cor_roi = (147, 130, 99)  # Indigo em BGR (#6366f1)
    espessura = 2
    canto = 25

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + canto, y1), cor_roi, espessura)
    cv2.line(frame, (x1, y1), (x1, y1 + canto), cor_roi, espessura)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - canto, y1), cor_roi, espessura)
    cv2.line(frame, (x2, y1), (x2, y1 + canto), cor_roi, espessura)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + canto, y2), cor_roi, espessura)
    cv2.line(frame, (x1, y2), (x1, y2 - canto), cor_roi, espessura)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - canto, y2), cor_roi, espessura)
    cv2.line(frame, (x2, y2), (x2, y2 - canto), cor_roi, espessura)

    # Label
    cv2.putText(frame, "ZONA DE AUDITORIA", (x1 + 8, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_roi, 1, cv2.LINE_AA)

    return frame


def desenhar_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  id_obj: int, nome_classe: str, cor_bgr: tuple,
                  dentro_roi: bool) -> np.ndarray:
    """Desenha bounding box com cor da classe, ID e indicador de ROI."""
    # Se fora da ROI, dessaturar a cor (mais opaco)
    if dentro_roi:
        cor = cor_bgr
        esp = 2
    else:
        cor = tuple(int(c * 0.45) for c in cor_bgr)
        esp = 1

    # Cantos estilizados
    l = 18
    cv2.line(frame, (x1, y1), (x1 + l, y1), cor, esp + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + l), cor, esp + 1)
    cv2.line(frame, (x2, y1), (x2 - l, y1), cor, esp + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + l), cor, esp + 1)
    cv2.line(frame, (x1, y2), (x1 + l, y2), cor, esp + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - l), cor, esp + 1)
    cv2.line(frame, (x2, y2), (x2 - l, y2), cor, esp + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - l), cor, esp + 1)

    # Retângulo fino
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 1)

    # Label de classe + ID
    label = f"{nome_classe}"
    if id_obj >= 0:
        label += f" #{id_obj}"
    if dentro_roi:
        label += " [ROI]"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), cor, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Ponto de base (pés / rodas)
    cx = int((x1 + x2) / 2)
    cy = y2
    cv2.circle(frame, (cx, cy), 3, cor, -1)

    return frame


def adicionar_hud(frame: np.ndarray, contagens_roi: dict[str, int],
                  frame_idx: int, fps: float) -> np.ndarray:
    """Sobrepõe HUD profissional com contagens por classe e timestamp."""
    h, w = frame.shape[:2]

    # Barra superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (8, 12, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, 48), (w, 48), (99, 102, 241), 1)

    # Timestamp
    tempo = timedelta(seconds=int(frame_idx / max(fps, 1)))
    cv2.putText(frame, f"T: {tempo}", (14, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)

    # Contagens compactas à direita
    offset_x = w - 14
    for nome, qtd in reversed(list(contagens_roi.items())):
        if nome in NOME_PARA_ID:
            info = MAPA_CLASSES[NOME_PARA_ID[nome]]
            texto = f"{info['icone']} {nome}: {qtd}"
        else:
            texto = f"{nome}: {qtd}"
        (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        offset_x -= (tw + 20)
        cv2.putText(frame, texto, (offset_x, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 240), 1, cv2.LINE_AA)

    return frame


def formatar_tempo(segundos: float) -> str:
    """Formata segundos em HH:MM:SS."""
    return str(timedelta(seconds=int(segundos)))


def gerar_legenda_classes(classes_selecionadas: list[str]) -> str:
    """Gera HTML da legenda de cores para as classes ativas."""
    items = []
    for nome in classes_selecionadas:
        if nome in NOME_PARA_ID:
            info = MAPA_CLASSES[NOME_PARA_ID[nome]]
            r, g, b = info["cor_rgb"]
            items.append(
                f'<span style="display:inline-flex;align-items:center;gap:6px;'
                f'margin-right:16px;">'
                f'<span style="width:12px;height:12px;border-radius:3px;'
                f'background:rgb({r},{g},{b});display:inline-block;"></span>'
                f'<span style="color:#cbd5e1;font-size:0.85rem;">'
                f'{info["icone"]} {nome}</span></span>'
            )
    return " ".join(items)


# ════════════════════════════════════════════════════════════════════════════
# 🏭 INTERFACE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:12px 0 4px 0;">
        <h1 style="font-size:2.3rem; margin-bottom:0;">
            🏭 Auditoria Visual Contínua
        </h1>
        <p style="color:#64748b; font-size:1rem; margin-top:4px;">
            Plataforma SaaS de Mapeamento de Processos com Inteligência Artificial Multi-Classe
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Sidebar — Painel de Missão ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📁 Fonte de Dados")
        arquivo_video = st.file_uploader(
            "Carregar vídeo para auditoria",
            type=["mp4", "avi", "mov"],
            help="Formatos suportados: MP4, AVI, MOV"
        )

        st.divider()

        # ── Seleção Dinâmica de Alvos ───────────────────────────────────────
        st.markdown("### 🎯 Alvos de Auditoria")
        nomes_disponiveis = [v["nome"] for v in MAPA_CLASSES.values()]
        classes_selecionadas = st.multiselect(
            "Selecione as classes para rastreamento",
            options=nomes_disponiveis,
            default=["Pessoa", "Carro"],
            help="Escolha quais tipos de objetos o sistema deve detectar e rastrear."
        )

        # Converter nomes → IDs COCO numéricos
        ids_selecionados = [NOME_PARA_ID[n] for n in classes_selecionadas
                           if n in NOME_PARA_ID]

        # Legenda visual
        if classes_selecionadas:
            st.markdown(
                gerar_legenda_classes(classes_selecionadas),
                unsafe_allow_html=True,
            )

        st.divider()

        # ── ROI ─────────────────────────────────────────────────────────────
        st.markdown("### 🔲 Zona de Auditoria (ROI)")
        st.caption("Delimite a região de interesse em % do frame.")
        c1, c2 = st.columns(2)
        with c1:
            roi_x_min = st.slider("X Mínimo", 0, 100, 5, format="%d%%")
            roi_y_min = st.slider("Y Mínimo", 0, 100, 5, format="%d%%")
        with c2:
            roi_x_max = st.slider("X Máximo", 0, 100, 95, format="%d%%")
            roi_y_max = st.slider("Y Máximo", 0, 100, 95, format="%d%%")

        st.divider()

        # ── Ação ────────────────────────────────────────────────────────────
        iniciar = st.button("▶️ Iniciar Auditoria Visual",
                            width="stretch", type="primary")

        st.divider()
        st.markdown("""
        <div style="text-align:center;color:#475569;font-size:0.72rem;padding:6px;">
            <strong>Auditoria Visual Contínua</strong> v2.0<br>
            YOLOv8 Nano • Multi-Classe • Tracking IDs
        </div>
        """, unsafe_allow_html=True)

    # ── Tela de espera ──────────────────────────────────────────────────────
    if not iniciar:
        st.markdown("""
        <div style="text-align:center;padding:55px 20px;margin:25px auto;
                    max-width:750px;
                    background:linear-gradient(135deg,
                        rgba(99,102,241,0.05), rgba(168,85,247,0.05));
                    border:1px solid rgba(99,102,241,0.12);
                    border-radius:22px;">
            <div style="font-size:3.8rem;margin-bottom:14px;">🏭</div>
            <h2 style="font-size:1.5rem;margin-bottom:10px;">
                Plataforma Pronta para Auditoria
            </h2>
            <p style="color:#94a3b8;font-size:1rem;line-height:1.6;max-width:520px;margin:0 auto;">
                Configure os alvos de auditoria, delimite a zona de interesse
                e inicie o processamento analítico multi-classe.
            </p>
            <div style="display:flex;justify-content:center;gap:36px;margin-top:28px;">
                <div><div style="font-size:1.4rem;">🎯</div>
                     <div style="color:#64748b;font-size:0.78rem;margin-top:3px;">Multi-Classe</div></div>
                <div><div style="font-size:1.4rem;">🔍</div>
                     <div style="color:#64748b;font-size:0.78rem;margin-top:3px;">Tracking IDs</div></div>
                <div><div style="font-size:1.4rem;">📊</div>
                     <div style="color:#64748b;font-size:0.78rem;margin-top:3px;">BI Insights</div></div>
                <div><div style="font-size:1.4rem;">📦</div>
                     <div style="color:#64748b;font-size:0.78rem;margin-top:3px;">Export CSV</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Validações ──────────────────────────────────────────────────────────
    if arquivo_video is None:
        st.warning("⚠️ Nenhum vídeo carregado. Faça upload na barra lateral.")
        return
    if not classes_selecionadas:
        st.warning("⚠️ Selecione ao menos uma classe de auditoria.")
        return
    if roi_x_min >= roi_x_max or roi_y_min >= roi_y_max:
        st.error("❌ ROI inválida: X Mín < X Máx e Y Mín < Y Máx.")
        return

    # ── Salvar vídeo temporário ─────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(arquivo_video.read())
        caminho_video = tmp.name

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        st.error("❌ Falha ao abrir o vídeo. Arquivo corrompido?")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duracao = total_frames / fps

    # ROI em pixels
    rx1 = int(largura * roi_x_min / 100)
    rx2 = int(largura * roi_x_max / 100)
    ry1 = int(altura * roi_y_min / 100)
    ry2 = int(altura * roi_y_max / 100)

    # ── Carregar modelo ─────────────────────────────────────────────────────
    with st.spinner("🤖 Carregando YOLOv8 Nano..."):
        modelo = carregar_modelo()

    # ── Info do vídeo ───────────────────────────────────────────────────────
    cols_info = st.columns(4)
    cols_info[0].metric("📐 Resolução", f"{largura}×{altura}")
    cols_info[1].metric("🎞️ FPS", f"{fps:.1f}")
    cols_info[2].metric("⏱️ Duração", formatar_tempo(duracao))
    cols_info[3].metric("🎯 Classes Ativas", f"{len(classes_selecionadas)}")
    st.divider()

    # ── Layout de processamento ─────────────────────────────────────────────
    col_video, col_metricas = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown("#### 🎥 Feed de Auditoria ao Vivo")
        ph_frame = st.empty()

    with col_metricas:
        st.markdown("#### 📊 Painel de Rastreamento")
        # Métricas dinâmicas por classe (criar placeholders)
        n_classes = len(classes_selecionadas)
        cols_metric = st.columns(min(n_classes, 3))
        ph_metrics: dict[str, any] = {}
        for i, nome in enumerate(classes_selecionadas):
            info = MAPA_CLASSES.get(NOME_PARA_ID.get(nome, -1), {})
            icone = info.get("icone", "📌")
            with cols_metric[i % min(n_classes, 3)]:
                ph_metrics[nome] = st.empty()

        st.markdown("---")
        st.markdown("##### 📈 Ocupação Instantânea na ROI")
        ph_grafico = st.empty()
        ph_alerta = st.empty()

    barra = st.progress(0, text="Preparando auditoria...")

    # ════════════════════════════════════════════════════════════════════════
    # 📦 ESTRUTURAS DE DADOS DE RASTREAMENTO
    # ════════════════════════════════════════════════════════════════════════

    # Set de IDs únicos por classe (nunca conta duas vezes)
    unicos_por_classe: dict[str, set] = {n: set() for n in classes_selecionadas}

    # Todos os IDs detectados no vídeo (dentro ou fora da ROI), por classe
    todos_detectados: dict[str, set] = {n: set() for n in classes_selecionadas}

    # Histórico temporal: lista de dicts para DataFrame
    historico: list[dict] = []

    # Picos por classe
    picos: dict[str, dict] = {n: {"valor": 0, "segundo": 0.0}
                               for n in classes_selecionadas}

    # ════════════════════════════════════════════════════════════════════════
    # 🔄 LOOP DE PROCESSAMENTO
    # ════════════════════════════════════════════════════════════════════════
    frame_idx = 0

    # Pré-computar cores fora do loop (evita recalcular a cada frame)
    colunas_roi = [f"{n} (ROI)" for n in classes_selecionadas]
    cores_hex = []
    for n in classes_selecionadas:
        rgb = MAPA_CLASSES.get(NOME_PARA_ID.get(n, -1), {}).get(
            "cor_rgb", (200, 200, 200))
        cores_hex.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        seg_atual = frame_idx / fps

        # ── Inferência com rastreamento ─────────────────────────────────────
        resultados = modelo.track(
            frame,
            persist=True,
            classes=ids_selecionados,
            verbose=False,
            conf=0.45,
            imgsz=640
        )[0]

        # Contagem instantânea por classe neste frame
        contagem_frame: dict[str, int] = {n: 0 for n in classes_selecionadas}

        if resultados.boxes is not None and len(resultados.boxes) > 0:
            boxes = resultados.boxes
            ids_disponiveis = boxes.id is not None

            for i in range(len(boxes)):
                if not ids_disponiveis:
                    continue
                try:
                    id_obj = int(boxes.id[i])
                except (IndexError, TypeError, AttributeError):
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]),
                                   int(xyxy[2]), int(xyxy[3]))

                cls_id = int(boxes.cls[i].cpu().item())
                if cls_id not in MAPA_CLASSES:
                    continue
                info_cls = MAPA_CLASSES[cls_id]
                nome_cls = info_cls["nome"]
                if nome_cls not in classes_selecionadas:
                    continue

                todos_detectados[nome_cls].add(id_obj)
                cx = int((x1 + x2) / 2)
                cy = y2
                dentro_roi = (rx1 <= cx <= rx2 and ry1 <= cy <= ry2)

                if dentro_roi:
                    contagem_frame[nome_cls] += 1
                    unicos_por_classe[nome_cls].add(id_obj)

                frame = desenhar_bbox(frame, x1, y1, x2, y2,
                                      id_obj, nome_cls, info_cls["cor_bgr"],
                                      dentro_roi)

        # ── Desenhar ROI + HUD ──────────────────────────────────────────────
        frame = desenhar_roi(frame, rx1, ry1, rx2, ry2)
        frame = adicionar_hud(frame, contagem_frame, frame_idx, fps)

        # ── Registrar picos ─────────────────────────────────────────────────
        for nome, qtd in contagem_frame.items():
            if qtd > picos[nome]["valor"]:
                picos[nome]["valor"] = qtd
                picos[nome]["segundo"] = seg_atual

        # ── BLINDAGEM 3: Garbage Collector a cada 30 frames ─────────────
        if frame_idx % 30 == 0:
            gc.collect()

        # ── Histórico temporal ──────────────────────────────────────────────
        registro = {
            "frame": frame_idx,
            "segundo": round(seg_atual, 2),
            "timestamp": formatar_tempo(seg_atual),
        }
        for nome in classes_selecionadas:
            registro[f"{nome} (ROI)"] = contagem_frame[nome]
            registro[f"{nome} (Únicos Acum.)"] = len(unicos_por_classe[nome])
        historico.append(registro)

        # ════════════════════════════════════════════════════════════════════
        # 🖥️ ATUALIZAÇÃO DA UI — ESTRATÉGIA ESCALONADA
        #   Frame + Métricas: a cada 5 frames (leve)
        #   Gráfico: a cada 20 frames (pesado — serializa DataFrame)
        #   Progresso: a cada 10 frames
        # ════════════════════════════════════════════════════════════════════
        is_ultimo = (frame_idx >= total_frames)

        # ── Frame do vídeo (a cada 5 frames) ────────────────────────────
        if frame_idx % 5 == 0 or is_ultimo:
            h_f, w_f = frame.shape[:2]
            if w_f > 960:
                escala = 960 / w_f
                frame_small = cv2.resize(
                    frame, (960, int(h_f * escala)),
                    interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            ph_frame.image(frame_rgb, channels="RGB", width="stretch")

        # ── Métricas por classe (a cada 5 frames) ──────────────────────
        if frame_idx % 5 == 0 or is_ultimo:
            for nome in classes_selecionadas:
                info = MAPA_CLASSES.get(NOME_PARA_ID.get(nome, -1), {})
                icone = info.get("icone", "📌")
                ph_metrics[nome].metric(
                    f"{icone} {nome}",
                    f"{contagem_frame[nome]} na ROI",
                    delta=f"{len(unicos_por_classe[nome])} únicos",
                )

        # ── Gráfico multi-série (a cada 20 frames — PESADO) ────────────
        if frame_idx % 20 == 0 or is_ultimo:
            df_parcial = pd.DataFrame(historico)
            if len(df_parcial) > 120:
                step = len(df_parcial) // 120
                df_chart = df_parcial.iloc[::step]
            else:
                df_chart = df_parcial
            ph_grafico.line_chart(
                df_chart.set_index("segundo")[colunas_roi],
                color=cores_hex,
                width="stretch",
            )
            del df_parcial, df_chart

        # ── Barra de progresso (a cada 10 frames) ──────────────────────
        if frame_idx % 10 == 0 or is_ultimo:
            prog = min(frame_idx / max(total_frames, 1), 1.0)
            barra.progress(
                prog,
                text=f"Auditando: {frame_idx:,}/{total_frames:,} frames "
                     f"({prog * 100:.1f}%)"

            )

    # ── Limpeza ─────────────────────────────────────────────────────────────
    cap.release()
    try:
        os.unlink(caminho_video)
    except OSError:
        pass

    barra.empty()
    ph_frame.empty()
    ph_alerta.empty()

    # ════════════════════════════════════════════════════════════════════════
    # 📋 RELATÓRIO DE AUDITORIA E MELHORIA CONTÍNUA
    # ════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("""
    <div style="text-align:center;padding:18px 0 8px 0;">
        <h1 style="font-size:2rem;">📋 Relatório de Auditoria Visual</h1>
        <p style="color:#64748b;">Insights consolidados de mapeamento de processos</p>
    </div>
    """, unsafe_allow_html=True)

    df_final = pd.DataFrame(historico)

    # ── Métricas de Ocupação: cards comparativos ────────────────────────────
    st.markdown("### 📊 Métricas de Ocupação por Classe")

    cols_final = st.columns(len(classes_selecionadas))
    ranking_volume: list[tuple[str, int]] = []

    for i, nome in enumerate(classes_selecionadas):
        total_unicos = len(unicos_por_classe[nome])
        total_det = len(todos_detectados[nome])
        pico_val = picos[nome]["valor"]
        pico_seg = picos[nome]["segundo"]
        ranking_volume.append((nome, total_unicos))

        info = MAPA_CLASSES.get(NOME_PARA_ID.get(nome, -1), {})
        icone = info.get("icone", "📌")

        with cols_final[i]:
            st.metric(f"{icone} {nome} — Únicos na ROI", total_unicos)
            st.metric(f"📈 Pico Simultâneo", f"{pico_val}")
            st.caption(f"Pico em {formatar_tempo(pico_seg)} "
                       f"({pico_seg:.1f}s)")
            if total_det > 0:
                taxa = (total_unicos / total_det) * 100
                st.caption(f"🎯 Taxa de entrada na ROI: {taxa:.1f}%")

    st.divider()

    # ── Gráfico Dinâmico Multi-Classe ───────────────────────────────────────
    st.markdown("### 📈 Evolução Temporal Comparativa")
    if len(df_final) > 0:
        colunas_chart = [f"{n} (ROI)" for n in classes_selecionadas]
        cores_chart = []
        for n in classes_selecionadas:
            rgb = MAPA_CLASSES.get(NOME_PARA_ID.get(n, -1), {}).get(
                "cor_rgb", (200, 200, 200))
            cores_chart.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

        st.line_chart(
            df_final.set_index("segundo")[colunas_chart],
            color=cores_chart,
            width="stretch",
        )

    st.divider()

    # ── Análise Heurística e Inteligência de Negócio ────────────────────────
    st.markdown("### 🧠 Análise Heurística e Insights")

    # Ranking de volume
    ranking_volume.sort(key=lambda x: x[1], reverse=True)
    if ranking_volume and ranking_volume[0][1] > 0:
        lider = ranking_volume[0]
        info_lider = MAPA_CLASSES.get(NOME_PARA_ID.get(lider[0], -1), {})
        st.info(
            f"🏆 **Classe Dominante na Zona de Auditoria**\n\n"
            f"A classe **{info_lider.get('icone', '')} {lider[0]}** registrou o "
            f"maior volume de tráfego com **{lider[1]} objetos únicos** "
            f"transitando pela ROI. "
            f"{'As demais classes em ordem: ' + ', '.join(f'{n} ({v})' for n, v in ranking_volume[1:] if v > 0) + '.' if len([x for x in ranking_volume[1:] if x[1] > 0]) > 0 else ''}"
        )

    # Correlação Pessoa + Celular
    if "Pessoa" in classes_selecionadas and "Celular" in classes_selecionadas:
        if len(df_final) > 0:
            col_pessoa = f"Pessoa (ROI)"
            col_celular = f"Celular (ROI)"
            # Detectar picos simultâneos (ambos acima da mediana)
            med_p = df_final[col_pessoa].median()
            med_c = df_final[col_celular].median()
            picos_simult = df_final[
                (df_final[col_pessoa] > med_p) & (df_final[col_celular] > med_c)
            ]
            if len(picos_simult) > 0:
                pct_correl = (len(picos_simult) / len(df_final)) * 100
                st.warning(
                    f"📱 **Correlação Detectada: Pessoa × Celular**\n\n"
                    f"Em **{pct_correl:.1f}%** dos frames analisados, houve "
                    f"picos simultâneos de pessoas e celulares acima da mediana "
                    f"na zona de auditoria. Isso pode indicar:\n"
                    f"- Área de uso intenso de dispositivos móveis\n"
                    f"- Ponto de espera / fila (pessoas paradas com celular)\n"
                    f"- Zona de interação digital que requer atenção ergonômica"
                )
            else:
                st.info(
                    f"📱 **Análise: Pessoa × Celular**\n\n"
                    f"Não foram detectados picos simultâneos significativos entre "
                    f"pessoas e celulares na zona de auditoria. O uso de "
                    f"dispositivos parece disperso e não correlacionado."
                )

    # Correlação Pessoa + Mochila/Bolsa
    itens_pessoais = [c for c in ["Mochila", "Bolsa", "Mala"]
                      if c in classes_selecionadas]
    if "Pessoa" in classes_selecionadas and itens_pessoais:
        total_p = len(unicos_por_classe["Pessoa"])
        total_itens = sum(len(unicos_por_classe[c]) for c in itens_pessoais)
        if total_p > 0:
            ratio = (total_itens / total_p) * 100
            nomes_itens = ", ".join(itens_pessoais)
            st.info(
                f"🎒 **Análise de Bagagem: Pessoa × {nomes_itens}**\n\n"
                f"Foram detectados **{total_itens} itens pessoais** para "
                f"**{total_p} pessoas** na ROI (ratio: {ratio:.0f}%). "
                f"{'Alta taxa de bagagem — possível área de trânsito/embarque.' if ratio > 60 else 'Taxa normal de itens pessoais na zona.'}"
            )

    # Correlação veículos
    veiculos = [c for c in ["Carro", "Moto", "Ônibus", "Caminhão", "Bicicleta"]
                if c in classes_selecionadas]
    if "Pessoa" in classes_selecionadas and veiculos and len(df_final) > 0:
        total_veic = sum(len(unicos_por_classe[v]) for v in veiculos)
        total_ped = len(unicos_por_classe["Pessoa"])
        if total_ped > 0 and total_veic > 0:
            ratio_vp = total_veic / total_ped
            nomes_veic = ", ".join(veiculos)
            nivel = ("⚠️ ALTO" if ratio_vp > 2
                     else "MODERADO" if ratio_vp > 0.5
                     else "BAIXO")
            st.warning(
                f"🚗 **Índice de Conflito Pedestre × Veículo ({nivel})**\n\n"
                f"Rácio veículos/pedestres na ROI: **{ratio_vp:.2f}** "
                f"({total_veic} {nomes_veic} para {total_ped} pessoas). "
                f"{'Considere medidas de segurança viária — alta densidade veicular em zona com pedestres.' if ratio_vp > 1 else 'Fluxo equilibrado entre pedestres e veículos na zona monitorada.'}"
            )

    # Análise temporal geral
    if len(df_final) > 0:
        colunas_roi = [f"{n} (ROI)" for n in classes_selecionadas]
        soma_total = df_final[colunas_roi].sum(axis=1)
        idx_pico_global = soma_total.idxmax()
        pico_global_seg = df_final.loc[idx_pico_global, "segundo"]
        pico_global_val = int(soma_total.loc[idx_pico_global])
        media_global = soma_total.mean()

        st.info(
            f"📊 **Resumo Temporal Global**\n\n"
            f"Ocupação média total da ROI: **{media_global:.1f} objetos/frame**. "
            f"O pico máximo agregado ocorreu em **{formatar_tempo(pico_global_seg)}** "
            f"com **{pico_global_val} objetos** simultâneos. "
            f"Duração total auditada: **{formatar_tempo(duracao)}** "
            f"({total_frames:,} frames a {fps:.0f} FPS)."
        )

    st.divider()

    # ── Exportação CSV ──────────────────────────────────────────────────────
    st.markdown("### 💾 Exportar Dados de Auditoria")
    if len(df_final) > 0:
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar Relatório de Auditoria Completo (CSV)",
            data=csv,
            file_name=f"auditoria_visual_{int(time.time())}.csv",
            mime="text/csv",
            width="stretch",
        )

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;color:#475569;font-size:0.78rem;
                padding:28px 0 10px 0;border-top:1px solid rgba(99,102,241,0.1);
                margin-top:18px;">
        🏭 <strong>Auditoria Visual Contínua</strong> v2.0 • Plataforma SaaS<br>
        YOLOv8 Nano • Rastreamento Multi-Objeto Multi-Classe • Análise Heurística
    </div>
    """, unsafe_allow_html=True)


# ── Entrypoint ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
