# ============================================================================
# 🏭 VisionAudit — Auditoria Visual Contínua com IA
# Detecção Multi-Classe · Rastreamento · Groq LLM
# ============================================================================
#
# 📦 INSTALAÇÃO:
#   pip install -r requirements.txt
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
import json
from datetime import timedelta
from collections import defaultdict

from llm_analyzer import analisar_com_llm, carregar_credenciais


# ── Configuração da Página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAudit — Auditoria Visual Contínua com IA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════
# 🧠 DICIONÁRIO DE CLASSES — O Cérebro da Auditoria
# ════════════════════════════════════════════════════════════════════════════
MAPA_CLASSES: dict[int, dict] = {
    0:  {"nome": "Pessoa",    "cor_rgb": (34, 211, 238),  "cor_bgr": (238, 211, 34),  "hex": "#22d3ee"},
    1:  {"nome": "Bicicleta", "cor_rgb": (163, 230, 53),  "cor_bgr": (53, 230, 163),  "hex": "#a3e635"},
    2:  {"nome": "Carro",     "cor_rgb": (245, 158, 11),  "cor_bgr": (11, 158, 245),  "hex": "#f59e0b"},
    3:  {"nome": "Moto",      "cor_rgb": (251, 146, 60),  "cor_bgr": (60, 146, 251),  "hex": "#fb923c"},
    5:  {"nome": "Ônibus",    "cor_rgb": (250, 204, 21),  "cor_bgr": (21, 204, 250),  "hex": "#facc15"},
    7:  {"nome": "Caminhão",  "cor_rgb": (239, 68, 68),   "cor_bgr": (68, 68, 239),   "hex": "#ef4444"},
    16: {"nome": "Cachorro",  "cor_rgb": (251, 113, 133), "cor_bgr": (133, 113, 251), "hex": "#fb7185"},
    24: {"nome": "Mochila",   "cor_rgb": (167, 139, 250), "cor_bgr": (250, 139, 167), "hex": "#a78bfa"},
    26: {"nome": "Bolsa",     "cor_rgb": (244, 114, 182), "cor_bgr": (182, 114, 244), "hex": "#f472b6"},
    28: {"nome": "Mala",      "cor_rgb": (251, 113, 133), "cor_bgr": (133, 113, 251), "hex": "#fb7185"},
    63: {"nome": "Laptop",    "cor_rgb": (56, 189, 248),  "cor_bgr": (248, 189, 56),  "hex": "#38bdf8"},
    67: {"nome": "Celular",   "cor_rgb": (52, 211, 153),  "cor_bgr": (153, 211, 52),  "hex": "#34d399"},
}

NOME_PARA_ID: dict[str, int] = {v["nome"]: k for k, v in MAPA_CLASSES.items()}


# ════════════════════════════════════════════════════════════════════════════
# 🎨 CSS — VisionAudit Design System
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Base dark theme ── */
    .stApp {
        background: #0a0f1a;
        font-family: 'Space Grotesk', system-ui, sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0c1220;
        border-right: 1px solid rgba(34,211,238,0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        font-family: 'Space Grotesk', system-ui, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', system-ui, sans-serif !important;
        color: #e2e8f0 !important;
    }
    h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
    h2 { font-weight: 600 !important; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(12,18,32,0.7);
        border: 1px solid rgba(34,211,238,0.12);
        border-radius: 10px;
        padding: 16px 18px;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0e7490, #22d3ee) !important;
        color: #020617 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', system-ui, sans-serif !important;
        box-shadow: 0 0 20px rgba(34,211,238,0.2) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(34,211,238,0.35) !important;
        transform: translateY(-1px) !important;
    }

    .stDownloadButton > button {
        background: rgba(34,211,238,0.1) !important;
        color: #22d3ee !important;
        border: 1px solid rgba(34,211,238,0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(12,18,32,0.5);
        border-radius: 8px;
        padding: 4px;
        border: 1px solid rgba(34,211,238,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(34,211,238,0.12) !important;
        color: #22d3ee !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Multiselect */
    div[data-baseweb="select"] {
        font-family: 'Space Grotesk', system-ui, sans-serif;
    }
    span[data-baseweb="tag"] {
        background: rgba(34,211,238,0.15) !important;
        border: 1px solid rgba(34,211,238,0.3) !important;
        color: #22d3ee !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    /* Divider */
    hr { border-color: rgba(34,211,238,0.08) !important; }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #0e7490, #22d3ee, #34d399) !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        font-family: 'Space Grotesk', system-ui, sans-serif;
    }

    /* Slider */
    .stSlider label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* JSON viewer */
    .stJson {
        font-family: 'JetBrains Mono', monospace !important;
    }
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
# 🎨 FUNÇÕES DE DESENHO — VisionAudit Canvas Style
# ════════════════════════════════════════════════════════════════════════════

def desenhar_fundo_grid(frame: np.ndarray) -> np.ndarray:
    """Aplica sutil grid pattern no fundo do frame."""
    h, w = frame.shape[:2]
    # Grid horizontal e vertical sutil
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (34, 211, 238), 1)
        # Tornar quase invisível
        frame[y, :] = cv2.addWeighted(frame[y:y+1, :], 0.97, np.full((1, w, 3), (34, 211, 238), dtype=np.uint8), 0.03, 0)[0]
    return frame


def desenhar_roi(frame: np.ndarray, x1: int, y1: int,
                 x2: int, y2: int) -> np.ndarray:
    """Desenha a zona de auditoria com estilo VisionAudit (cyan dashed)."""
    h, w = frame.shape[:2]

    # Escurecer exterior
    overlay = frame.copy()
    mask = np.zeros_like(frame)
    mask[y1:y2, x1:x2] = 255
    exterior = cv2.addWeighted(frame, 0.35, np.zeros_like(frame), 0.65, 0)
    frame = np.where(mask > 0, frame, exterior)

    # Borda tracejada cyan
    cor_roi = (238, 211, 34)  # #22d3ee em BGR
    dash_len = 12
    gap_len = 8

    # Top
    x = x1
    while x < x2:
        end = min(x + dash_len, x2)
        cv2.line(frame, (x, y1), (end, y1), cor_roi, 2)
        x += dash_len + gap_len
    # Bottom
    x = x1
    while x < x2:
        end = min(x + dash_len, x2)
        cv2.line(frame, (x, y2), (end, y2), cor_roi, 2)
        x += dash_len + gap_len
    # Left
    y = y1
    while y < y2:
        end = min(y + dash_len, y2)
        cv2.line(frame, (x1, y), (x1, end), cor_roi, 2)
        y += dash_len + gap_len
    # Right
    y = y1
    while y < y2:
        end = min(y + dash_len, y2)
        cv2.line(frame, (x2, y), (x2, end), cor_roi, 2)
        y += dash_len + gap_len

    # Label ROI
    cv2.putText(frame, "ROI", (x1 + 8, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_roi, 1, cv2.LINE_AA)

    return frame


def desenhar_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  id_obj: int, nome_classe: str, cor_bgr: tuple,
                  dentro_roi: bool) -> np.ndarray:
    """Desenha bounding box estilo VisionAudit com cantos accent."""
    if dentro_roi:
        cor = cor_bgr
        esp = 2
    else:
        cor = tuple(int(c * 0.4) for c in cor_bgr)
        esp = 1

    # Retângulo principal
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 1)

    # Cantos accent (top-left, top-right)
    c = 8
    cv2.line(frame, (x1, y1 + c), (x1, y1), cor, esp + 1)
    cv2.line(frame, (x1, y1), (x1 + c, y1), cor, esp + 1)
    cv2.line(frame, (x2 - c, y1), (x2, y1), cor, esp + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + c), cor, esp + 1)

    # Label
    label = f"{nome_classe} #{id_obj}"
    if dentro_roi:
        label += " [ROI]"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), cor, -1)

    # Texto escuro sobre fundo colorido
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (10, 14, 26), 1, cv2.LINE_AA)

    return frame


def adicionar_hud(frame: np.ndarray, contagens_roi: dict[str, int],
                  frame_idx: int, fps: float, classes_sel: list[str],
                  running: bool = True) -> np.ndarray:
    """Sobrepõe HUD estilo VisionAudit (barra superior com status)."""
    h, w = frame.shape[:2]

    # Barra superior escura
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (2, 6, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Linha divisória cyan
    cv2.line(frame, (0, 30), (w, 30), (238, 211, 34), 1)

    # Timestamp
    tempo = timedelta(seconds=int(frame_idx / max(fps, 1)))
    m = str(tempo).split(":")
    ts = f"{m[-2].zfill(2)}:{m[-1].zfill(2)}"

    # Dot + AUDIT + timestamp
    cor_cyan = (238, 211, 34)  # BGR for #22d3ee
    cv2.circle(frame, (14, 16), 4, cor_cyan, -1)
    cv2.putText(frame, f"AUDIT  {ts}", (24, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor_cyan, 1, cv2.LINE_AA)

    # Contagens por classe
    offset_x = 160
    for nome in classes_sel:
        if nome in NOME_PARA_ID:
            info = MAPA_CLASSES[NOME_PARA_ID[nome]]
            qtd = contagens_roi.get(nome, 0)
            cor = info["cor_bgr"]
            # Color chip
            cv2.rectangle(frame, (offset_x, 10), (offset_x + 8, 20), cor, -1)
            texto = f"{nome}: {qtd}"
            cv2.putText(frame, texto, (offset_x + 12, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (226, 232, 240), 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            offset_x += tw + 32

    # REC indicator
    if running:
        cv2.circle(frame, (w - 18, 16), 5, (68, 68, 239), -1)
        cv2.putText(frame, "REC", (w - 55, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (202, 254, 254), 1, cv2.LINE_AA)

    return frame


def formatar_tempo(segundos: float) -> str:
    """Formata segundos em MM:SS."""
    m = int(segundos) // 60
    s = int(segundos) % 60
    return f"{m:02d}:{s:02d}"


# ════════════════════════════════════════════════════════════════════════════
# 📦 PIPELINE C2 — RESULTADO ESTRUTURADO REUTILIZÁVEL
# ════════════════════════════════════════════════════════════════════════════

def coletar_resultados_c2(
    largura: int, altura: int, fps: float, duracao: float, total_frames: int,
    classes_selecionadas: list[str],
    roi_x_min: int, roi_x_max: int, roi_y_min: int, roi_y_max: int,
    unicos_por_classe: dict[str, set],
    todos_detectados: dict[str, set],
    picos: dict[str, dict],
    historico: list[dict],
) -> dict:
    """Encapsula os resultados da C2 em formato estruturado para LLM."""
    metricas_por_classe = {}
    for nome in classes_selecionadas:
        total_det = len(todos_detectados[nome])
        total_unicos = len(unicos_por_classe[nome])
        taxa = (total_unicos / total_det * 100) if total_det > 0 else 0.0
        metricas_por_classe[nome] = {
            "unicos_roi": total_unicos,
            "total_detectados": total_det,
            "pico_simultaneo": picos[nome]["valor"],
            "pico_segundo": round(picos[nome]["segundo"], 2),
            "taxa_entrada_roi_pct": round(taxa, 1),
        }

    df_hist = pd.DataFrame(historico) if historico else pd.DataFrame()
    resumo_temporal = {}
    if len(df_hist) > 0:
        colunas_roi = [f"{n} (ROI)" for n in classes_selecionadas]
        cols_existentes = [c for c in colunas_roi if c in df_hist.columns]
        if cols_existentes:
            soma = df_hist[cols_existentes].sum(axis=1)
            idx_pico = soma.idxmax()
            resumo_temporal = {
                "ocupacao_media_objetos_por_frame": round(float(soma.mean()), 2),
                "pico_global_objetos": int(soma.loc[idx_pico]),
                "pico_global_segundo": round(float(df_hist.loc[idx_pico, "segundo"]), 2),
            }

    return {
        "video_info": {
            "resolucao": f"{largura}x{altura}",
            "fps": round(fps, 1),
            "duracao_s": round(duracao, 1),
            "total_frames": total_frames,
        },
        "classes_auditadas": classes_selecionadas,
        "roi_config": {
            "x_min_pct": roi_x_min, "x_max_pct": roi_x_max,
            "y_min_pct": roi_y_min, "y_max_pct": roi_y_max,
        },
        "metricas_por_classe": metricas_por_classe,
        "resumo_temporal": resumo_temporal,
        "historico_df": df_hist,
    }


# ════════════════════════════════════════════════════════════════════════════
# 🖥️ COMPONENTES HTML — VisionAudit Design
# ════════════════════════════════════════════════════════════════════════════

def st_html(html_content: str):
    """Renderiza HTML limpando recuos para evitar formatação indesejada do markdown."""
    import re
    # Colapsar quebras de linha dentro de tags HTML (<...>) para evitar que sejam interpretadas como texto
    cleaned_tags = re.sub(r'<[^>]+>', lambda m: m.group(0).replace('\n', ' ').replace('\r', ' '), html_content)
    # Remover recuos de início de linha para evitar blocos de código markdown
    cleaned = "\n".join(line.strip() for line in cleaned_tags.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def html_header():
    """Cabeçalho VisionAudit com branding e badges."""
    return """
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding:12px 0; margin-bottom:8px;
    ">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="
                width:36px; height:36px; border-radius:8px;
                background:rgba(34,211,238,0.15);
                display:grid; place-items:center;
                box-shadow: 0 0 12px rgba(34,211,238,0.3);
            ">
                <span style="color:#22d3ee; font-size:18px;">⚡</span>
            </div>
            <div>
                <div style="
                    font-family:'JetBrains Mono',monospace;
                    font-size:10px; text-transform:uppercase;
                    letter-spacing:0.25em; color:#64748b;
                ">IA · Auditoria Contínua</div>
                <div style="
                    font-size:18px; font-weight:600;
                    font-family:'Space Grotesk',sans-serif;
                    color:#e2e8f0; letter-spacing:-0.02em;
                ">Vision<span style="color:#22d3ee; text-shadow:0 0 20px rgba(34,211,238,0.5);">Audit</span></div>
            </div>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
            <span style="
                font-family:'JetBrains Mono',monospace; font-size:11px;
                border:1px solid rgba(226,232,240,0.2); border-radius:6px;
                padding:4px 10px; color:#94a3b8;
            ">⚙ YOLOv8 · Nano</span>
            <span style="
                font-family:'JetBrains Mono',monospace; font-size:11px;
                border:1px solid rgba(226,232,240,0.2); border-radius:6px;
                padding:4px 10px; color:#94a3b8;
            ">◎ 12 fps</span>
        </div>
    </div>
    <div style="height:1px; background:rgba(34,211,238,0.1); margin-bottom:16px;"></div>
    """


def html_panel_header(eyebrow: str, title: str, right: str = ""):
    """Cabeçalho de painel estilo VisionAudit."""
    right_html = f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#64748b;">{right}</span>' if right else ""
    return f"""
    <div style="display:flex; align-items:flex-end; justify-content:space-between; margin-bottom:12px;">
        <div>
            <div style="
                font-family:'JetBrains Mono',monospace;
                font-size:10px; text-transform:uppercase;
                letter-spacing:0.2em; color:#22d3ee;
            ">{eyebrow}</div>
            <div style="
                font-size:18px; font-weight:600;
                font-family:'Space Grotesk',sans-serif;
                color:#e2e8f0; letter-spacing:-0.01em;
            ">{title}</div>
        </div>
        {right_html}
    </div>
    """


def html_sidebar_label(text: str):
    """Label de seção da sidebar em monospace."""
    return f"""
    <div style="
        font-family:'JetBrains Mono',monospace;
        font-size:10px; text-transform:uppercase;
        letter-spacing:0.2em; color:#64748b;
        margin-bottom:8px; display:flex; align-items:center; gap:6px;
    ">{text}</div>
    """


def html_stat_card(label: str, value, accent: str = ""):
    """Card de estatística compacto estilo VisionAudit."""
    color = accent if accent else "#e2e8f0"
    return f"""
    <div style="
        background:rgba(10,15,26,0.5); border-radius:8px;
        padding:10px 14px; flex:1;
    ">
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:9px; text-transform:uppercase;
            letter-spacing:0.12em; color:#64748b;
        ">{label}</div>
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:18px; font-weight:700;
            color:{color};
        ">{value}</div>
    </div>
    """


def html_insight_card(kind: str, title: str, body: str):
    """Card de insight com borda colorida."""
    if kind == "warn":
        border = "rgba(239,68,68,0.4)"
        bg = "rgba(239,68,68,0.08)"
        icon = "⚠"
    elif kind == "ok":
        border = "rgba(34,211,238,0.3)"
        bg = "rgba(34,211,238,0.05)"
        icon = "✓"
    else:
        border = "rgba(34,211,238,0.12)"
        bg = "rgba(12,18,32,0.5)"
        icon = "■"

    return f"""
    <div style="
        display:flex; gap:12px; border-radius:8px;
        border:1px solid {border}; background:{bg};
        padding:14px; margin-bottom:8px;
    ">
        <span style="font-size:14px; margin-top:2px;">{icon}</span>
        <div>
            <div style="font-size:14px; font-weight:600; color:#e2e8f0;
                        font-family:'Space Grotesk',sans-serif;">{title}</div>
            <div style="font-size:12px; color:#94a3b8; margin-top:4px;
                        line-height:1.5;">{body}</div>
        </div>
    </div>
    """


def html_class_report_card(label: str, color: str, unicos: int, pico: int,
                            pico_em: str, taxa: float, total_det: int):
    """Card de relatório por classe com borda superior colorida."""
    return f"""
    <div style="
        border-radius:10px; border:1px solid rgba(34,211,238,0.12);
        border-top:3px solid {color};
        background:rgba(12,18,32,0.5);
        padding:16px; margin-bottom:8px;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:14px; font-weight:600; color:#e2e8f0;
                         font-family:'Space Grotesk',sans-serif;">{label}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:10px;
                         color:#64748b;">{taxa}% entrada ROI</span>
        </div>
        <div style="display:flex; gap:8px; margin-top:12px;">
            {html_stat_card("únicos ROI", unicos, color)}
            {html_stat_card("pico", pico)}
            {html_stat_card("pico em", pico_em)}
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                    color:#64748b; margin-top:8px;">
            Total detectado: <span style="color:#e2e8f0;">{total_det}</span>
        </div>
    </div>
    """


def html_welcome():
    """Tela de boas-vindas VisionAudit."""
    features = [
        ("Detecção multi-classe", "12 categorias COCO simultâneas"),
        ("Rastreamento persistente", "IDs únicos frame a frame"),
        ("ROI configurável", "Delimite a zona exata da auditoria"),
        ("Dashboard ao vivo", "Métricas e gráfico em tempo real"),
        ("Heurísticas automáticas", "Correlações cruzadas entre classes"),
        ("Exportação CSV", "Pronto para Power BI / Tableau"),
    ]
    cards = ""
    for t, d in features:
        cards += f"""
        <div style="
            border-radius:8px; border:1px solid rgba(34,211,238,0.12);
            background:rgba(12,18,32,0.4); padding:14px;
        ">
            <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                        text-transform:uppercase; letter-spacing:0.15em;
                        color:#22d3ee;">▸ feature</div>
            <div style="font-size:14px; font-weight:600; color:#e2e8f0;
                        margin-top:4px; font-family:'Space Grotesk',sans-serif;">{t}</div>
            <div style="font-size:12px; color:#94a3b8; margin-top:2px;">{d}</div>
        </div>
        """

    return f"""
    <div style="
        position:relative; overflow:hidden; border-radius:12px;
        border:1px solid rgba(34,211,238,0.12);
        background:rgba(12,18,32,0.6);
        padding:32px; backdrop-filter:blur(8px);
    ">
        <div style="
            position:absolute; inset:0; opacity:0.15;
            background-image:
                linear-gradient(rgba(34,211,238,0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(34,211,238,0.06) 1px, transparent 1px);
            background-size:40px 40px;
        "></div>
        <div style="position:relative;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                        text-transform:uppercase; letter-spacing:0.3em;
                        color:#22d3ee;">IA · Auditoria Visual Contínua</div>
            <h2 style="font-size:2rem; font-weight:700; line-height:1.2;
                       margin:8px 0; max-width:640px; color:#e2e8f0 !important;">
                Mapeie processos em vídeo com IA
                <span style="color:#22d3ee; text-shadow:0 0 20px rgba(34,211,238,0.5);">multi-classe</span>,
                frame a frame.
            </h2>
            <p style="color:#94a3b8; font-size:14px; max-width:520px; line-height:1.6;">
                Carregue um vídeo, defina a Região de Interesse e veja a plataforma
                detectar, rastrear e correlacionar pessoas, veículos, dispositivos e
                bagagens — sem enviar nada para a nuvem.
            </p>
            <div style="display:grid; grid-template-columns:repeat(3,1fr);
                        gap:12px; margin-top:24px;">
                {cards}
            </div>
            <div style="
                display:inline-flex; align-items:center; gap:8px;
                border-radius:8px; border:1px dashed rgba(34,211,238,0.4);
                background:rgba(34,211,238,0.05);
                padding:8px 16px; margin-top:24px;
                font-family:'JetBrains Mono',monospace;
                font-size:12px; color:#22d3ee;
            ">← Selecione um vídeo no painel lateral para começar</div>
        </div>
    </div>
    """


def html_footer():
    """Footer VisionAudit."""
    return """
    <div style="
        border-top:1px solid rgba(34,211,238,0.08);
        padding:16px 0; margin-top:24px;
        display:flex; justify-content:space-between;
        font-family:'JetBrains Mono',monospace;
        font-size:11px; color:#64748b;
    ">
        <span>VisionAudit · processamento 100% local · sem upload externo</span>
        <span>v1.0 · YOLOv8 Nano + Groq LLM</span>
    </div>
    """


# ════════════════════════════════════════════════════════════════════════════
# 🏭 INTERFACE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st_html(html_header())

    # ── Sidebar — Painel de Configuração ────────────────────────────────────
    with st.sidebar:
        st_html(html_sidebar_label("1 · Upload do Vídeo"))
        arquivo_video = st.file_uploader(
            "Carregar vídeo para auditoria",
            type=["mp4", "avi", "mov"],
            help="Formatos suportados: MP4, AVI, MOV",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st_html(html_sidebar_label("2 · Alvos de Auditoria"))
        nomes_disponiveis = [v["nome"] for v in MAPA_CLASSES.values()]
        classes_selecionadas = st.multiselect(
            "Classes",
            options=nomes_disponiveis,
            default=["Pessoa", "Carro", "Celular"],
            label_visibility="collapsed",
        )

        ids_selecionados = [NOME_PARA_ID[n] for n in classes_selecionadas if n in NOME_PARA_ID]

        # Contador de classes
        st_html(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                    color:#64748b; margin-top:4px;">
            {len(classes_selecionadas)}/12 classes ativas
        </div>
        """)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st_html(html_sidebar_label("3 · Zona de Auditoria (ROI)"))
        c1, c2 = st.columns(2)
        with c1:
            roi_x_min = st.slider("X Mín", 0, 100, 15, format="%d%%")
            roi_y_min = st.slider("Y Mín", 0, 100, 25, format="%d%%")
        with c2:
            roi_x_max = st.slider("X Máx", 0, 100, 85, format="%d%%")
            roi_y_max = st.slider("Y Máx", 0, 100, 85, format="%d%%")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        iniciar = st.button("▶  Iniciar Auditoria", type="primary", use_container_width=True)

        st_html(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                    color:#475569; margin-top:8px; text-align:center;">
            Modelo YOLOv8n mantido em cache
        </div>
        """)

    # ── Session State Inicialização ──────────────────────────────────────────
    if "audit_iniciado" not in st.session_state:
        st.session_state.audit_iniciado = False
    if "resultado_c2" not in st.session_state:
        st.session_state.resultado_c2 = None
    if "resultado_llm" not in st.session_state:
        st.session_state.resultado_llm = None

    if iniciar:
        st.session_state.audit_iniciado = True
        st.session_state.resultado_c2 = None
        st.session_state.resultado_llm = None

    # ── Tela de espera ──────────────────────────────────────────────────────
    if not st.session_state.audit_iniciado:
        st_html(html_welcome())
        
        # Status da IA Generativa na Tela Inicial
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st_html(html_panel_header("IA Generativa", "Análise inteligente via LLM"))
        
        api_key = carregar_credenciais()
        if not api_key:
            st_html(html_insight_card(
                "warn", "API Groq não configurada",
                "Crie um arquivo .env com GROQ_API_KEY=sua_chave — Obtenha em console.groq.com"
            ))
        else:
            st_html(html_insight_card(
                "ok", "API Groq ativa",
                "Modelo Llama-3.3-70b-versatile pronto para gerar análises operacionais após o processamento do vídeo."
            ))
            
        st_html(html_footer())
        return

    # ── Validações ──────────────────────────────────────────────────────────
    if arquivo_video is None:
        st.warning("⚠️ Nenhum vídeo carregado. Faça upload na barra lateral.")
        return
    if not classes_selecionadas:
        st.warning("⚠️ Selecione ao menos uma classe de auditoria.")
        return
    if roi_x_min >= roi_x_max or roi_y_min >= roi_y_max:
        st.error("❌ ROI inválida: mínimos devem ser menores que máximos.")
        return

    api_key = carregar_credenciais()

    # ── Orquestração de Execução com Session State ───────────────────────────
    if st.session_state.resultado_c2 is None:
        # ── Salvar vídeo temporário ─────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(arquivo_video.read())
            caminho_video = tmp.name

        cap = cv2.VideoCapture(caminho_video)
        if not cap.isOpened():
            st.error("❌ Falha ao abrir o vídeo.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duracao = total_frames / fps

        rx1, rx2 = int(largura * roi_x_min / 100), int(largura * roi_x_max / 100)
        ry1, ry2 = int(altura * roi_y_min / 100), int(altura * roi_y_max / 100)

        # ── Carregar modelo ─────────────────────────────────────────────────────
        with st.spinner("⚡ Carregando YOLOv8 Nano..."):
            modelo = carregar_modelo()

        # ── Info do vídeo ───────────────────────────────────────────────────────
        cols_info = st.columns(4)
        cols_info[0].metric("Resolução", f"{largura}×{altura}")
        cols_info[1].metric("FPS", f"{fps:.1f}")
        cols_info[2].metric("Duração", formatar_tempo(duracao))
        cols_info[3].metric("Classes", f"{len(classes_selecionadas)}")

        # ── Layout de processamento ─────────────────────────────────────────────
        col_video, col_metricas = st.columns([3, 2], gap="large")

        with col_video:
            st_html(html_panel_header("Feed ao vivo", "Detecção & Rastreamento"))
            ph_frame = st.empty()

        with col_metricas:
            st_html(html_panel_header("Métricas", "ROI · em tempo real"))
            n_classes = len(classes_selecionadas)
            cols_metric = st.columns(min(n_classes, 3))
            ph_metrics: dict[str, any] = {}
            for i, nome in enumerate(classes_selecionadas):
                with cols_metric[i % min(n_classes, 3)]:
                    ph_metrics[nome] = st.empty()

            st.markdown("---")
            st_html(html_panel_header("Série temporal", "Ocupação da ROI"))
            ph_grafico = st.empty()

        barra = st.progress(0, text="Preparando auditoria...")

        # 📦 ESTRUTURAS DE DADOS
        unicos_por_classe: dict[str, set] = {n: set() for n in classes_selecionadas}
        todos_detectados: dict[str, set] = {n: set() for n in classes_selecionadas}
        historico: list[dict] = []
        picos: dict[str, dict] = {n: {"valor": 0, "segundo": 0.0} for n in classes_selecionadas}

        # Cores para gráfico
        colunas_roi = [f"{n} (ROI)" for n in classes_selecionadas]
        cores_hex = []
        for n in classes_selecionadas:
            cores_hex.append(MAPA_CLASSES.get(NOME_PARA_ID.get(n, -1), {}).get("hex", "#94a3b8"))

        # 🔄 LOOP DE PROCESSAMENTO
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            seg_atual = frame_idx / fps

            resultados = modelo.track(
                frame, persist=True, classes=ids_selecionados,
                verbose=False, conf=0.45, imgsz=640
            )[0]

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
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
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
                                          id_obj, nome_cls, info_cls["cor_bgr"], dentro_roi)

            frame = desenhar_roi(frame, rx1, ry1, rx2, ry2)
            frame = adicionar_hud(frame, contagem_frame, frame_idx, fps, classes_selecionadas)

            for nome, qtd in contagem_frame.items():
                if qtd > picos[nome]["valor"]:
                    picos[nome]["valor"] = qtd
                    picos[nome]["segundo"] = seg_atual

            if frame_idx % 30 == 0:
                gc.collect()

            registro = {"frame": frame_idx, "segundo": round(seg_atual, 2), "timestamp": formatar_tempo(seg_atual)}
            for nome in classes_selecionadas:
                registro[f"{nome} (ROI)"] = contagem_frame[nome]
                registro[f"{nome} (Únicos)"] = len(unicos_por_classe[nome])
            historico.append(registro)

            is_ultimo = (frame_idx >= total_frames)

            if frame_idx % 5 == 0 or is_ultimo:
                h_f, w_f = frame.shape[:2]
                if w_f > 960:
                    escala = 960 / w_f
                    frame_small = cv2.resize(frame, (960, int(h_f * escala)), interpolation=cv2.INTER_AREA)
                else:
                    frame_small = frame
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                ph_frame.image(frame_rgb, channels="RGB", use_container_width=True)

            if frame_idx % 5 == 0 or is_ultimo:
                for nome in classes_selecionadas:
                    info = MAPA_CLASSES.get(NOME_PARA_ID.get(nome, -1), {})
                    ph_metrics[nome].metric(
                        nome, f"{contagem_frame[nome]} na ROI",
                        delta=f"{len(unicos_por_classe[nome])} únicos",
                    )

            if frame_idx % 20 == 0 or is_ultimo:
                df_parcial = pd.DataFrame(historico)
                if len(df_parcial) > 120:
                    step = len(df_parcial) // 120
                    df_chart = df_parcial.iloc[::step]
                else:
                    df_chart = df_parcial
                ph_grafico.line_chart(
                    df_chart.set_index("segundo")[colunas_roi],
                    color=cores_hex, use_container_width=True,
                )
                del df_parcial, df_chart

            if frame_idx % 10 == 0 or is_ultimo:
                prog = min(frame_idx / max(total_frames, 1), 1.0)
                barra.progress(prog, text=f"frame {frame_idx:,}/{total_frames:,} · {prog*100:.1f}%")

        # ── Limpeza ─────────────────────────────────────────────────────────────
        cap.release()
        try:
            os.unlink(caminho_video)
        except OSError:
            pass
        barra.empty()
        ph_frame.empty()

        # Coletar resultado final C2
        resultado_c2 = coletar_resultados_c2(
            largura, altura, fps, duracao, total_frames,
            classes_selecionadas, roi_x_min, roi_x_max, roi_y_min, roi_y_max,
            unicos_por_classe, todos_detectados, picos, historico,
        )
        st.session_state.resultado_c2 = resultado_c2
        
        # ── ANÁLISE IA AUTOMÁTICA ────────────────────────────────────────────
        if api_key:
            with st.spinner("🧠 Gerando Análise Inteligente com IA (Groq Llama-3.3)..."):
                st.session_state.resultado_llm = analisar_com_llm(resultado_c2)

    # ── Recuperar do Session State se já processado ──────────────────────────
    resultado_c2 = st.session_state.resultado_c2
    df_final = resultado_c2["historico_df"]
    classes_selecionadas = resultado_c2["classes_auditadas"]
    duracao = resultado_c2["video_info"]["duracao_s"]
    total_frames = resultado_c2["video_info"]["total_frames"]
    fps = resultado_c2["video_info"]["fps"]

    cores_hex = []
    for n in classes_selecionadas:
        cores_hex.append(MAPA_CLASSES.get(NOME_PARA_ID.get(n, -1), {}).get("hex", "#94a3b8"))

    # ════════════════════════════════════════════════════════════════════════
    # 📋 RELATÓRIO — VisionAudit Design
    # ════════════════════════════════════════════════════════════════════════
    st_html(html_panel_header("Relatório", "Métricas por classe"))

    ranking_volume: list[tuple[str, int]] = []
    report_cols = st.columns(min(len(classes_selecionadas), 3))

    for i, nome in enumerate(classes_selecionadas):
        metrics = resultado_c2["metricas_por_classe"][nome]
        total_unicos = metrics["unicos_roi"]
        total_det = metrics["total_detectados"]
        pico_val = metrics["pico_simultaneo"]
        pico_seg = metrics["pico_segundo"]
        taxa = metrics["taxa_entrada_roi_pct"]
        ranking_volume.append((nome, total_unicos))
        info = MAPA_CLASSES.get(NOME_PARA_ID.get(nome, -1), {})

        with report_cols[i % min(len(classes_selecionadas), 3)]:
            st_html(html_class_report_card(
                nome, info.get("hex", "#94a3b8"),
                total_unicos, pico_val, f"{pico_seg:.1f}s",
                round(taxa, 1), total_det,
            ))

    # ── Gráfico final ──────────────────────────────────────────────────────
    st_html(html_panel_header("Série temporal", "Ocupação da ROI ao longo do vídeo"))
    if len(df_final) > 0:
        st.line_chart(
            df_final.set_index("segundo")[[f"{n} (ROI)" for n in classes_selecionadas]],
            color=cores_hex, use_container_width=True,
        )

    # ── Insights Heurísticos ───────────────────────────────────────────────
    st_html(html_panel_header("IA Heurística", "Correlações & Insights"))

    ranking_volume.sort(key=lambda x: x[1], reverse=True)
    if ranking_volume and ranking_volume[0][1] > 0:
        lider = ranking_volume[0]
        st_html(html_insight_card(
            "info", "Classe dominante",
            f"{lider[0]} liderou o tráfego na ROI com {lider[1]} indivíduos únicos."
        ))

    if "Pessoa" in classes_selecionadas and "Celular" in classes_selecionadas:
        if len(df_final) > 0:
            med_p = df_final["Pessoa (ROI)"].median()
            med_c = df_final["Celular (ROI)"].median()
            picos_simult = df_final[(df_final["Pessoa (ROI)"] > med_p) & (df_final["Celular (ROI)"] > med_c)]
            if len(picos_simult) > 0:
                pct = (len(picos_simult) / len(df_final)) * 100
                st_html(html_insight_card(
                    "warn", "Correlação Pessoa × Celular",
                    f"Em {pct:.1f}% dos frames houve pico simultâneo — possível zona de espera ou uso intenso de dispositivos."
                ))

    veiculos = [c for c in ["Carro", "Moto", "Ônibus", "Caminhão", "Bicicleta"] if c in classes_selecionadas]
    if "Pessoa" in classes_selecionadas and veiculos:
        total_veic = sum(resultado_c2["metricas_por_classe"][v]["unicos_roi"] for v in veiculos)
        total_ped = resultado_c2["metricas_por_classe"]["Pessoa"]["unicos_roi"]
        if total_ped > 0 and total_veic > 0:
            ratio = total_veic / total_ped
            nivel = "ALTO" if ratio > 0.7 else "MODERADO" if ratio > 0.25 else "BAIXO"
            kind = "warn" if ratio > 0.7 else "info" if ratio > 0.25 else "ok"
            st_html(html_insight_card(
                kind, f"Conflito Pedestre × Veículo · {nivel}",
                f"Índice = {ratio:.2f}. {'Avaliar sinalização e faixa segregada.' if ratio > 0.7 else 'Convivência dentro de parâmetros aceitáveis.'}"
            ))

    # ── Resumo global ──────────────────────────────────────────────────────
    if len(df_final) > 0:
        soma_total = df_final[[f"{n} (ROI)" for n in classes_selecionadas]].sum(axis=1)
        idx_pico = soma_total.idxmax()
        st_html(html_panel_header("Resumo Global", "Visão consolidada da auditoria"))
        st_html(f"""
        <div style="display:flex; gap:12px;">
            {html_stat_card("ocupação média", f"{soma_total.mean():.2f}")}
            {html_stat_card("pico simultâneo", int(soma_total.loc[idx_pico]), "#34d399")}
            {html_stat_card("momento de pico", f"{df_final.loc[idx_pico, 'segundo']:.1f}s")}
            {html_stat_card("duração auditada", formatar_tempo(duracao))}
        </div>
        """)

    # ── Exportar CSV ───────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if len(df_final) > 0:
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Exportar CSV", data=csv,
            file_name=f"visionaudit_{int(time.time())}.csv",
            mime="text/csv", use_container_width=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # 🤖 ANÁLISE INTELIGENTE POR IA (Groq LLM)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st_html(html_panel_header("IA Generativa", "Análise inteligente via LLM"))

    if not api_key:
        st_html(html_insight_card(
            "warn", "API Groq não configurada",
            "Crie um arquivo .env com GROQ_API_KEY=sua_chave — Obtenha em console.groq.com"
        ))
    else:
        tab_analise, tab_dados = st.tabs(["🤖  Análise Inteligente", "📦  Dados Brutos (JSON)"])

        with tab_dados:
            dados_exibicao = {k: v for k, v in resultado_c2.items() if k != "historico_df"}
            st.json(dados_exibicao, expanded=True)

        with tab_analise:
            resultado_llm = st.session_state.resultado_llm
            if resultado_llm is not None:
                if resultado_llm["sucesso"]:
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Modelo", resultado_llm["modelo"])
                    mc2.metric("Tokens", f"{resultado_llm['tokens_total']:,}")
                    mc3.metric("Latência", f"{resultado_llm['latencia_ms']:,.0f}ms")
                    st.markdown("---")
                    st.markdown(resultado_llm["analise"])
                else:
                    st_html(html_insight_card(
                        "warn", "Falha na análise",
                        f"{resultado_llm['erro']} — Os dados brutos continuam disponíveis na aba Dados Brutos."
                    ))
            else:
                if st.button("🧠  Gerar Análise Inteligente Manual", type="primary",
                             help="Envia dados para a LLM", key="btn_llm_manual"):
                    with st.spinner("Analisando com IA... ~10 segundos"):
                        resultado_llm = analisar_com_llm(resultado_c2)
                        st.session_state.resultado_llm = resultado_llm
                    st.rerun()

    # ── Footer ──────────────────────────────────────────────────────────────
    st_html(html_footer())


# ── Entrypoint ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
