# 🏭 Auditoria Visual Contínua — Plataforma SaaS

Sistema de Análise de Fluxo e Mapeamento de Processos com Inteligência Artificial multi-classe, rodando localmente com interface web profissional.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57+-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-00FFFF?logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)

## ✨ Funcionalidades

- **🎯 Detecção Multi-Classe** — 12 categorias COCO (Pessoa, Carro, Moto, Ônibus, Mochila, Celular...)
- **🔍 Rastreamento com IDs Únicos** — Tracking persistente via BoT-SORT/ByteTrack
- **🔲 ROI Configurável** — Zona de auditoria definida por sliders em tempo real
- **📊 Dashboard ao Vivo** — Métricas e gráficos atualizados durante processamento
- **🧠 Análise Heurística** — Correlações automáticas entre classes (Pessoa×Celular, Pedestre×Veículo)
- **💾 Exportação CSV** — Dados temporais completos para auditoria externa

## 🚀 Instalação e Execução

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute a aplicação
python -m streamlit run app.py
```

> O modelo YOLOv8 Nano (`yolov8n.pt`) será baixado automaticamente na primeira execução.

## 🛠️ Stack Tecnológica

| Tecnologia | Função |
|---|---|
| **Streamlit** | Interface web e dashboards |
| **YOLOv8 Nano** | Detecção e rastreamento de objetos |
| **OpenCV** | Processamento de vídeo e desenho |
| **Pandas** | Estruturação de dados e export CSV |
| **NumPy** | Operações numéricas |

## 📁 Estrutura

```
├── app.py              # Aplicação principal (arquivo único)
├── requirements.txt    # Dependências Python
├── .gitignore          # Arquivos ignorados pelo Git
└── README.md           # Documentação
```

## 📋 Como Usar

1. Abra a aplicação no navegador (`http://localhost:8501`)
2. Faça upload de um vídeo (MP4, AVI, MOV) na barra lateral
3. Selecione as classes de auditoria desejadas
4. Configure a Zona de Auditoria (ROI) com os sliders
5. Clique em **"▶️ Iniciar Auditoria Visual"**
6. Acompanhe métricas em tempo real e exporte o relatório final

## 📄 Licença

MIT License
