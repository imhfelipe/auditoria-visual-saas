# 🏭 Auditoria Visual Contínua — Plataforma de Inteligência e Monitoramento

Sistema de Análise de Fluxo e Mapeamento de Processos com Inteligência Artificial multi-classe, rodando localmente com interface web profissional. Inclui **análise inteligente por LLM** via API Groq.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57+-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-00FFFF?logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM_API-F55036?logo=groq&logoColor=white)

## ✨ Funcionalidades

- **🎯 Detecção Multi-Classe** — 12 categorias COCO (Pessoa, Carro, Moto, Ônibus, Mochila, Celular...)
- **🔍 Rastreamento com IDs Únicos** — Tracking persistente via BoT-SORT/ByteTrack
- **🔲 ROI Configurável** — Zona de auditoria definida por sliders em tempo real
- **📊 Dashboard ao Vivo** — Métricas e gráficos atualizados durante processamento
- **🧠 Análise Heurística** — Correlações automáticas entre classes (Pessoa×Celular, Pedestre×Veículo)
- **🤖 Análise Inteligente por IA** — Insights gerados por LLM (Groq) a partir dos dados da auditoria
- **📦 Dados Estruturados** — Pipeline C2 reutilizável com saída em formato JSON
- **💾 Exportação CSV** — Dados temporais completos para auditoria externa

## 🚀 Instalação e Execução

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Configure a chave da API Groq
#    Crie um arquivo .env na raiz do projeto:
echo "GROQ_API_KEY=sua_chave_aqui" > .env

# 4. Execute a aplicação
python -m streamlit run app.py
```

> O modelo YOLOv8 Nano (`yolov8n.pt`) será baixado automaticamente na primeira execução.

## 🔑 Configuração da API Groq

A análise inteligente por IA utiliza a API do [Groq](https://console.groq.com) para gerar insights a partir dos dados da auditoria.

### Obtendo a chave:
1. Acesse [console.groq.com](https://console.groq.com)
2. Crie uma conta ou faça login
3. Gere uma API Key em **API Keys**
4. Crie um arquivo `.env` na raiz do projeto com o conteúdo:

```
GROQ_API_KEY=gsk_sua_chave_aqui
```

> ⚠️ **IMPORTANTE**: O arquivo `.env` está no `.gitignore` e **nunca deve ser commitado** no repositório.

### Modelo utilizado:
- **`llama-3.3-70b-versatile`** — Escolhido por oferecer o melhor equilíbrio entre capacidade analítica e velocidade de resposta. É o modelo mais capaz disponível gratuitamente no Groq, com janela de contexto de 128K tokens e excelente compreensão em português.

## 🛠️ Stack Tecnológica

| Tecnologia | Função |
|---|---|
| **Streamlit** | Interface web e dashboards |
| **YOLOv8 Nano** | Detecção e rastreamento de objetos |
| **OpenCV** | Processamento de vídeo e desenho |
| **Pandas** | Estruturação de dados e export CSV |
| **NumPy** | Operações numéricas |
| **Groq SDK** | Integração com LLM para análise inteligente |
| **python-dotenv** | Gestão de credenciais via `.env` |

## 📁 Estrutura

```
├── app.py              # Aplicação principal (interface + pipeline C2)
├── llm_analyzer.py     # Módulo de integração com LLM via Groq
├── requirements.txt    # Dependências Python
├── .env                # Credenciais (NÃO commitado)
├── .gitignore          # Arquivos ignorados pelo Git
└── README.md           # Documentação
```

## 🤖 Integração com LLM (Etapa C3)

A integração segue 4 etapas:

1. **Pipeline C2 Reutilizável** — A função `coletar_resultados_c2()` encapsula os resultados do processamento em formato JSON estruturado
2. **Engenharia de Prompt** — System prompt define o papel de "Consultor Sênior de Processos" e user prompt insere os dados de auditoria para análise
3. **Chamada à API Groq** — Com timeout de 30s, tratamento de erros (rate limit, autenticação, modelo indisponível) e logging de tokens/latência
4. **Apresentação** — Tabs na interface permitem comparar dados brutos (JSON) com a análise gerada pela IA

### Tratamento de Erros
- Falhas na chamada à LLM **nunca quebram o sistema**
- Mensagens de erro amigáveis são exibidas na interface
- Fallback gracioso: os dados brutos da auditoria sempre ficam disponíveis

## 📋 Como Usar

1. Abra a aplicação no navegador (`http://localhost:8501`)
2. Faça upload de um vídeo (MP4, AVI, MOV) na barra lateral
3. Selecione as classes de auditoria desejadas
4. Configure a Zona de Auditoria (ROI) com os sliders
5. Clique em **"▶️ Iniciar Auditoria Visual"**
6. Acompanhe métricas em tempo real e veja o relatório final
7. Na seção **"🤖 Análise Inteligente por IA"**, clique em **"🧠 Gerar Análise Inteligente"** para obter insights da LLM
8. Compare os dados brutos na aba **"📦 Dados Brutos (JSON)"** com a análise gerada

## 📄 Licença

MIT License
