# 🏭 VisionAudit — Auditoria Visual Contínua com IA
> Detecção de Objetos (YOLOv8) e Análise Inteligente Operacional (Groq LLM) em Streamlit

Sistema profissional de monitoramento, rastreamento de fluxo e auditoria de processos por inteligência artificial multi-classe, rodando localmente com interface em alta definição (tema escuro de alto contraste) e análises geradas automaticamente via IA Generativa.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57+-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-00FFFF?logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM_API-F55036?logo=groq&logoColor=white)

---

## 📋 Sumário
1. [Detalhamento Técnico dos Requisitos (Etapas 1 a 4)](#-detalhamento-tcnico-dos-requisitos-etapas-1-a-4)
2. [Justificativa de Escolha do Modelo LLM](#-justificativa-de-escolha-do-modelo-llm)
3. [Funcionalidades Principais](#-funcionalidades-principais)
4. [Instalação e Execução Local](#-instalao-e-execuo-local)
5. [Configuração de Credenciais](#-configurao-de-credenciais)
6. [Estrutura do Projeto](#-estrutura-do-projeto)
7. [Como Publicar / Deploy](#-como-publicar--deploy)

---

## 🔍 Detalhamento Técnico dos Requisitos (Etapas 1 a 4)

Este projeto foi desenvolvido com foco em conformidade técnica rigorosa. Abaixo descrevemos como cada etapa exigida foi implementada no código:

### ⚙️ Etapa 1 — Preparar o Resultado
* **Função Reutilizável de Pipeline**: Toda a consolidação analítica do processamento de vídeo (C2) é encapsulada na função reutilizável `coletar_resultados_c2()` localizada em [`app.py`](file:///c:/Users/Usuario/Desktop/lkkf%20-%20Copia%20(2)/app.py). Ela recebe os contadores agregados, históricos de frames e picos de tráfego, separando a lógica de exibição gráfica da agregação dos dados brutos.
* **Saída em Formato Estruturado**: A função padroniza o retorno em um `dict` Python, facilmente serializável em JSON, contendo:
  ```json
  {
    "video_info": { "resolucao": "1920x1080", "fps": 30.0, "duracao_s": 45.2, "total_frames": 1356 },
    "classes_auditadas": ["Pessoa", "Carro", "Celular"],
    "roi_config": { "x_min_pct": 15, "x_max_pct": 85, "y_min_pct": 25, "y_max_pct": 85 },
    "metricas_por_classe": {
      "Pessoa": { "unicos_roi": 12, "total_detectados": 18, "pico_simultaneo": 5, "pico_segundo": 23.4, "taxa_entrada_roi_pct": 66.7 }
    },
    "resumo_temporal": { "ocupacao_media_objetos_por_frame": 3.42, "pico_global_objetos": 8, "pico_global_segundo": 23.4 }
  }
  ```
  *(Nota: Um Pandas DataFrame completo contendo o histórico temporal segundo-a-segundo é anexado na chave `historico_df` para uso no dashboard e exportação CSV).*
* **Dados e Contexto Ricos**: O pipeline fornece contexto detalhado para subsidiar decisões de engenharia de processos, incluindo a taxa de entrada na ROI (indivíduos que cruzaram a área útil em relação ao total rastreado), o segundo exato dos picos de concorrência e médias de ocupação de espaço físico.

### 🧠 Etapa 2 — Engenharia de Prompt
* **Papel do System Prompt**: O `SYSTEM_PROMPT` em [`llm_analyzer.py`](file:///c:/Users/Usuario/Desktop/lkkf%20-%20Copia%20(2)/llm_analyzer.py) define que a LLM deve agir sob o papel de um **Consultor Master em Otimização de Processos (Lean Six Sigma Black Belt) e Engenheiro de Segurança Operacional (EHS)**, garantindo respostas formais, embasadas em dados objetivos e com foco pragmático em melhoria contínua e mitigação de riscos industriais.
* **Construção Clássica do User Prompt**: A função `construir_prompt_usuario(resultado_c2)` recebe o dicionário gerado pelo pipeline C2, sanitiza estruturas não-serializáveis (como o DataFrame de histórico) e injeta os dados telemétricos de tráfego de forma limpa em blocos JSON estruturados dentro de um bloco de código Markdown.
* **Especificação do Formato de Resposta**: O prompt determina explicitamente à LLM o formato Markdown de saída com seis seções requeridas:
  1. *Resumo Executivo* (diagnóstico geral da dinâmica operacional);
  2. *Tabela Comparativa de Performance de Tráfego* em formato de tabela Markdown estruturada;
  3. *Análise de Gargalos e Eficiência* (detalhando picos e ociosidade de espaço);
  4. *Análise Crítica de Segurança Ocupacional (EHS)* (cruzamento pedestre-veículo, uso indevido de celulares ou acúmulo de bagagens em rotas);
  5. *Plano de Ação Estruturado* (composto por 3 a 5 ações, cada uma classificada em uma Matriz de Impacto × Esforço);
  6. *Índice de Complexidade Operacional (ICO)* (uma nota de 1.0 a 10.0 com devida justificação técnica).
* **Configuração de Temperatura**: Definida deterministicamente em **`temperatura = 0.4`** para assegurar que a análise gerada seja técnica, altamente baseada nos números fornecidos e livre de alucinações criativas, mantendo a capacidade analítica e coesão textual necessárias para planos de ação EHS.

### 🌐 Etapa 3 — Integração com o Groq
* **Chamada de API Robustecida**: O módulo [`llm_analyzer.py`](file:///c:/Users/Usuario/Desktop/lkkf%20-%20Copia%20(2)/llm_analyzer.py) implementa a biblioteca oficial `groq` passando um parâmetro explícito de **`timeout=30`** (30 segundos) para evitar travamento da interface.
* **Tratamento de Limites e Falhas de Rede**: Blocos `try-except` monitoram exceções específicas da API. Há tratamento customizado para:
  * *Erro 429 (Rate Limit)*: Avisa o usuário que os limites de requisição foram atingidos e solicita re-tentativa após alguns segundos;
  * *Timeouts*: Trata e informa quando a rede falha ou quando o servidor do Groq excede o tempo limite;
  * *Erro 401 (Autenticação)*: Alerta sobre chave de API inválida ou incorreta;
  * *Erro de Modelo Não Encontrado*: Identifica problemas na seleção de nomes de modelos.
  Qualquer falha do serviço de nuvem é capturada de forma limpa, não comprometendo a integridade ou execução da aplicação principal (fallback gracioso).
* **Registro de Latência e Consumo**: A função utiliza `time.perf_counter()` para medir com precisão milimétrica a latência da chamada e extrai os metadados de consumo fornecidos pelo objeto `usage` da resposta do Groq, registrando ativamente no console do terminal (via módulo `logging` do Python) a contagem exata de:
  * Tokens de entrada (Prompt)
  * Tokens de saída (Resposta)
  * Tokens totais consumidos
  * Tempo total de resposta (em milissegundos)

### 📊 Etapa 4 — Apresentação do Resultado
* **Exibição na Interface Streamlit**: Após a execução bem-sucedida do rastreamento de vídeo, os dados telemétricos acionam automaticamente a LLM (caso a chave API esteja configurada). A análise final estruturada em Markdown é renderizada com formatação e hierarquia de títulos clara na tela do usuário.
* **Aba de Comparação Direta**: Na seção de Inteligência Artificial, o dashboard implementa componentes de abas (`st.tabs`):
  * A aba **`🤖 Análise Inteligente`** exibe o relatório executivo gerado pela LLM formatado em Markdown. O painel utiliza uma regra customizada de CSS injetado que força o contraste do texto Markdown a renderizar em branco (`#ffffff`), garantindo legibilidade perfeita sobre o tema escuro do dashboard. Adicionalmente, as métricas de performance (Modelo, Consumo de Tokens e Latência da chamada) são impressas no topo deste painel.
  * A aba **`📦 Dados Brutos (JSON)`** exibe a saída bruta coletada diretamente pelo YOLOv8 e pela ROI, permitindo que o auditor confronte e audite a veracidade da análise do LLM com os dados exatos do banco de rastreamento.
* **Persistência de Estado (Session State)**: Os resultados de detecção (`resultado_c2`) e a resposta da LLM (`resultado_llm`) são cacheados em variáveis de estado do Streamlit. Isso impede re-processamentos de vídeo desnecessários quando o usuário interage com controles secundários (como download de CSV, mudança de abas de exibição ou geração manual).

---

## 📊 Justificativa de Escolha do Modelo LLM

O modelo padrão configurado na aplicação é o **`llama-3.3-70b-versatile`** fornecido pela API Groq. 

### Por que este modelo foi o escolhido?
1. **Velocidade de Geração Excepcional (Latência)**: Através da infraestrutura de chips LPU (Language Processing Units) do Groq, o modelo de 70 bilhões de parâmetros consegue entregar respostas completas de ~2.000 tokens em menos de 3.5 segundos. Modelos equivalentes hospedados em outros provedores possuem latências superiores a 15 segundos para saídas complexas como relatórios corporativos.
2. **Capacidade de Raciocínio Estruturado (Reasoning)**: O Llama-3.3 de 70B apresenta um desempenho próximo ao GPT-4 em tarefas de raciocínio lógico, interpretação de dados tabulares estruturados (como JSON) e síntese executiva. Ele é capaz de correlacionar de forma inteligente picos de tráfego, identificar desvios lógicos e formular tabelas perfeitas em Markdown sem perder a consistência estrutural.
3. **Suporte Robusto ao Idioma Português**: Apresenta excelente coesão textual em língua portuguesa, utilizando terminologias técnicas precisas da área de engenharia de processos (como *gargalos*, *lead time*, *concorrência de fluxo*) e de segurança de trabalho brasileira (NRs, rotas de fuga, equipamentos de proteção).
4. **Janela de Contexto Ampla**: A janela de contexto nativa de 128K permite receber com folga históricos e telemétricas de vídeos longos sem necessidade de sumarização prévia de dados.

---

## ✨ Funcionalidades Principais

* **🎯 Detecção e Rastreamento Local**: Detecção simultânea de até 12 categorias baseadas na arquitetura YOLOv8 rodando em CPU/GPU local, garantindo privacidade e sem custos de infraestrutura de nuvem.
* **🔲 Customização de Área Útil (ROI)**: Sliders interativos para delimitar as coordenadas da Zona de Auditoria em tempo real.
* **📊 Indicador de Conexão na Tela Inicial**: Indica visualmente se a API Groq está ativa ou se o arquivo `.env` precisa de ajustes antes de iniciar o processamento.
* **🧠 Sugestões Heurísticas Locais**: Módulo autônomo baseado em regras locais para detectar incidentes simples (Ex: Pessoa operando próximo a veículos ou picos de uso de celulares em área ativa).
* **💾 Exportação de Métricas**: Exportação completa da telemétrica segundo-a-segundo compilada em planilha CSV compatível com Power BI e Excel.

---

## 🚀 Instalação e Execução Local

Siga os passos abaixo para rodar a aplicação em seu ambiente:

### 1. Clonar o Repositório
```bash
git clone https://github.com/imhfelipe/auditoria-visual-saas.git
cd auditoria-visual-saas
```

### 2. Configurar o Ambiente Virtual (Recomendado)
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (Windows)
.venv\Scripts\activate

# Ativar ambiente (Linux/MacOS)
source .venv/bin/activate
```

### 3. Instalar as Dependências
```bash
pip install -r requirements.txt
```
*(Nota: A biblioteca `lapx` é necessária para o rastreamento persistente do YOLOv8).*

### 4. Configurar Variáveis de Ambiente
Crie um arquivo chamado `.env` na raiz do projeto (mesmo nível de `app.py`) e insira sua chave da API Groq:
```env
GROQ_API_KEY=gsk_sua_chave_groq_aqui
```

### 5. Executar a Aplicação
```bash
python -m streamlit run app.py
```
> O modelo YOLOv8 Nano (`yolov8n.pt`) será baixado automaticamente na primeira execução e armazenado em cache para execuções futuras.

---

## 🔑 Configuração de Credenciais

Para obter sua chave gratuita do Groq:
1. Acesse o console de desenvolvedores do [Groq](https://console.groq.com).
2. Vá em **API Keys** no menu lateral esquerdo.
3. Clique em **Create API Key**, dê um nome descritivo e copie o código gerado (`gsk_...`).
4. Salve este valor no arquivo `.env` sob o parâmetro `GROQ_API_KEY`.

---

## 📁 Estrutura do Projeto

```
auditoria-visual-saas/
│
├── app.py              # Interface do Dashboard Streamlit e motor de vídeo YOLOv8
├── llm_analyzer.py     # Lógica de encapsulamento, prompts e chamada à API Groq
├── requirements.txt    # Lista de dependências e bibliotecas do ecossistema
├── .env                # Configuração local da chave API (Excluído do Git)
├── .gitignore          # Filtro de arquivos ignorados pelo Git (Ignora .env e cache de IA)
└── README.md           # Esta documentação técnica
```

---

## 🚀 Como Publicar / Deploy

A hospedagem do projeto pode ser feita de forma 100% gratuita através do **Streamlit Community Cloud**:

1. Crie um repositório no GitHub contendo o código atualizado do projeto.
2. Acesse a plataforma [Streamlit Share](https://share.streamlit.io/) e faça login utilizando suas credenciais do GitHub.
3. Clique no botão **"New App"** ou **"Create App"**.
4. Aponte para o repositório público criado, selecione a branch `main` e aponte o arquivo principal como `app.py`.
5. **Configurando Secrets (API Key)**:
   Antes de confirmar o deploy, acesse o menu **"Advanced settings..."** (ou abra as configurações do app após o deploy) e localize o campo **Secrets**. Insira a variável de ambiente no formato TOML para que a nuvem do Streamlit possa conectá-la:
   ```toml
   GROQ_API_KEY = "gsk_sua_chave_groq_aqui"
   ```
6. Clique em **Deploy!** em poucos minutos a aplicação estará no ar em um subdomínio público gratuito do Streamlit.

---
Desenvolvido com foco em auditoria visual de alta performance, eficiência de processos e segurança ocupacional.
