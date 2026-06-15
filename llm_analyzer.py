# ============================================================================
# 🤖 Módulo de Análise Inteligente com LLM via Groq API
# Auditoria Visual Contínua — Plataforma de Inteligência e Monitoramento
# ============================================================================
#
# Este módulo encapsula toda a lógica de integração com a API Groq para
# gerar análises inteligentes a partir dos dados brutos da auditoria visual.
#
# ============================================================================

import os
import json
import time
import logging
from datetime import datetime

from dotenv import load_dotenv

# ── Configuração de Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm_analyzer")


# ── Constantes ──────────────────────────────────────────────────────────────
MODELO_PADRAO = "llama-3.3-70b-versatile"
TEMPERATURA_PADRAO = 0.4
MAX_TOKENS_RESPOSTA = 4096
TIMEOUT_SEGUNDOS = 30


# ════════════════════════════════════════════════════════════════════════════
# 🔑 GESTÃO DE CREDENCIAIS
# ════════════════════════════════════════════════════════════════════════════

def carregar_credenciais() -> str | None:
    """
    Carrega a chave da API Groq de variáveis de ambiente, .env ou Streamlit Secrets.
    Retorna None se não encontrada.
    """
    # 1. Tentar carregar de Streamlit Secrets (Cloud Deploy)
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            key = st.secrets["GROQ_API_KEY"]
            if key:
                logger.info("GROQ_API_KEY carregada com sucesso do Streamlit Secrets.")
                return key
    except Exception:
        pass

    # 2. Tentar carregar de .env ou variáveis de ambiente (com caminho absoluto e override)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, ".env")
    
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning(f"GROQ_API_KEY não encontrada no ambiente (procurou em: {env_path})")
        return None
    logger.info("GROQ_API_KEY carregada com sucesso.")
    return api_key


# ════════════════════════════════════════════════════════════════════════════
# 📝 ENGENHARIA DE PROMPT
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Você é um **Consultor Master em Otimização de Processos (Lean Six Sigma Black Belt) e Engenheiro de Segurança Operacional (EHS)**.
Sua especialidade é interpretar dados analíticos de monitoramento e rastreamento de vídeo com visão computacional para auditar fluxos em ambientes industriais, logísticos, de varejo e urbanos.

Seu papel é diagnosticar a eficiência e a segurança operacional, identificando gargalos, riscos, desvios de conduta e oportunidades claras de melhoria com base exclusivamente em evidências de tráfego de objetos (pessoas, veículos, equipamentos e eletrônicos).

Suas diretrizes fundamentais:
1. **Linguagem Técnica e Executiva**: Escreva com seriedade, foco em eficiência de processos e conformidade de segurança. Use terminologias formais de engenharia de processos e segurança do trabalho.
2. **Base Científica**: Evite inferências genéricas. Use os números exatos e relações matemáticas deduzidas a partir das métricas e históricos.
3. **Equilíbrio de Crítica**: Reconheça pontos positivos de fluidez, mas seja extremamente rigoroso em alertas de falha operacional, ociosidade ou perigo.
4. **Foco em Solução Prática**: Toda recomendação deve ser operacionalizável e clara."""


def construir_prompt_usuario(resultado_c2: dict) -> str:
    """
    Constrói o prompt do usuário inserindo os dados da auditoria visual (C2)
    de forma estruturada para análise detalhada pela LLM.
    """
    # Serializar dados (excluindo o DataFrame que não é JSON-serializável)
    dados_para_llm = {k: v for k, v in resultado_c2.items() if k != "historico_df"}

    dados_json = json.dumps(dados_para_llm, ensure_ascii=False, indent=2)

    prompt = f"""## Dados de Entrada da Auditoria Visual (C2)

Abaixo estão as métricas consolidadas pelo módulo de rastreamento visual YOLOv8 com delimitação de Zona de Interesse (ROI).

```json
{dados_json}
```

---

## Estrutura da Análise Operacional Requerida

Com base nos dados estruturados acima, elabore um **Diagnóstico de Fluxo e Segurança Operacional** contendo as seguintes seções estruturadas:

### 📋 1. Resumo Executivo e Diagnóstico de Alto Nível
Forneça uma síntese analítica (2 a 3 parágrafos) avaliando a dinâmica geral de tráfego, ocupação do espaço auditado e os principais incidentes ou padrões observados no vídeo.

### 📊 2. Tabela Comparativa de Performance de Tráfego
Gere uma tabela em Markdown consolidando as métricas de todas as classes auditadas para facilitar a visualização comparativa direta:
| Classe de Objeto | Indivíduos Únicos (ROI) | Total Detectado | Pico Simultâneo | Momento do Pico (s) | Taxa de Entrada na ROI (%) |
|---|---|---|---|---|---|

### ⚙️ 3. Análise de Eficiência de Fluxo e Gargalos Operacionais
Identifique gargalos, picos de congestionamento e analise a ociosidade/ocupação das áreas da ROI utilizando os momentos de pico global de objetos e a ocupação média por frame. Detalhe como a taxa de entrada na ROI de cada classe aponta para desvios ou padrões lógicos no fluxo de trabalho.

### 🦺 4. Análise Crítica de Segurança Ocupacional, Conformidade e Riscos
Examine possíveis riscos de segurança, vulnerabilidades de processo e comportamentos anômalos. Foque especialmente em:
- **Presença indevida ou cruzamento de fluxos**: Proximidade ou interações arriscadas entre pedestres (Pessoas) e veículos (Carro, Moto, Caminhão, etc.).
- **Distrações ou desvios de conduta**: Uso ou presença de dispositivos eletrônicos (Celulares, Laptops) em áreas de operação ativa.
- **Acúmulo de bagagens/obstáculos**: Presença excessiva de Mochilas, Bolsas ou Malas bloqueando vias de circulação na ROI.

### 💡 5. Plano de Ação Estruturado (Matriz de Impacto × Esforço)
Proponha entre 3 e 5 recomendações executivas e prioritárias para mitigar os gargalos e riscos de segurança identificados. Formate cada recomendação informando:
- **Ação Recomendada**: O que deve ser feito.
- **Justificativa Técnica**: Baseada nos dados da auditoria.
- **Impacto no Processo**: Alto / Médio / Baixo.
- **Esforço de Implementação**: Alto / Médio / Baixo.

### 📉 6. Índice de Complexidade Operacional (ICO)
Atribua um índice de complexidade e risco operacional de **1.0 a 10.0** (onde 1.0 é um ambiente calmo, sem conflitos ou gargalos, e 10.0 representa um gargalo crítico ou risco iminente de acidente). Apresente uma justificativa matemática ou lógica sólida baseada nas correlações e contagens de pico para a pontuação atribuída."""

    return prompt


# ════════════════════════════════════════════════════════════════════════════
# 🚀 INTEGRAÇÃO COM A API GROQ
# ════════════════════════════════════════════════════════════════════════════

def analisar_com_llm(
    resultado_c2: dict,
    modelo: str = MODELO_PADRAO,
    temperatura: float = TEMPERATURA_PADRAO,
) -> dict:
    """
    Envia os dados da auditoria visual para a LLM via Groq e retorna a análise.

    Args:
        resultado_c2: Dicionário com os resultados da auditoria (saída de coletar_resultados_c2).
        modelo: Nome do modelo Groq a usar.
        temperatura: Temperatura de geração (0.0 = determinístico, 1.0 = criativo).

    Returns:
        dict com:
            - sucesso (bool): Se a chamada foi bem-sucedida
            - analise (str): Texto Markdown da análise gerada
            - modelo (str): Modelo utilizado
            - tokens_entrada (int): Tokens do prompt
            - tokens_saida (int): Tokens da resposta
            - tokens_total (int): Total de tokens consumidos
            - latencia_ms (float): Latência em milissegundos
            - erro (str|None): Mensagem de erro se falhou
    """
    resultado_padrao = {
        "sucesso": False,
        "analise": "",
        "modelo": modelo,
        "tokens_entrada": 0,
        "tokens_saida": 0,
        "tokens_total": 0,
        "latencia_ms": 0.0,
        "erro": None,
    }

    # ── Carregar credenciais ────────────────────────────────────────────────
    api_key = carregar_credenciais()
    if not api_key:
        resultado_padrao["erro"] = (
            "Chave da API Groq não encontrada. "
            "Configure GROQ_API_KEY no arquivo .env ou variáveis de ambiente."
        )
        logger.error(resultado_padrao["erro"])
        return resultado_padrao

    # ── Construir prompts ───────────────────────────────────────────────────
    try:
        prompt_usuario = construir_prompt_usuario(resultado_c2)
    except Exception as e:
        resultado_padrao["erro"] = f"Erro ao construir prompt: {str(e)}"
        logger.error(resultado_padrao["erro"])
        return resultado_padrao

    # ── Chamada à API Groq ──────────────────────────────────────────────────
    try:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=TIMEOUT_SEGUNDOS)

        logger.info(
            f"Enviando requisição à API Groq | Modelo: {modelo} | "
            f"Temperatura: {temperatura}"
        )

        inicio = time.perf_counter()

        resposta = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_usuario},
            ],
            temperature=temperatura,
            max_tokens=MAX_TOKENS_RESPOSTA,
            top_p=1,
            stream=False,
        )

        latencia = (time.perf_counter() - inicio) * 1000  # ms

        # ── Extrair resposta ────────────────────────────────────────────────
        analise = resposta.choices[0].message.content
        uso = resposta.usage

        resultado_final = {
            "sucesso": True,
            "analise": analise,
            "modelo": modelo,
            "tokens_entrada": uso.prompt_tokens if uso else 0,
            "tokens_saida": uso.completion_tokens if uso else 0,
            "tokens_total": uso.total_tokens if uso else 0,
            "latencia_ms": round(latencia, 1),
            "erro": None,
        }

        logger.info(
            f"✅ Análise gerada com sucesso | "
            f"Modelo: {modelo} | "
            f"Tokens: {resultado_final['tokens_total']} "
            f"(entrada: {resultado_final['tokens_entrada']}, "
            f"saída: {resultado_final['tokens_saida']}) | "
            f"Latência: {resultado_final['latencia_ms']}ms"
        )

        return resultado_final

    except ImportError:
        resultado_padrao["erro"] = (
            "Biblioteca 'groq' não instalada. Execute: pip install groq"
        )
        logger.error(resultado_padrao["erro"])
        return resultado_padrao

    except Exception as e:
        erro_tipo = type(e).__name__
        erro_msg = str(e)

        # Tratar erros específicos da API
        if "rate_limit" in erro_msg.lower() or "429" in erro_msg:
            resultado_padrao["erro"] = (
                "Limite de requisições da API Groq atingido. "
                "Aguarde alguns segundos e tente novamente."
            )
        elif "timeout" in erro_msg.lower() or "timed out" in erro_msg.lower():
            resultado_padrao["erro"] = (
                f"Timeout na chamada à API Groq ({TIMEOUT_SEGUNDOS}s). "
                "O servidor pode estar sobrecarregado."
            )
        elif "401" in erro_msg or "authentication" in erro_msg.lower():
            resultado_padrao["erro"] = (
                "Chave da API Groq inválida. Verifique GROQ_API_KEY no .env."
            )
        elif "model" in erro_msg.lower() and "not found" in erro_msg.lower():
            resultado_padrao["erro"] = (
                f"Modelo '{modelo}' não disponível no Groq. "
                "Tente outro modelo como 'mixtral-8x7b-32768'."
            )
        else:
            resultado_padrao["erro"] = (
                f"Erro na chamada à API Groq ({erro_tipo}): {erro_msg}"
            )

        logger.error(
            f"❌ Falha na chamada à API Groq | "
            f"Tipo: {erro_tipo} | Erro: {erro_msg}"
        )
        return resultado_padrao


# ════════════════════════════════════════════════════════════════════════════
# 🧪 TESTE RÁPIDO (execução direta)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Dados de exemplo para teste
    dados_teste = {
        "video_info": {
            "resolucao": "1920x1080",
            "fps": 30.0,
            "duracao_s": 60.0,
            "total_frames": 1800,
        },
        "classes_auditadas": ["Pessoa", "Carro"],
        "roi_config": {
            "x_min_pct": 5, "x_max_pct": 95,
            "y_min_pct": 5, "y_max_pct": 95,
        },
        "metricas_por_classe": {
            "Pessoa": {
                "unicos_roi": 12,
                "total_detectados": 18,
                "pico_simultaneo": 5,
                "pico_segundo": 23.4,
                "taxa_entrada_roi_pct": 66.7,
            },
            "Carro": {
                "unicos_roi": 4,
                "total_detectados": 6,
                "pico_simultaneo": 2,
                "pico_segundo": 30.1,
                "taxa_entrada_roi_pct": 66.7,
            },
        },
        "resumo_temporal": {
            "ocupacao_media_objetos_por_frame": 2.8,
            "pico_global_objetos": 7,
            "pico_global_segundo": 23.4,
        },
    }

    print("🧪 Testando integração com Groq API...\n")
    resultado = analisar_com_llm(dados_teste)

    if resultado["sucesso"]:
        print(f"✅ Sucesso! Modelo: {resultado['modelo']}")
        print(f"📊 Tokens: {resultado['tokens_total']} | Latência: {resultado['latencia_ms']}ms")
        print(f"\n{'='*60}\n")
        print(resultado["analise"])
    else:
        print(f"❌ Erro: {resultado['erro']}")
