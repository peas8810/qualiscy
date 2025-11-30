import os
import textwrap
import requests
import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# Configurações básicas
# =========================

DATA_PATH = "data/qualis_revistas.csv"

# Inicializa cliente OpenAI (pega a chave das Secrets do Streamlit)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="Qualiscy – IA Científica",
    page_icon="📚",
    layout="wide"
)

# =========================
# Funções de dados
# =========================

@st.cache_data
def load_revistas() -> pd.DataFrame:
    """Carrega base de revistas com Qualis a partir do CSV."""
    df = pd.read_csv(DATA_PATH)
    # Opcional: normalizar nomes de colunas
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def filtrar_revistas(area: str, subarea: str | None = None) -> pd.DataFrame:
    """Filtra revistas por área e, se informado, por subárea/tema."""
    df = load_revistas()
    mask = df["area"].str.contains(area, case=False, na=False)

    if subarea:
        # Tenta casar subárea no escopo ou palavras-chave da revista
        if "escopo" in df.columns:
            mask = mask & df["escopo"].str.contains(subarea, case=False, na=False)
        elif "subarea" in df.columns:
            mask = mask & df["subarea"].str.contains(subarea, case=False, na=False)

    filtrado = df[mask].copy()

    # Ordena do melhor estrato para o pior, SE existir coluna estrato_qualis
    if "estrato_qualis" in filtrado.columns:
        ordem_qualis = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C"]
        cat_type = pd.CategoricalDtype(categories=ordem_qualis, ordered=True)
        filtrado["estrato_qualis"] = filtrado["estrato_qualis"].astype(cat_type)
        filtrado = filtrado.sort_values("estrato_qualis")
    return filtrado

def buscar_artigos_crossref(query: str, rows: int = 10) -> list[dict]:
    """
    Busca artigos na API pública da Crossref.
    Retorna lista de dicionários com título, ano e DOI.
    """
    if not query:
        return []

    base_url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": rows,
        "sort": "relevance"  # tenta trazer os mais relevantes
    }

    try:
        resp = requests.get(base_url, params=params, timeout=20)
        resp.raise_for_status()
        message = resp.json().get("message", {})
        items = message.get("items", [])
    except Exception as e:
        st.warning(f"⚠️ Erro ao acessar a API da Crossref: {e}")
        return []

    artigos = []
    for it in items:
        titulo = it.get("title", ["Sem título"])[0]
        doi = it.get("DOI")
        link = f"https://doi.org/{doi}" if doi else None

        # Tenta extrair ano de publicação
        ano = None
        for key in ["published-print", "published-online", "issued"]:
            if key in it:
                try:
                    ano = it[key]["date-parts"][0][0]
                    break
                except Exception:
                    pass

        artigos.append(
            {
                "titulo": titulo,
                "ano": ano,
                "doi": doi,
                "link": link
            }
        )
    return artigos

# =========================
# Função de chamada à IA
# =========================

def gerar_relatorio_ia(
    area: str,
    subarea: str | None,
    revistas: pd.DataFrame,
    artigos: list[dict],
    palavras_chave: str | None
) -> str:
    """
    Pede para o modelo da OpenAI organizar toda a informação
    e gerar um relatório em formato amigável.
    """

    revistas_records = revistas.to_dict(orient="records")

    user_prompt = f"""
    Você é um consultor científico especializado em avaliação de periódicos
    e estratégias de publicação, com foco no sistema Qualis brasileiro.

    Contexto do aluno:
    - Área principal do artigo: {area}
    - Subárea / tema específico: {subarea or "não informado"}
    - Palavras-chave fornecidas pelo aluno: {palavras_chave or "não informado"}

    Revistas disponíveis (dados da base interna):
    {revistas_records}

    Artigos encontrados a partir das palavras-chave (se houver):
    {artigos}

    TAREFA:
    Monte um RELATÓRIO ESTRUTURADO em português, com as seguintes seções:

    1. Visão geral da área e subárea
       - Explique em 1 parágrafo curto o foco da área/subárea informada.

    2. Melhores revistas para publicação
       - Liste as principais revistas indicadas, com:
         * nome do periódico
         * estrato Qualis (se disponível)
         * breve descrição do foco/escopo
         * em que tipo de trabalho elas costumam aceitar (ex.: relatos de caso, artigos originais, revisões, etc.)
         * link do site ou do sistema de submissão (se o campo 'link_site' ou similar existir nos dados)

    3. Template e instruções para submissão
       - Para cada revista listada, indique:
         * se há link ou informação de template/instruções aos autores nos dados fornecidos
         * oriente o aluno sobre onde encontrar essas informações no site da revista
         * explique brevemente quais são os pontos críticos de formatação a observar (tamanho do resumo, número de palavras, estrutura IMRAD, normas de citação, etc.)

    4. Artigos mais relevantes na temática
       - Caso haja artigos na seção 'artigos encontrados':
         * selecione até 10 mais relevantes
         * apresente em lista com: título, ano e DOI/link
         * comente, em 1–2 frases, o foco principal de cada artigo.

    5. Principais palavras-chave da área
       - Com base nos artigos listados e na temática:
         * indique um conjunto de 8 a 15 palavras-chave sugeridas em português
         * se possível, sugira também versão em inglês entre parênteses.

    6. O que está sendo pesquisado atualmente
       - Descreva em 2–4 parágrafos:
         * principais linhas de pesquisa atuais na área/subárea
         * lacunas frequentes (o que ainda falta estudar)
         * tendências emergentes.

    7. Sugestão de organização do artigo do aluno
       - Proponha uma estrutura de artigo (tópicos numerados) incluindo:
         * título provisório
         * sugestão de resumo (em 1 parágrafo)
         * tópicos da introdução (em forma de itens)
         * possíveis objetivos geral e específicos
         * sugestão de estrutura para métodos, resultados e discussão
         * considerações finais e implicações práticas.

    IMPORTANTE:
    - Use linguagem clara, direta, sem floreios desnecessários.
    - Não invente DOI ou revistas; use apenas os dados fornecidos.
    - Se alguma informação não estiver disponível nos dados, deixe claro que não foi fornecida e faça uma orientação genérica.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "Você é um consultor de publicação científica extremamente objetivo, "
                "especializado em Qualis brasileiro, escolha de periódicos e estratégia de publicação."
            ),
        },
        {"role": "user", "content": textwrap.dedent(user_prompt)},
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content

# =========================
# Interface Streamlit
# =========================

def main():
    st.title("📚 Qualiscy – IA Científica para Escolha de Revista")
    st.write(
        "Ferramenta de apoio à publicação: recomenda periódicos, indica templates, "
        "apresenta artigos relevantes e sugere a organização do seu artigo."
    )

    # --- Barra lateral: parâmetros de entrada ---
    st.sidebar.header("Configurações da consulta")

    # Você pode trocar essa lista por algo lido do CSV
    areas_possiveis = sorted(load_revistas()["area"].dropna().unique().tolist())
    area = st.sidebar.selectbox("Área principal do artigo", areas_possiveis)

    subarea = st.sidebar.text_input(
        "Subárea / tema específico",
        placeholder="Ex.: saneamento básico rural, bioquímica clínica, gestão tributária..."
    )

    palavras_chave = st.sidebar.text_area(
        "Palavras-chave (opcional)",
        placeholder="Ex.: saneamento básico, água potável, resíduos sólidos...",
        help="Se você preencher, a IA buscará artigos na Crossref e ajustará as recomendações."
    )

    num_artigos = st.sidebar.slider(
        "Quantidade de artigos a buscar (quando houver palavras-chave)",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "🔐 Sua chave da OpenAI fica salva nas *Secrets* do Streamlit. "
        "Nenhum dado sensível é salvo no servidor desta demonstração."
    )

    # --- Botão principal ---
    if st.button("Gerar relatório de recomendação"):
        if not area:
            st.warning("Selecione pelo menos a área principal do artigo.")
            return

        with st.spinner("Gerando recomendações com a IA Qualiscy..."):

            # 1. Filtra revistas
            revistas = filtrar_revistas(area, subarea)
            if revistas.empty:
                st.warning(
                    "Não encontrei revistas para essa combinação de área/subárea "
                    "na base interna. Ajuste a área ou atualize o CSV de Qualis."
                )

            # 2. Busca artigos (se tiver palavras-chave)
            if palavras_chave.strip():
                query = f"{area} {subarea or ''} {palavras_chave}"
                artigos = buscar_artigos_crossref(query=query, rows=num_artigos)
            else:
                artigos = []

            # 3. Chama IA para montar o relatório
            relatorio = gerar_relatorio_ia(
                area=area,
                subarea=subarea,
                revistas=revistas,
                artigos=artigos,
                palavras_chave=palavras_chave,
            )

        # --- Resultado ---
        st.success("Relatório gerado com sucesso!")
        st.markdown(relatorio)

        # Opcional: mostrar tabela de revistas e artigos separadamente
        with st.expander("Ver tabela de revistas utilizadas na recomendação"):
            st.dataframe(revistas.reset_index(drop=True))

        if artigos:
            with st.expander("Ver lista de artigos encontrados na Crossref"):
                st.dataframe(pd.DataFrame(artigos))

if __name__ == "__main__":
    main()

