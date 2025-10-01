import os
import re
import sys
import logging
import warnings
import unicodedata
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import spacy
from nltk.corpus import stopwords
import nltk

from wordcloud import WordCloud
from leia import SentimentIntensityAnalyzer
from collections import Counter

# -------------------- CONFIGURAÇÃO --------------------

warnings.filterwarnings("ignore")

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_NUM = re.compile(r"\d+")
RE_PUNC = re.compile(r"[^\w\s]")
RE_SPACE = re.compile(r"\s+")

nlp = spacy.load("pt_core_news_sm", disable=["parser","ner"])

# -------------------- UTILITÁRIOS --------------------

def remove_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = remove_accents(text)
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    text = RE_HASHTAG.sub(" ", text)
    text = RE_NUM.sub(" ", text)
    text = RE_PUNC.sub(" ", text)
    text = RE_SPACE.sub(" ", text).strip()
    return text

def safe_nltk_download(resource: str, path: str):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

safe_nltk_download("stopwords", "corpora/stopwords")

BASE_STOP = set(stopwords.words("portuguese"))
CUSTOM_STOP = {
    "pra","pro","tá","ta","tô","to","tava","vamo","bora","né","ah","eh",
    "ai","aí","outra","outro","assim","tipo","cara","mano","velho","gente","pessoal",
    "poxa","oxe","aff","kkk","kkkk","rs","slk","tmj","blz","etc","q","pq",
    "tb","tbm","dps","hj","amanha","amanhã","ontem","agora","ainda","so","só",
    "mt","mto","muito","pouco","bem","mal","aqui","ali","lá","la","cada","todos",
    "todas","todo","toda","algum","alguns","alguma","algumas","coisa","coisas","nada",
    "tem","tinha","ter","fazer","fazendo","fez","feita","feitas","feito","fica","ficar",
    "sei","sabe","saber","acho","achar","vamos","vai","vou","pode","podia","podem",
    "porque","porquê","por que","por","pois","entao","então","desde","ate","até",
    "sobre","depois","antes","tambem","também","mesmo","mesma","mesmos","mesmas",
    "num","nuns","numa","numas","dum","duns","duma","dumas","bluesky","blsky","social",
    "app","profile","org","br","com","www","http","https","rt","via","nan","bsky",
    "poro","emo","emesse","enum","de","o","deo","de o","por o"
}
ALL_STOP = BASE_STOP.union(CUSTOM_STOP)

def tokenize_text(text: str) -> List[str]:
    doc = nlp(text)
    return [tok.lemma_.lower() for tok in doc 
            if tok.is_alpha 
            and tok.lemma_.lower() not in ALL_STOP 
            and len(tok) > 2]

# -------------------- CARREGAR DADOS --------------------

def load_and_clean(csv_path: str, text_col: str, term_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if text_col not in df.columns or term_col not in df.columns:
        raise ValueError(f"Colunas '{text_col}' ou '{term_col}' não encontradas no CSV")
    df = df.dropna(subset=[text_col, term_col]).reset_index(drop=True)
    df[text_col] = df[text_col].astype(str).str.strip()
    df["TextoLimpo"] = df[text_col].apply(clean_text)
    return df


# -------------------- ANÁLISE DE SENTIMENTO --------------------
def analyze_sentiments(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer,
                       text_col: str, pos_thresh: float = 0.05, neg_thresh: float = -0.05) -> pd.DataFrame:    
    df["Sentimento"] = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    
    df["Classificacao"] = "Neutro"
    
    df.loc[df["Sentimento"] > pos_thresh, "Classificacao"] = "Positivo"
    df.loc[df["Sentimento"] < neg_thresh, "Classificacao"] = "Negativo"
    
    counts = df["Classificacao"].value_counts()
    logging.info(f"Contagem de sentimentos: {counts.to_dict()}")
    
    return df


# -------------------- VISUALIZAÇÕES --------------------

def plot_global_distribution(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    palette = {"Positivo": "#2ca02c", "Negativo": "#d62728", "Neutro": "#7f7f7f"}

    counts = df["Classificacao"].value_counts().reindex(palette.keys(), fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[palette[s] for s in counts.index])

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f'{val:,}'.replace(',', '.'),
                ha='center', va='bottom', fontweight='bold')

    ax.set_title("Distribuição Global de Sentimentos")
    ax.set_ylabel("Número de Postagens")
    ax.set_ylim(0, counts.values.max() * 1.2 if counts.values.max() > 0 else 1)

    plt.tight_layout()
    fname = os.path.join(save_dir, "distribuicao_global.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    logging.info(f"Gráfico de distribuição global salvo: {fname}")
    plt.show()
    plt.close(fig)


def plot_distribution_per_term(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    palette = {"Positivo":"#2ca02c","Negativo":"#d62728","Neutro":"#7f7f7f"}

    for termo in df["Termo"].unique():
        df_term = df[df["Termo"] == termo]

        counts = df_term["Classificacao"].value_counts().reindex(palette.keys(), fill_value=0)

        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(counts.index, counts.values, color=[palette[s] for s in counts.index])
        
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, str(val), 
                    ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f"Distribuição de Sentimentos - {termo}")
        ax.set_ylabel("Número de Postagens")
        ax.set_ylim(0, counts.values.max() * 1.2 if counts.values.max() > 0 else 1)
        
        plt.tight_layout()
        fname = os.path.join(save_dir, f"distribuicao_{termo.lower().replace(' ','_')}.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        logging.info(f"Gráfico de distribuição para {termo} salvo: {fname}")
        plt.show()
        plt.close(fig)

def plot_global_sentiment_intensity(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    media_pos = df[df["Classificacao"]=="Positivo"]["Sentimento"].mean()
    media_neg = df[df["Classificacao"]=="Negativo"]["Sentimento"].mean()
    media_pos = media_pos if not np.isnan(media_pos) else 0
    media_neg = media_neg if not np.isnan(media_neg) else 0

    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(["Positivo","Negativo"], [media_pos, media_neg],
                  color=["#2ca02c","#d62728"], alpha=0.8)
    
    for bar, valor in zip(bars, [media_pos, media_neg]):
        ax.text(bar.get_x() + bar.get_width()/2, valor + 0.02 if valor>=0 else valor - 0.02,
                f"{valor:.3f}", ha="center", va='bottom' if valor>=0 else 'top', fontweight="bold")
    
    ax.set_title("Intensidade Média Global de Sentimento")
    ax.set_ylabel("Score Médio (Compound)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    fname = os.path.join(save_dir, "intensidade_global.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    logging.info(f"Gráfico de intensidade global salvo: {fname}")
    plt.show()
    plt.close(fig)

def plot_sentiment_intensity_per_term(df: pd.DataFrame, save_dir: str):
    """Gera gráficos de intensidade de sentimento por termo com eixo Y fixo."""
    os.makedirs(save_dir, exist_ok=True)
    for termo in df["Termo"].unique():
        df_term = df[df["Termo"]==termo]
        media_pos = df_term[df_term["Classificacao"]=="Positivo"]["Sentimento"].mean()
        media_neg = df_term[df_term["Classificacao"]=="Negativo"]["Sentimento"].mean()
        media_pos = media_pos if not np.isnan(media_pos) else 0
        media_neg = media_neg if not np.isnan(media_neg) else 0
        
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(["Positivo","Negativo"], [media_pos, media_neg],
                      color=["#2ca02c","#d62728"], alpha=0.8)
                      
        for bar, valor in zip(bars, [media_pos, media_neg]):
            ax.text(bar.get_x() + bar.get_width()/2, valor + 0.02 if valor>=0 else valor - 0.02,
                    f"{valor:.3f}", ha="center", va='bottom' if valor>=0 else 'top', fontweight="bold")
                    
        ax.set_title(f"Intensidade Média - {termo}")
        ax.set_ylabel("Score Médio (Compound)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        fname = os.path.join(save_dir, f"intensidade_{termo.lower().replace(' ','_')}.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        logging.info(f"Gráfico de intensidade para {termo} salvo: {fname}")
        plt.show()
        plt.close(fig)


def plot_engagement_by_sentiment(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    eng_cols = ["Likes","Replies","Reposts","Quotes"]
    sentimentos = ["Positivo", "Negativo", "Neutro"]

    def gerar_heatmap(data, titulo, fname):
        data = data.reindex(sentimentos).fillna(0)

        plt.figure(figsize=(8,6))
        sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)

        plt.title(titulo, fontsize=14)
        plt.ylabel("Sentimento")
        plt.xlabel("Métrica de Engajamento")
        plt.tight_layout()

        plt.savefig(fname, dpi=300, bbox_inches="tight")
        logging.info(f"Heatmap salvo: {fname}")
        plt.show()
        plt.close()

    # --- Heatmap Global ---
    df_eng_global = df.groupby("Classificacao")[eng_cols].mean().round(2)
    fname_global = os.path.join(save_dir, "heatmap_engajamento_global.png")
    gerar_heatmap(df_eng_global, "Engajamento Médio por Sentimento (Global)", fname_global)

    # --- Heatmap por Termo ---
    for termo in df["Termo"].unique():
        df_term = df[df["Termo"] == termo]
        df_eng_term = df_term.groupby("Classificacao")[eng_cols].mean()

        fname_term = os.path.join(
            save_dir, f"heatmap_engajamento_{termo.lower().replace(' ','_')}.png"
        )
        gerar_heatmap(
            df_eng_term,
            f"Engajamento Médio por Sentimento - {termo}",
            fname_term
        )        


# --- Dicionário de unificação (expressões e variações) ---
term_unify_map = {
    "queimada": "queimadas",
    "queimadas": "queimadas",
    "novela": "novelas",
    "novelas": "novelas",
    "desmatamento": "desmatamento",
    "fumaça": "fumaca",
    "fumaca": "fumaca",
    "mudancas climaticas": "mudancas_climaticas",
    "mudanças climaticas": "mudancas_climaticas",
    "mudancas climáticas": "mudancas_climaticas",
    "mudanças climáticas": "mudancas_climaticas",
    "aquecimento global": "aquecimento_global",
    "efeito estufa": "efeito_estufa",
    "amazônia": "amazonia",
    "amazonia": "amazonia",
    "pantanal": "pantanal",
    "agronegocio": "agronegocio",
    "agronegócio": "agronegocio",
    "incendio": "incendio",
    "incêndio": "incendio"
}

def unify_tokens(text: str) -> str:
    text = text.lower()
    text = remove_accents(text)
    for old, new in term_unify_map.items():
        text = re.sub(r'\b{}\b'.format(re.escape(old)), new, text)
    return text

def generate_wordclouds(df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    stopwords_wc = ALL_STOP

    # --- WordCloud por termo ---
    for termo in df["Termo"].unique():
        df_term = df[df["Termo"] == termo]
        tokens = []

        termo_unificado = unify_tokens(termo)

        for text in df_term["TextoLimpo"]:
            text = unify_tokens(text)
            doc = nlp(text)
            for tok in doc:
                tok_text = tok.text.lower()
                if tok_text.startswith("#"):
                    tok_text = tok_text[1:]
                if tok.is_alpha and tok_text not in stopwords_wc and len(tok_text) > 2 and tok_text != termo_unificado:
                    tokens.append(tok_text)

        if not tokens:
            continue

        freq = Counter(tokens)
        wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis")
        wc.generate_from_frequencies(freq)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"WordCloud - {termo}", fontsize=16)
        plt.tight_layout()
        fname = os.path.join(save_dir, f"wordcloud_{termo.lower().replace(' ','_')}.png")
        wc.to_file(fname)
        logging.info(f"WordCloud salva: {fname}")
        plt.show()
        plt.close(fig)

    # --- WordCloud global ---
    tokens_global = []
    for text in df["TextoLimpo"]:
        text = unify_tokens(text)
        doc = nlp(text)
        for tok in doc:
            tok_text = tok.text.lower()
            if tok_text.startswith("#"):
                tok_text = tok_text[1:]
            if tok.is_alpha and tok_text not in stopwords_wc and len(tok_text) > 2:
                tokens_global.append(tok_text)

    if tokens_global:
        freq_global = Counter(tokens_global)
        wc_global = WordCloud(width=1000, height=500, background_color="white", colormap="viridis")
        wc_global.generate_from_frequencies(freq_global)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc_global, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("WordCloud Global", fontsize=18)
        plt.tight_layout()
        fname_global = os.path.join(save_dir, "wordcloud_global.png")
        wc_global.to_file(fname_global)
        logging.info(f"WordCloud global salva: {fname_global}")
        plt.show()
        plt.close(fig)

# -------------------- EXECUÇÃO --------------------

def main():
    CSV_PATH = "dados_completos.csv"
    OUT_DIR = "saida_sentimentos"
    TEXT_COL = "Conteúdo"
    TERM_COL = "Termo"

    os.makedirs(OUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[
                            logging.FileHandler(os.path.join(OUT_DIR,"pipeline.log"), encoding="utf-8"),
                            logging.StreamHandler(sys.stdout)
                        ])

    logging.info("Carregando dados...")
    df = load_and_clean(CSV_PATH, TEXT_COL, TERM_COL)

    logging.info("Analisando sentimentos...")
    analyzer = SentimentIntensityAnalyzer()
    df = analyze_sentiments(df, analyzer, TEXT_COL, pos_thresh=0.05, neg_thresh=-0.05)

    out_csv = os.path.join(OUT_DIR,"dados_com_sentimento.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logging.info(f"CSV de saída salvo: {out_csv}")

    dist_dir = os.path.join(OUT_DIR,"distribuicao")
    intensity_dir = os.path.join(OUT_DIR,"intensidade")
    wc_dir = os.path.join(OUT_DIR,"wordclouds")
    eng_dir = os.path.join(OUT_DIR,"engajamento")
    
    logging.info("Gerando heatmap de engajamento por sentimento...")
    plot_engagement_by_sentiment(df, eng_dir)

    logging.info("Gerando gráficos de distribuição...")
    plot_global_distribution(df, dist_dir)
    plot_distribution_per_term(df, dist_dir)

    logging.info("Gerando gráficos de intensidade média...")
    plot_global_sentiment_intensity(df, intensity_dir)
    plot_sentiment_intensity_per_term(df, intensity_dir)

    logging.info("Gerando WordClouds por termo...")
    generate_wordclouds(df, wc_dir)

    logging.info("Processamento concluído.")

if __name__=="__main__":
    main()
