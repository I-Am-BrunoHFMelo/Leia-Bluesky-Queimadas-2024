# -*- coding: utf-8 -*-
import os
import re
import sys
import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# gensim & LDA
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# spaCy pt
import spacy

# NLTK stopwords (com download seguro)
import nltk
from nltk.corpus import stopwords

# WordCloud & pyLDAvis
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# --------------------------- CONFIG GERAL ---------------------------

CSV_PATH   = "dados_completos.csv"
TEXT_COL   = "Conteúdo"
OUT_DIR    = "saida_lda"

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, "pipeline.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------- NLTK: download seguro ----------------------

def safe_nltk_download(resource: str, path: str):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

safe_nltk_download("stopwords", "corpora/stopwords")

# ------------------------ STOPWORDS pt-BR ---------------------------

BASE_STOP = set(stopwords.words("portuguese"))
CUSTOM_STOP = {
    # gírias, fillers, abreviações, pronomes e variações comuns em mídia social
    "pra","pro","tá","ta","tô","to","tava","tamos","vamo","bora","né","ah","eh",
    "ai","aí","outra","outro","assim","tipo","cara","mano","velho","gente","pessoal",
    "poxa","oxe","aff","kkk","kkkk","kkkkk","kkkkkk","rs","slk","tmj","blz","etc","q","pq",
    "pqq","tb","tbm","dps","hj","amanha","amanhã","ontem","agora","ainda","so","só",
    "mt","mto","muito","pouco","bem","mal","aqui","ali","lá","la","cada","todos",
    "todas","todo","toda","algum","alguns","alguma","algumas","coisa","coisas","nada",
    "tem","tinha","ter","fazer","fazendo","fez","feita","feitas","feito","fica","ficar",
    "sei","sabe","saber","acho","achar","vamos","vai","vou","pode","podia","podem",
    "porque","porquê","por que","por","pois","entao","então","desde","ate","até",
    "sobre","depois","antes","tambem","também","mesmo","mesma","mesmos","mesmas",
    "num","nuns","numa","numas","dum","duns","duma","dumas","pro","pra","pras","pros",
    "bluesky","blsky","social","app","profile","org","br","com","www","http","https",
    "rt","via","nan","bsky","novela"
}

ALL_STOP = BASE_STOP.union(CUSTOM_STOP)

# ----------------------- PREPROCESSAMENTO --------------------------

# Regex utilitários
RE_URL      = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
RE_MENTION  = re.compile(r"(?<!\w)@[\w\.\-]+")
RE_HASHTAG  = re.compile(r"(?<!\w)#[\w\_]+")
RE_EMOJI    = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u26FF]|\u2702|\uFE0F",
    flags=re.UNICODE
)
RE_NUM      = re.compile(r"\b\d+([.,]\d+)?\b")
RE_SPACE    = re.compile(r"\s+")

def clean_text_basic(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip().lower()
    t = RE_URL.sub(" ", t)
    t = RE_MENTION.sub(" ", t)
    t = RE_HASHTAG.sub(" ", t)
    t = RE_EMOJI.sub(" ", t)
    t = RE_NUM.sub(" ", t)
    t = RE_SPACE.sub(" ", t)
    return t.strip()

def spacy_lemmas(
    texts: List[str],
    nlp,
    pos_whitelist=("NOUN","PROPN","ADJ","VERB"), 
    min_len=3,
    batch_size=1000,
    n_process=2
) -> List[List[str]]:
    processed = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        toks = []
        for tok in doc:
            if not tok.is_alpha:          # ignora tokens com dígitos/pontuação
                continue
            if tok.pos_ not in pos_whitelist:
                continue
            lemma = tok.lemma_.lower()
            if len(lemma) < min_len:
                continue
            if lemma in ALL_STOP:
                continue
            toks.append(lemma)
        processed.append(toks)
    return processed

# ------------------------ COERÊNCIA & LDA --------------------------

def make_dict_corpus(texts: List[List[str]],
                     no_below=10, no_above=0.3, keep_n=100000):
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dictionary.doc2bow(t) for t in texts]
    return dictionary, corpus

def grid_coherence(
    texts: List[List[str]],
    dictionary,
    corpus,
    k_min=4, k_max=12,
    passes=10, iterations=400, random_state=42
) -> Tuple[int, list]:
    results = []
    for k in range(k_min, k_max + 1):
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=random_state,
            passes=passes,
            iterations=iterations,
            alpha="auto",
            eta="auto",
            eval_every=None,
            minimum_probability=0.0
        )
        cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v")
        score = cm.get_coherence()
        results.append((k, score))
        logging.info(f"K={k:>2} | Coerência c_v = {score:.4f}")
    best_k = max(results, key=lambda x: x[1])[0]
    return best_k, results

def plot_coherence(results: list, out_path: str):
    xs = [k for k,_ in results]
    ys = [s for _,s in results]
    plt.figure(figsize=(9,5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Número de Tópicos (K)")
    plt.ylabel("Coerência c_v")
    plt.title("Coerência vs K")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def train_final_lda(dictionary, corpus, num_topics: int,
                    passes=12, iterations=500, random_state=42) -> LdaModel:
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
        iterations=iterations,
        alpha="auto",
        eta="auto",
        eval_every=None,
        minimum_probability=0.0
    )
    return lda

# --------------------------- OUTPUTS -------------------------------

def assign_dominant_topic(lda: LdaModel, corpus) -> Tuple[np.ndarray, np.ndarray]:
    dom = []
    prob = []
    for bow in corpus:
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        if not dist:
            dom.append(-1); prob.append(0.0); continue
        k, p = max(dist, key=lambda x: x[1])
        dom.append(k); prob.append(p)
    return np.array(dom), np.array(prob)

def save_wordclouds(lda: LdaModel, out_dir: str, topn=30):
    for k in range(lda.num_topics):
        freqs = dict(lda.show_topic(k, topn=topn))
        if not freqs:
            continue
        wc = WordCloud(width=1000, height=500, background_color="white").generate_from_frequencies(freqs)
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Tópico {k}")
        fname = os.path.join(out_dir, f"wordcloud_topico_{k}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
        logging.info(f"Wordcloud salvo: {fname}")

def save_pyldavis(lda, corpus, dictionary, out_html: str):
    vis = gensimvis.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(vis, out_html)
    logging.info(f"pyLDAvis salvo em: {out_html}")

def save_topic_keywords(lda: LdaModel, out_csv: str, topn=15):
    rows = []
    for k in range(lda.num_topics):
        terms = lda.show_topic(k, topn=topn)
        rows.append({
            "topico": k,
            "palavras": ", ".join([w for w,_ in terms])
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    logging.info(f"Palavras por tópico salvas em: {out_csv}")

# ----------------------------- MAIN --------------------------------

def main():

    logging.info("Carregando CSV...")
    df = pd.read_csv(CSV_PATH)
    if TEXT_COL not in df.columns:
        raise ValueError(f"A coluna '{TEXT_COL}' não existe no CSV. Colunas disponíveis: {list(df.columns)}")

    logging.info("Limpando textos (pré-spaCy)...")
    raw = df[TEXT_COL].astype(str).fillna("")
    clean = raw.apply(clean_text_basic)

    logging.info("Carregando spaCy pt_core_news_sm...")
    nlp = spacy.load("pt_core_news_sm", disable=["parser","ner"])
    logging.info("Lematizando e filtrando tokens...")
    texts = spacy_lemmas(
        clean.tolist(),
        nlp,
        pos_whitelist=("NOUN","PROPN","ADJ", "VERB"),  # ajuste se quiser incluir VERB
        min_len=3,
        batch_size=1000,
        n_process=2
    )

    mask_nonempty = [len(t)>0 for t in texts]
    df = df.loc[mask_nonempty].reset_index(drop=True)
    texts = [t for t in texts if len(t)>0]
    logging.info(f"Documentos após limpeza: {len(texts)}")

    dictionary, corpus = make_dict_corpus(texts, no_below=10, no_above=0.3)

    if len(dictionary) == 0 or sum(len(b) for b in corpus) == 0:
        logging.warning("Corpus muito esparso com os filtros atuais; relaxando para no_below=5, no_above=0.5")
        dictionary, corpus = make_dict_corpus(texts, no_below=5, no_above=0.5)

    logging.info("Avaliando coerência para múltiplos K...")
    best_k, results = grid_coherence(
        texts, dictionary, corpus,
        k_min=4, k_max=12,
        passes=10, iterations=400, random_state=42
    )
    logging.info(f"K ótimo (c_v): {best_k}")
    plot_coherence(results, os.path.join(OUT_DIR, "coerencia_vs_k.png"))

    logging.info("Treinando LDA final...")
    lda = train_final_lda(dictionary, corpus, num_topics=best_k, passes=12, iterations=500)

    dom_k, dom_p = assign_dominant_topic(lda, corpus)
    df["Topico_Principal"] = dom_k
    df["Topico_Prob"] = dom_p

    save_topic_keywords(lda, os.path.join(OUT_DIR, "topicos_palavras.csv"), topn=20)
    save_wordclouds(lda, OUT_DIR, topn=30)
    save_pyldavis(lda, corpus, dictionary, os.path.join(OUT_DIR, "lda_vis.html"))

    out_csv = os.path.join(OUT_DIR, "dados_com_topicos.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    lda.save(os.path.join(OUT_DIR, "lda_model.model"))
    dictionary.save(os.path.join(OUT_DIR, "lda_dictionary.dict"))
    logging.info(f"DF salvo em: {out_csv}")
    logging.info("Concluído.")

if __name__ == "__main__":
    main()