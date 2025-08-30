import sys
import requests
from datetime import datetime, timezone
import pandas as pd
import time
import logging
import random
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------- CONFIGURAÇÃO DE LOG -----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8")
    ]
)

# ----------------- CONSTANTES -----------------
PDS_URL = "https://bsky.social/xrpc"

USERNAME = "SEU NOME DE USUARIO"
PASSWORD = "SUA SENHA"

SEARCH_TERMS = [
    "queimada", "queimadas", "desmatamento", "fumaça", "mudanças climáticas",
    "aquecimento global", "efeito estufa", "amazônia", "pantanal",
    "agronegócio", "incêndio"
]

START_DATE = datetime(2024, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DATE   = datetime(2024, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
SINCE_STR = START_DATE.strftime('%Y-%m-%dT%H:%M:%SZ')
UNTIL_STR = END_DATE.strftime('%Y-%m-%dT%H:%M:%SZ')

TIMEOUT    = 20
MAX_RETRIES_5XX = 5

MAX_PAGES = 100
MAX_WORKERS = 5
SLEEP_BETWEEN_PAGES = 2.0
STAGGER_SUBMIT = 0.15

# ----------------- VARIÁVEIS GLOBAIS -----------------
session_data = {"access": None, "refresh": None}
HEADERS_BASE = {"User-Agent": "TCC-ColetaDadosBluesky (brunohf131@gmail.com)"}
http = requests.Session()
refresh_lock = Lock()
vistos_posts_lock = Lock()
vistos_replies_lock = Lock()

# ----------------- AUTENTICAÇÃO -----------------
def login():
    url = f"{PDS_URL}/com.atproto.server.createSession"
    resp = http.post(url, json={"identifier": USERNAME, "password": PASSWORD}, headers=HEADERS_BASE, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    session_data["access"] = data["accessJwt"]
    session_data["refresh"] = data["refreshJwt"]
    logging.info("Login ok – token armazenado.")

def refresh():
    url = f"{PDS_URL}/com.atproto.server.refreshSession"
    hdr = HEADERS_BASE | {"Authorization": f"Bearer {session_data.get('refresh')}"}
    resp = http.post(url, headers=hdr, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    session_data["access"] = data["accessJwt"]
    session_data["refresh"] = data["refreshJwt"]
    logging.info("Token renovado.")

# ----------------- REQUISIÇÃO -----------------
def request_pds(method: str, endpoint: str, use_auth: bool = True, **kwargs):
    url = f"{PDS_URL}/{endpoint}"
    for attempt in range(MAX_RETRIES_5XX):
        hdr = HEADERS_BASE.copy()
        if use_auth and session_data.get("access"):
            hdr["Authorization"] = f"Bearer {session_data['access']}"
        try:
            r = http.request(method, url, headers=hdr, timeout=TIMEOUT, **kwargs)

            if use_auth and r.status_code in (400, 401):
                content_type = r.headers.get("content-type", "")
                try:
                    if content_type.startswith("application/json") and r.json().get("error") == "ExpiredToken":
                        logging.warning("accessJwt expirado – tentando refresh...")
                        with refresh_lock:
                            refresh()
                        continue
                except Exception:
                    pass

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = int(retry_after)
                    except ValueError:
                        wait = min(60, (2 ** attempt) + random.uniform(0, 1))
                else:
                    wait = min(60, (2 ** attempt) + random.uniform(0, 1))
                logging.warning(f"429 recebido de {endpoint} (tentativa {attempt+1}). Esperando {wait:.1f}s antes de tentar novamente.")
                time.sleep(wait)
                continue

            if r.status_code >= 500:
                wait = min(60, (2 ** attempt) + random.uniform(0, 1))
                logging.warning(f"{r.status_code} em {endpoint} (tentativa {attempt+1}). Esperando {wait:.1f}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as exc:
            wait = min(60, (2 ** attempt) + random.uniform(0, 1))
            logging.error(f"Network error em {endpoint}: {exc} — retry em {wait:.1f}s (tentativa {attempt+1})")
            time.sleep(wait)
    raise RuntimeError(f"Falha após {MAX_RETRIES_5XX} tentativas em {endpoint}")

# ----------------- BUSCA DE POSTS -----------------
def coletar_postagens(termo: str):
    endpoint = "app.bsky.feed.searchPosts"
    next_cursor = None
    out = []
    page = 0

    # while page < MAX_PAGES:
    while True:
        current_params = {
            "q": termo,
            "limit": 100,
            "sort": "latest",
            "lang": "pt",
            "since": SINCE_STR,
            "until": UNTIL_STR
            }
        if next_cursor:
            current_params["cursor"] = next_cursor

        logging.info(f"[{termo}] Coletando página {page+1}...")
        resp = request_pds("GET", endpoint, params=current_params, use_auth=True)
        data = resp.json()
        posts = data.get("posts", [])
        next_cursor = data.get("cursor")

        logging.info(f"[{termo}] Página {page+1}: {len(posts)} posts recebidos.")
        for post in posts:
            created_str = post["record"].get("createdAt")
            if not created_str:
                continue
            dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))

            resumo = post["record"].get("text", "").replace("\n", " ")[:80]

            logging.info(f"[POST] ({dt}): {resumo}")

            out.append({
                "Termo": termo,
                "Conteúdo": post["record"].get("text", ""),
                "Data": dt,
                "Link": post.get("uri"),
                "Likes": post.get("likeCount", 0),
                "Replies": post.get("replyCount", 0),
                "Reposts": post.get("repostCount", 0),
                "Quotes": post.get("quoteCount", 0),
                "Tipo": "Postagem"
            })

        page += 1
        if not next_cursor:
            logging.info(f"[{termo}] Fim dos resultados (página {page}).")
            break

        logging.info(f"[{termo}] Aguardando {SLEEP_BETWEEN_PAGES}s antes da próxima página...")
        time.sleep(SLEEP_BETWEEN_PAGES)

    return out

# ----------------- BUSCA DE REPLIES -----------------
def coletar_replies(uri: str):
    endpoint = "app.bsky.feed.getPostThread"
    params = {"uri": uri, "depth": 3, "parentHeight": 80}
    all_replies = []

    try:
        resp = request_pds("GET", endpoint, params=params, use_auth=True)
        data = resp.json()
    except Exception as e:
        logging.error(f"Erro na requisição de replies para {uri}: {e}")
        return all_replies

    def extract_posts_from_thread(thread_node, collected_posts):
        if not isinstance(thread_node, dict):
            return
        post_data = thread_node.get('post')
        if isinstance(post_data, dict) and post_data.get('uri') != uri:
            collected_posts.append(post_data)
        for reply_thread in thread_node.get('replies', []):
            extract_posts_from_thread(reply_thread, collected_posts)

    extract_posts_from_thread(data.get('thread', {}), all_replies)

    replies = []
    for rep in all_replies:
        if not isinstance(rep, dict):
            continue

        created_str = rep.get('record', {}).get('createdAt', '')
        if not created_str:
            continue

        try:
            if '.' in created_str and created_str.endswith('Z'):
                created_str = created_str.split('.')[0] + 'Z'
            dt = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
        except ValueError as e:
            logging.warning(f"Erro ao converter data '{created_str}': {e}")
            continue

        if not (START_DATE <= dt <= END_DATE):
            continue

        text_content = rep.get('record', {}).get('text', '')
        resumo_reply = text_content.replace('\n', ' ')[:80] if text_content else ''

        logging.info(f"[REPLY]: {resumo_reply}")

        replies.append({
            "Termo": None,
            "Conteúdo": text_content,
            "Data": dt,
            "Link": rep.get('uri', ''),
            "Likes": rep.get('likeCount', 0),
            "Replies": rep.get('replyCount', 0),
            "Reposts": rep.get('repostCount', 0),
            "Quotes": rep.get('quoteCount', 0),
            "Tipo": "Reply",
            "Post Original": uri
        })

    return replies

# ----------------- MAIN -----------------
if __name__ == "__main__":
    try:
        login()
    except Exception as e:
        logging.critical(f"Login falhou: {e}")
        raise SystemExit(1)

    todos = []
    vistos_posts = set()
    vistos_replies = set()

    for termo in SEARCH_TERMS:
        logging.info(f"=== Termo: {termo} ===")
        posts = coletar_postagens(termo)

        added_posts = 0
        for post in posts:
            pid = post["Link"]
            with vistos_posts_lock:
                if pid not in vistos_posts:
                    vistos_posts.add(pid)
                    todos.append(post)
                    added_posts += 1
                    logging.info(f"[{termo}] Adicionado post: {pid} @{post['Data']} - {post['Conteúdo'][:50]}...")
        logging.info(f"[{termo}] Posts únicos adicionados: {added_posts} (total acumulado: {len(vistos_posts)})")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_post = {}
            for post in posts:
                uri = post["Link"]
                future = executor.submit(coletar_replies, uri)
                future_to_post[future] = post
                time.sleep(STAGGER_SUBMIT)

            for future in as_completed(future_to_post):
                base_post = future_to_post[future]
                try:
                    replies = future.result()
                    added_replies = 0
                    for reply in replies:
                        rlink = reply.get("Link", "")
                        with vistos_replies_lock:
                            if rlink and rlink not in vistos_replies:
                                vistos_replies.add(rlink)
                                reply["Termo"] = base_post["Termo"]
                                todos.append(reply)
                                added_replies += 1
                    logging.info(f"[{base_post['Termo']}] Replies adicionadas para {base_post['Link']}: {added_replies}")
                except Exception as exc:
                    logging.error(f"Erro ao coletar replies para {base_post.get('Link')}: {exc}")

    # salvar CSV
    df = pd.DataFrame(todos)
    df["Termo"] = df["Termo"].replace("queimada", "queimadas")
    df.to_csv("dados_completos.csv", index=False, encoding="utf-8")
    logging.info(f"Salvo {len(df)} registros em dados_completos.csv")