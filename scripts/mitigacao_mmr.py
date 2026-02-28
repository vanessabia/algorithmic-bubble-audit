import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dados", "coleta_youtube.csv")
OUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUT_DIR, exist_ok=True)

def detect_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
    return ";" if head.count(";") > head.count(",") else ","

def herfindahl_index(items):
    if len(items) == 0:
        return np.nan
    counts = pd.Series(items).value_counts()
    p = counts / counts.sum()
    return float((p ** 2).sum())

def compute_metrics(g):
    g = g.sort_values("posicao")
    n = len(g)
    canais = list(g["canal"])
    temas = list(g["tema"])

    canais_unicos = len(set(canais))
    diversidade_canais = canais_unicos / n if n else np.nan
    hhi_canais = herfindahl_index(canais)

    seen = set()
    rep_count = 0
    for c in canais:
        if c in seen:
            rep_count += 1
        else:
            seen.add(c)
    repeticao_canal = rep_count / n if n else np.nan

    temas_unicos = len(set(temas))
    diversidade_temas = temas_unicos / n if n else np.nan
    hhi_temas = herfindahl_index(temas)

    return {
        "n": n,
        "diversidade_canais": diversidade_canais,
        "hhi_canais": hhi_canais,
        "repeticao_canal": repeticao_canal,
        "diversidade_temas": diversidade_temas,
        "hhi_temas": hhi_temas,
    }

def mmr_rerank(items, lam=0.7):
    """
    MMR simples:
    - Relevância: 1 / posicao_original (quanto mais alto no ranking original, mais relevante)
    - Similaridade: 1 se mesmo canal OU mesmo tema, 0 caso contrário
    lam controla tradeoff: 0.7 privilegia relevância, 0.3 diversidade
    """
    items = items.sort_values("posicao").copy()
    items["rel"] = 1.0 / items["posicao"].astype(float)

    selected = []
    remaining = list(items.index)

    def sim(i, j):
        same_channel = items.loc[i, "canal"] == items.loc[j, "canal"]
        same_theme = items.loc[i, "tema"] == items.loc[j, "tema"]
        return 1.0 if (same_channel or same_theme) else 0.0

    while remaining:
        best_idx = None
        best_score = -1e9
        for ridx in remaining:
            rel = items.loc[ridx, "rel"]
            if not selected:
                score = rel
            else:
                max_sim = max(sim(ridx, sidx) for sidx in selected)
                score = lam * rel - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_idx = ridx
        selected.append(best_idx)
        remaining.remove(best_idx)

    reranked = items.loc[selected].copy()
    reranked["posicao_mmr"] = range(1, len(reranked) + 1)
    return reranked

# LOAD
sep = detect_sep(DATA_PATH)
df = pd.read_csv(DATA_PATH, sep=sep, encoding="utf-8", engine="python")
df.columns = [c.strip().lower() for c in df.columns]

# normaliza
df["perfil"] = df["perfil"].astype(str).str.strip().str.upper()
df["fonte"] = df["fonte"].astype(str).str.strip().str.upper()
df["tema"] = df["tema"].astype(str).str.strip().str.upper()
df["canal"] = df["canal"].astype(str).str.strip()
df["posicao"] = pd.to_numeric(df["posicao"], errors="coerce").astype("Int64")
df["dia"] = pd.to_numeric(df["dia"], errors="coerce").astype("Int64")

df = df.dropna(subset=["perfil", "dia", "fonte", "posicao", "canal", "tema"])

rows = []
mmr_rows_out = []

for (perfil, dia, fonte), g in df.groupby(["perfil", "dia", "fonte"]):
    g = g.sort_values("posicao")

    before = compute_metrics(g)

    # aplica MMR
    reranked = mmr_rerank(g, lam=0.7)

    # para medir "depois", usamos a ordem mmr como ranking
    g_after = reranked.copy()
    g_after["posicao"] = g_after["posicao_mmr"]
    after = compute_metrics(g_after)

    row = {
        "perfil": perfil,
        "dia": int(dia),
        "fonte": fonte,
        **{f"antes_{k}": v for k, v in before.items()},
        **{f"depois_{k}": v for k, v in after.items()},
    }
    rows.append(row)

    # salva ranking antes/depois para inspeção manual (muito bom pro artigo)
    out_rank = reranked[["perfil","dia","fonte","posicao","posicao_mmr","canal","tema","titulo","url"]].copy()
    mmr_rows_out.append(out_rank)

mmr_metrics = pd.DataFrame(rows).sort_values(["perfil", "fonte", "dia"])
mmr_metrics.to_csv(os.path.join(OUT_DIR, "mitigacao_mmr_metricas.csv"), index=False, encoding="utf-8")

if mmr_rows_out:
    rank_df = pd.concat(mmr_rows_out, ignore_index=True)
    rank_df.to_csv(os.path.join(OUT_DIR, "mitigacao_mmr_rankings.csv"), index=False, encoding="utf-8")

print("OK! Gerado:")
print(" - resultados/mitigacao_mmr_metricas.csv (antes vs depois)")
print(" - resultados/mitigacao_mmr_rankings.csv (listas reordenadas)")