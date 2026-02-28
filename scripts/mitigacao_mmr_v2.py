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

def exposure_weight(pos: int, scheme="log"):
    # peso por posição (exposição). Quanto menor pos, maior peso.
    if scheme == "inv":
        return 1.0 / pos
    # padrão: desconto log (bem usado em IR/rec sys)
    return 1.0 / np.log2(1 + pos)

def weighted_hhi(items, positions, scheme="log"):
    """
    HHI ponderado por exposição (posição).
    p_i = soma(pesos dos itens do canal i) / soma(pesos)
    wHHI = sum(p_i^2)
    """
    if len(items) == 0:
        return np.nan
    w = np.array([exposure_weight(int(p), scheme=scheme) for p in positions], dtype=float)
    w = w / w.sum()

    # agrega pesos por item (canal)
    agg = {}
    for c, wi in zip(items, w):
        agg[c] = agg.get(c, 0.0) + wi
    probs = np.array(list(agg.values()), dtype=float)
    return float((probs ** 2).sum())

def topk_unique(items, k=5):
    return len(set(items[:k]))

def mmr_rerank(items_df, lam=0.3):
    """
    MMR agressivo:
    - lam menor => mais diversidade
    Similaridade: 1 se mesmo canal (forte), 0.5 se mesmo tema, 0 caso contrário.
    Relevância proxy: 1/posicao_original.
    """
    items = items_df.sort_values("posicao").copy()
    items["rel"] = 1.0 / items["posicao"].astype(float)

    selected = []
    remaining = list(items.index)

    def sim(i, j):
        same_channel = items.loc[i, "canal"] == items.loc[j, "canal"]
        if same_channel:
            return 1.0
        same_theme = items.loc[i, "tema"] == items.loc[j, "tema"]
        return 0.5 if same_theme else 0.0

    while remaining:
        best_idx = None
        best_score = -1e18
        for ridx in remaining:
            rel = float(items.loc[ridx, "rel"])
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

# corrige alias
df = df.rename(columns={"vews": "views"})

# normaliza
df["perfil"] = df["perfil"].astype(str).str.strip().str.upper()
df["fonte"] = df["fonte"].astype(str).str.strip().str.upper()
df["tema"] = df["tema"].astype(str).str.strip().str.upper()
df["canal"] = df["canal"].astype(str).str.strip()
df["posicao"] = pd.to_numeric(df["posicao"], errors="coerce").astype("Int64")
df["dia"] = pd.to_numeric(df["dia"], errors="coerce").astype("Int64")

df = df.dropna(subset=["perfil", "dia", "fonte", "posicao", "canal", "tema"])

rows = []
rank_out = []

for (perfil, dia, fonte), g in df.groupby(["perfil", "dia", "fonte"]):
    g = g.sort_values("posicao").copy()
    canais_before = list(g["canal"])
    pos_before = list(g["posicao"])

    # MMR agressivo
    reranked = mmr_rerank(g, lam=0.3)
    canais_after = list(reranked["canal"])
    pos_after = list(reranked["posicao_mmr"])

    # MÉTRICAS que mudam com reordenação
    before_wHHI = weighted_hhi(canais_before, pos_before, scheme="log")
    after_wHHI  = weighted_hhi(canais_after,  pos_after,  scheme="log")

    before_top5 = topk_unique(canais_before, k=5)
    after_top5  = topk_unique(canais_after,  k=5)

    rows.append({
        "perfil": perfil,
        "dia": int(dia),
        "fonte": fonte,
        "antes_wHHI_canais": before_wHHI,
        "depois_wHHI_canais": after_wHHI,
        "antes_top5_canais_unicos": before_top5,
        "depois_top5_canais_unicos": after_top5,
    })

    rank_out.append(reranked[["perfil","dia","fonte","posicao","posicao_mmr","canal","tema","titulo","url"]])

mmr_metrics = pd.DataFrame(rows).sort_values(["perfil","fonte","dia"])
mmr_metrics.to_csv(os.path.join(OUT_DIR, "mitigacao_mmr_metricas_v2.csv"), index=False, encoding="utf-8")

pd.concat(rank_out, ignore_index=True).to_csv(
    os.path.join(OUT_DIR, "mitigacao_mmr_rankings_v2.csv"), index=False, encoding="utf-8"
)

print("OK! Gerado:")
print(" - resultados/mitigacao_mmr_metricas_v2.csv (wHHI + top5 antes/depois)")
print(" - resultados/mitigacao_mmr_rankings_v2.csv (rankings reordenados)")