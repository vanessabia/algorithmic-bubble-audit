import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dados", "coleta_youtube.csv")
OUT_DIR = os.path.join(BASE_DIR, "resultados")
FIG_DIR = os.path.join(BASE_DIR, "figuras")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def detect_sep(path: str) -> str:
    # tenta detectar separador (vírgula ou ponto-e-vírgula)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
    if head.count(";") > head.count(","):
        return ";"
    return ","

def normalize_str(s):
    if pd.isna(s):
        return ""
    return str(s).strip()

def parse_views(x):
    """
    Aceita exemplos:
    - "83.000" (pt-BR) -> 83000
    - "1,2 mi" -> 1200000
    - "390 mil" -> 390000
    - "523" -> 523
    - ""/NaN -> NaN
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == "":
        return np.nan

    # remove espaços
    s = re.sub(r"\s+", " ", s)

    # casos com "mi"/"mil"
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(mi|mil)\b", s)
    if m:
        num = m.group(1).replace(".", "").replace(",", ".")
        try:
            val = float(num)
        except:
            return np.nan
        mult = 1_000_000 if m.group(2) == "mi" else 1_000
        return int(val * mult)

    # remove tudo que não seja dígito, ponto, vírgula
    s2 = re.sub(r"[^0-9.,]", "", s)

    # formato pt-BR "83.000" -> remove pontos, troca vírgula por ponto
    # se tem ponto e não tem vírgula, pode ser separador de milhar
    if "." in s2 and "," not in s2:
        s2 = s2.replace(".", "")

    # se tem vírgula e ponto, assume vírgula decimal e ponto milhar
    if "." in s2 and "," in s2:
        s2 = s2.replace(".", "").replace(",", ".")

    # se só vírgula, pode ser decimal
    if "," in s2 and "." not in s2:
        s2 = s2.replace(",", ".")

    try:
        if s2 == "":
            return np.nan
        # se virou decimal, arredonda
        return int(float(s2))
    except:
        return np.nan

def parse_duration_to_seconds(x):
    """
    Aceita:
    - "05:59" (mm:ss)
    - "1:02:33" (hh:mm:ss)
    - "28:23:00" (hh:mm:ss) vindo do Sheets
    Retorna segundos (int) ou NaN.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except:
        return np.nan

    if len(parts) == 2:
        mm, ss = parts
        return mm * 60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss
    return np.nan

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a.intersection(set_b))
    uni = len(set_a.union(set_b))
    return inter / uni if uni else 0.0

def herfindahl_index(items):
    """
    HHI = sum(p_i^2), p_i = freq_i / N
    Varia de ~1/k (bem diverso) até 1 (100% concentrado).
    """
    if len(items) == 0:
        return np.nan
    counts = pd.Series(items).value_counts()
    p = counts / counts.sum()
    return float((p ** 2).sum())

# =========================
# LOAD
# =========================
sep = detect_sep(DATA_PATH)
df = pd.read_csv(DATA_PATH, sep=sep, encoding="utf-8", engine="python")

# normaliza nomes de colunas comuns (caso tenha variações)
df.columns = [c.strip().lower() for c in df.columns]

required = ["data", "perfil", "dia", "fonte", "posicao", "titulo", "canal", "url", "duracao", "publicado", "views", "tema"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Faltam colunas no CSV: {missing}. Colunas encontradas: {list(df.columns)}")

# limpeza básica
df["perfil"] = df["perfil"].astype(str).str.strip().str.upper()
df["fonte"] = df["fonte"].astype(str).str.strip().str.upper()
df["tema"] = df["tema"].astype(str).str.strip().str.upper()

df["dia"] = pd.to_numeric(df["dia"], errors="coerce").astype("Int64")
df["posicao"] = pd.to_numeric(df["posicao"], errors="coerce").astype("Int64")

df["canal"] = df["canal"].apply(normalize_str)
df["titulo"] = df["titulo"].apply(normalize_str)
df["url"] = df["url"].apply(normalize_str)

df["views_n"] = df["views"].apply(parse_views)
df["duracao_s"] = df["duracao"].apply(parse_duration_to_seconds)

# remove linhas totalmente quebradas
df = df.dropna(subset=["dia", "posicao", "perfil", "fonte"])

# =========================
# QUALITY CHECKS
# =========================
# checar se cada (perfil,dia,fonte) tem 10 itens e posicao 1..10 (ideal)
qc = (
    df.groupby(["perfil", "dia", "fonte"])
      .agg(n=("url", "count"),
           pos_min=("posicao", "min"),
           pos_max=("posicao", "max"))
      .reset_index()
)
qc.to_csv(os.path.join(OUT_DIR, "quality_checks.csv"), index=False, encoding="utf-8")

# =========================
# METRICS
# =========================
metrics_rows = []

for (perfil, dia, fonte), g in df.groupby(["perfil", "dia", "fonte"]):
    g = g.sort_values("posicao")
    n = len(g)

    canais = list(g["canal"])
    temas = list(g["tema"])
    urls = list(g["url"])

    canais_unicos = len(set(canais))
    diversidade_canais = canais_unicos / n if n else np.nan

    hhi_canais = herfindahl_index(canais)
    hhi_temas = herfindahl_index(temas)

    # repetição de canal: % itens cujo canal já apareceu antes na mesma lista
    seen = set()
    rep_count = 0
    for c in canais:
        if c in seen:
            rep_count += 1
        else:
            seen.add(c)
    repeticao_canal = rep_count / n if n else np.nan

    # diversidade temática simples
    temas_unicos = len(set(temas))
    diversidade_temas = temas_unicos / n if n else np.nan

    metrics_rows.append({
        "perfil": perfil,
        "dia": int(dia),
        "fonte": fonte,
        "n": n,
        "canais_unicos": canais_unicos,
        "diversidade_canais": diversidade_canais,
        "hhi_canais": hhi_canais,
        "repeticao_canal": repeticao_canal,
        "temas_unicos": temas_unicos,
        "diversidade_temas": diversidade_temas,
        "hhi_temas": hhi_temas,
    })

metrics = pd.DataFrame(metrics_rows).sort_values(["perfil", "fonte", "dia"])
metrics.to_csv(os.path.join(OUT_DIR, "metricas_por_dia.csv"), index=False, encoding="utf-8")

# =========================
# JACCARD (STABILITY)
# =========================
jac_rows = []
for (perfil, fonte), g in df.groupby(["perfil", "fonte"]):
    # conjuntos de canais por dia (ou URLs, você escolhe)
    by_day = {int(d): set(x["canal"]) for d, x in g.groupby("dia")}
    days = sorted(by_day.keys())
    for i in range(1, len(days)):
        d_prev, d_cur = days[i-1], days[i]
        jac = jaccard(by_day[d_prev], by_day[d_cur])
        jac_rows.append({"perfil": perfil, "fonte": fonte, "dia_de": d_prev, "dia_para": d_cur, "jaccard_canais": jac})

jacc = pd.DataFrame(jac_rows).sort_values(["perfil", "fonte", "dia_de"])
jacc.to_csv(os.path.join(OUT_DIR, "jaccard_estabilidade.csv"), index=False, encoding="utf-8")

# =========================
# PLOTS
# =========================
def plot_metric(metric_col, ylabel, fname_prefix):
    for fonte in sorted(metrics["fonte"].unique()):
        plt.figure()
        sub = metrics[metrics["fonte"] == fonte]
        for perfil in sorted(sub["perfil"].unique()):
            p = sub[sub["perfil"] == perfil]
            plt.plot(p["dia"], p[metric_col], marker="o", label=perfil)
        plt.xlabel("Dia do experimento")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} por dia ({fonte})")
        plt.legend()
        out = os.path.join(FIG_DIR, f"{fname_prefix}_{fonte}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

plot_metric("diversidade_canais", "Diversidade de canais (unicos/n)", "div_canais")
plot_metric("hhi_canais", "Concentracao (HHI) por canal", "hhi_canais")
plot_metric("repeticao_canal", "Repeticao de canal", "rep_canal")
plot_metric("diversidade_temas", "Diversidade de temas (unicos/n)", "div_temas")
plot_metric("hhi_temas", "Concentracao (HHI) por tema", "hhi_temas")

# plot Jaccard
if len(jacc) > 0:
    for fonte in sorted(jacc["fonte"].unique()):
        plt.figure()
        sub = jacc[jacc["fonte"] == fonte]
        for perfil in sorted(sub["perfil"].unique()):
            p = sub[sub["perfil"] == perfil]
            # coloca no eixo x o "dia_para"
            plt.plot(p["dia_para"], p["jaccard_canais"], marker="o", label=perfil)
        plt.xlabel("Dia (comparado com dia anterior)")
        plt.ylabel("Jaccard de canais (dia-1 vs dia)")
        plt.title(f"Estabilidade (Jaccard) - {fonte}")
        plt.legend()
        out = os.path.join(FIG_DIR, f"jaccard_{fonte}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

print("OK! Arquivos gerados em:")
print(" - resultados/metricas_por_dia.csv")
print(" - resultados/jaccard_estabilidade.csv")
print(" - resultados/quality_checks.csv")
print(" - figuras/*.png")