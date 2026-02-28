import os
import pandas as pd
from scipy.stats import ttest_rel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "resultados", "mitigacao_mmr_metricas_v2.csv")

df = pd.read_csv(DATA_PATH)

print("\n=== TESTE T PAREADO (wHHI antes vs depois) ===\n")

for fonte in df["fonte"].unique():
    print(f"\nFonte: {fonte}")
    sub = df[df["fonte"] == fonte].copy()

    antes = sub["antes_wHHI_canais"]
    depois = sub["depois_wHHI_canais"]

    valid = (~antes.isna()) & (~depois.isna())
    antes = antes[valid]
    depois = depois[valid]

    # diferenças
    dif = antes - depois
    if dif.std() == 0 or len(dif) < 2:
        print("Não aplicável: variância zero ou amostra insuficiente.")
        continue

    t_stat, p_val = ttest_rel(antes, depois)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.6f}")