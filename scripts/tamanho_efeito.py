import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "resultados", "mitigacao_mmr_metricas_v2.csv")

df = pd.read_csv(DATA_PATH)

print("\n=== TESTE + TAMANHO DE EFEITO (wHHI) ===\n")

for fonte in df["fonte"].unique():
    print(f"\nFonte: {fonte}")
    sub = df[df["fonte"] == fonte]

    antes = sub["antes_wHHI_canais"]
    depois = sub["depois_wHHI_canais"]

    diff = antes - depois

    # t-test
    t_stat, p_val = ttest_rel(antes, depois)

    # Cohen's d (pareado)
    d = diff.mean() / diff.std()

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.6f}")
    print(f"Cohen's d: {d:.4f}")

    if abs(d) < 0.2:
        tamanho = "pequeno"
    elif abs(d) < 0.5:
        tamanho = "pequeno-medio"
    elif abs(d) < 0.8:
        tamanho = "medio"
    else:
        tamanho = "grande"

    print(f"Tamanho do efeito: {tamanho}")