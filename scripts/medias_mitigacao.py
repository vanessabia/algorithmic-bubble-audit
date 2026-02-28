import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "resultados", "mitigacao_mmr_metricas_v2.csv")

df = pd.read_csv(DATA_PATH)

print("\n=== MÉDIAS GERAIS (wHHI) ===\n")

for fonte in df["fonte"].unique():
    print(f"\nFonte: {fonte}")
    sub = df[df["fonte"] == fonte]

    media_antes = sub["antes_wHHI_canais"].mean()
    media_depois = sub["depois_wHHI_canais"].mean()
    diff = media_antes - media_depois

    print(f"Média Antes : {media_antes:.4f}")
    print(f"Média Depois: {media_depois:.4f}")
    print(f"Redução média: {diff:.4f}")