import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "resultados", "mitigacao_mmr_metricas.csv")
FIG_DIR = os.path.join(BASE_DIR, "figuras")

os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# vamos focar na métrica principal: hhi_canais
for fonte in df["fonte"].unique():
    plt.figure()
    sub = df[df["fonte"] == fonte]

    for perfil in sub["perfil"].unique():
        p = sub[sub["perfil"] == perfil]

        plt.plot(
            p["dia"],
            p["antes_hhi_canais"],
            linestyle="--",
            marker="o",
            label=f"{perfil} Antes"
        )

        plt.plot(
            p["dia"],
            p["depois_hhi_canais"],
            linestyle="-",
            marker="o",
            label=f"{perfil} Depois"
        )

    plt.xlabel("Dia")
    plt.ylabel("HHI (Concentração por Canal)")
    plt.title(f"Mitigação MMR - {fonte}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"mitigacao_hhi_{fonte}.png"), dpi=200)
    plt.close()

print("Gráficos de mitigação gerados em /figuras/")