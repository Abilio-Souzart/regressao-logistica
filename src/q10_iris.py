"""Questão 10: setosa × demais na base Iris, usando só petal length."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .config import SEED
from .utils import header, save_figure, save_table

COL = "petal length (cm)"


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def carregar() -> tuple[pd.DataFrame, pd.Series]:
    iris = load_iris(as_frame=True)
    return iris.data[[COL]], (iris.target == 0).astype(int).rename("setosa")


def ajustar(X, y, C: float) -> tuple[float, float]:
    m = LogisticRegression(C=C, max_iter=100_000, random_state=SEED).fit(X, y)
    return float(m.intercept_[0]), float(m.coef_[0][0])


def caminho_regularizacao(X, y, Cs) -> pd.DataFrame:
    linhas = []
    for C in Cs:
        b0, b1 = ajustar(X, y, C)
        linhas.append({"C": C, "b0": b0, "b1": b1, "OR = exp(b1)": np.exp(b1),
                       "fronteira (P = 0,5) em cm": -b0 / b1,
                       "P(setosa | 2 cm)": sigmoid(b0 + 2 * b1),
                       "P(setosa | 5 cm)": sigmoid(b0 + 5 * b1),
                       "acurácia": accuracy_score(y, (sigmoid(b0 + b1 * X[COL]) >= 0.5).astype(int))})
    return pd.DataFrame(linhas).set_index("C")


def plot_curvas(X, y, caminho: pd.DataFrame) -> None:
    grade = np.linspace(0.5, 7.5, 400)
    fig, ax = plt.subplots(figsize=(10, 5))
    jitter = np.random.default_rng(SEED).uniform(-0.03, 0.03, len(y))
    ax.scatter(X[COL], y + jitter, c=np.where(y == 1, "#F44336", "#2196F3"), s=25, alpha=0.6)
    ax.axvspan(X.loc[y == 1, COL].max(), X.loc[y == 0, COL].min(), color="grey", alpha=0.12,
               label="Intervalo sem nenhuma flor (1,9 a 3,0 cm)")
    cores = plt.cm.viridis(np.linspace(0, 0.9, len(caminho)))
    for cor, (C, r) in zip(cores, caminho.iterrows()):
        estilo = "-" if C == 1 else "--"
        ax.plot(grade, sigmoid(r["b0"] + r["b1"] * grade), estilo, color=cor, lw=2.2 if C == 1 else 1.4,
                label=f"C = {C:g}" + (" (padrão, usado no notebook)" if C == 1 else ""))
    ax.axvline(2, color="k", ls=":", lw=1)
    ax.axvline(5, color="k", ls=":", lw=1)
    ax.set_xlabel(COL)
    ax.set_ylabel("P(setosa)")
    ax.set_title("Separação perfeita: cada C produz uma curva diferente, "
                 "todas com 100% de acurácia")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    save_figure(fig, "q10_curvas_regularizacao")


def run() -> dict:
    header("QUESTÃO 10 - Setosa × demais (Iris, só petal length)")
    X, y = carregar()
    print(X.join(y).groupby("setosa")[COL].describe().round(2).to_string())
    print(f"Maior pétala de setosa: {X.loc[y == 1, COL].max()} cm | "
          f"menor das demais: {X.loc[y == 0, COL].min()} cm  ->  separação perfeita")

    # Modelo do notebook: scikit-learn com C = 1
    b0, b1 = ajustar(X, y, 1.0)
    acc = accuracy_score(y, (sigmoid(b0 + b1 * X[COL]) >= 0.5).astype(int))
    print(f"\nscikit-learn (C = 1): b0 = {b0:.4f}, b1 = {b1:.4f}, OR = {np.exp(b1):.4f}, "
          f"acurácia = {acc:.0%}")
    for x in (2.0, 5.0):
        z = b0 + b1 * x
        print(f"  petal length = {x:.0f} cm: z = {b0:.4f} + ({b1:.4f})·{x:.0f} = {z:.4f}  ->  "
              f"P(setosa) = 1/(1 + e^(-z)) = {sigmoid(z):.4f}")

    # Máxima verossimilhança sem penalização
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = sm.Logit(y.to_numpy(), sm.add_constant(X.to_numpy())).fit(disp=False, maxiter=200)
    avisos = {type(a.message).__name__ for a in w}
    print(f"\nstatsmodels (sem penalização): b0 = {res.params[0]:.1f}, b1 = {res.params[1]:.1f}, "
          f"convergiu = {res.mle_retvals['converged']}, avisos = {sorted(avisos)}")

    # Caminho de regularização
    caminho = caminho_regularizacao(X, y, [0.01, 0.1, 1, 10, 100, 1e4])
    save_table(caminho.round(4), "q10_caminho_regularizacao")
    print("\nCoeficientes e probabilidades para diferentes C (menos regularização -> C maior):")
    print(caminho.round(4).to_string())
    plot_curvas(X, y, caminho.loc[[0.1, 1, 10, 1e4]])

    return {"b0": b0, "b1": b1, "caminho": caminho}
