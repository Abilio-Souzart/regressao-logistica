"""Análises complementares à Questão 9.

1. Por que a logística perde para a árvore em acurácia? Caudas longas.
2. Comparação por validação cruzada (mais estável que um único teste).
3. Ponto de corte considerando o custo dos erros.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import CUSTO_FN, CUSTO_FP, N_FOLDS, PROFUNDIDADES_ARVORE, SEED
from .preprocessing import feature_names, make_logit, make_tree
from .utils import header, save_figure, save_table


def assimetria(df: pd.DataFrame, cols) -> pd.DataFrame:
    return pd.DataFrame({"assimetria": df[cols].skew(),
                         "máx / mediana (|x|)": df[cols].abs().max() / df[cols].abs().median()
                         .replace(0, np.nan)}).sort_values("assimetria", ascending=False)


def comparar_cv(split) -> pd.DataFrame:
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    modelos = {"Logística (padronizada)": make_logit(split.X_tr, "padrao"),
               "Logística (log + padronizada)": make_logit(split.X_tr, "log")}
    modelos.update({f"Árvore profundidade {p}": make_tree(split.X_tr, p)
                    for p in PROFUNDIDADES_ARVORE})
    linhas = []
    for nome, m in modelos.items():
        cv = cross_validate(m, split.X_tr, split.y_tr, cv=folds,
                            scoring=["accuracy", "roc_auc"])
        m.fit(split.X_tr, split.y_tr)
        prob = m.predict_proba(split.X_te)[:, 1]
        linhas.append({
            "modelo": nome,
            "acc CV": cv["test_accuracy"].mean(), "AUC CV": cv["test_roc_auc"].mean(),
            "AUC CV (EP)": cv["test_roc_auc"].std(ddof=1) / np.sqrt(N_FOLDS),
            "acc teste": accuracy_score(split.y_te, prob >= 0.5),
            "AUC teste": roc_auc_score(split.y_te, prob),
        })
    return pd.DataFrame(linhas).set_index("modelo")


def plot_transformacao(df: pd.DataFrame, col: str) -> None:
    from .preprocessing import signed_log
    x = df[col].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(x, bins=60, color="#90A4AE")
    axes[0].set_title(f"{col} original (assimetria = {x.skew():.1f})")
    lx = pd.Series(signed_log(x))
    axes[1].hist(lx, bins=60, color="#2196F3")
    axes[1].set_title(f"sign(x)·log(1+|x|) (assimetria = {lx.skew():.1f})")
    fig.tight_layout()
    save_figure(fig, "comp_transformacao_log")


def plot_roc_comp(split, modelos: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for nome, m in modelos.items():
        prob = m.predict_proba(split.X_te)[:, 1]
        fpr, tpr, _ = roc_curve(split.y_te, prob)
        ax.plot(fpr, tpr, lw=2, label=f"{nome} ({roc_auc_score(split.y_te, prob):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curvas ROC no teste (AUC)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "comp_roc_transformacao")


def custo_por_corte(y, prob, cortes) -> pd.DataFrame:
    linhas = []
    for c in cortes:
        tn, fp, fn, tp = confusion_matrix(y, (prob >= c).astype(int), labels=[0, 1]).ravel()
        linhas.append({"corte": c, "bons recusados (FP)": fp, "maus aprovados (FN)": fn,
                       "acurácia": (tp + tn) / len(y), "custo (FP + 5·FN)": CUSTO_FP * fp + CUSTO_FN * fn})
    return pd.DataFrame(linhas).set_index("corte")


def run(q9: dict) -> dict:
    header("ANÁLISES COMPLEMENTARES - Questão 9")
    split, df = q9["split"], q9["df"]
    num = [c for c in split.X_tr.columns if c != "UF"]

    # 1. Caudas longas
    assim = assimetria(split.X_tr, num)
    save_table(assim.round(2), "comp_assimetria")
    print("1. Variáveis mais assimétricas (treino):")
    print(assim.head(8).round(1).to_string())
    plot_transformacao(split.X_tr, "REST_TIPO_1")

    log_model = make_logit(split.X_tr, "log").fit(split.X_tr, split.y_tr)
    coefs = pd.DataFrame({"β padronizada": q9["coef"],
                          "β log + padronizada": pd.Series(
                              log_model.named_steps["modelo"].coef_[0],
                              index=feature_names(log_model))})
    save_table(coefs.round(4), "comp_coef_padrao_vs_log")
    print("\n   Coeficientes nos dois modelos (maiores |β| do modelo log):")
    print(coefs.reindex(coefs["β log + padronizada"].abs().sort_values(ascending=False).index)
          .head(8).round(3).to_string())

    # 2. Validação cruzada
    comp = comparar_cv(split)
    save_table(comp.round(4), "comp_modelos_cv")
    print("\n2. Comparação (CV 10 folds no treino e teste):")
    print(comp.round(3).to_string())
    plot_roc_comp(split, {"Logística padronizada": q9["logit"],
                          "Logística log": log_model,
                          "Árvore prof. 4": make_tree(split.X_tr, 4).fit(split.X_tr, split.y_tr)})

    # 3. Corte por custo, no modelo com log
    prob = log_model.predict_proba(split.X_te)[:, 1]
    otimo = CUSTO_FP / (CUSTO_FP + CUSTO_FN)
    custos = custo_por_corte(split.y_te.to_numpy(), prob, [0.5, 0.4, 0.3, round(otimo, 3)])
    save_table(custos.round(4), "comp_custo_por_corte")
    print(f"\n3. Custo por corte (logística log, FN = {CUSTO_FN} x FP, corte teórico {otimo:.3f}):")
    print(custos.round(3).to_string())
    return {"comp": comp, "coefs": coefs, "custos": custos}
