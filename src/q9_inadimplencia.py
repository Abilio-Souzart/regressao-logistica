"""Questão 9: regressão logística para inadimplência (base PEJOTA)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

from .config import PROFUNDIDADES_ARVORE, TABLE_DIR, THRESHOLD
from .data import load_pejota, make_split
from .preprocessing import feature_names, make_logit, make_preprocessor, make_tree, transformar
from .utils import header, save_figure, save_table

AZUL, VERMELHO, CINZA = "#2196F3", "#F44336", "#B0BEC5"


def ajustar_statsmodels(pipe, X, y):
    """Logit sem penalização (MLE) nos mesmos dados padronizados, para p-valores e ICs."""
    Z = sm.add_constant(transformar(pipe, X))
    return sm.Logit(y.to_numpy(), Z).fit(disp=False)


def tabela_or(coef_sk: pd.Series, res_sm) -> pd.DataFrame:
    ic = res_sm.conf_int()
    return pd.DataFrame({
        "beta (sklearn)": coef_sk,
        "OR (sklearn)": np.exp(coef_sk),
        "beta (MLE)": res_sm.params.drop("const"),
        "OR (MLE)": np.exp(res_sm.params.drop("const")),
        "OR IC95% inf": np.exp(ic[0].drop("const")),
        "OR IC95% sup": np.exp(ic[1].drop("const")),
        "p-valor": res_sm.pvalues.drop("const"),
    })


def plot_coeficientes(coef: pd.Series) -> None:
    c = coef.sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(c.index, c.values, color=[VERMELHO if v > 0 else AZUL for v in c.values])
    ax.axvline(0, color="k", lw=1)
    for i, v in enumerate(c.values):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("β (por 1 desvio padrão)")
    ax.set_title("(a) Coeficientes da regressão logística\n"
                 "vermelho: aumenta P(DEFAULT) | azul: reduz")
    fig.tight_layout()
    save_figure(fig, "q9a_coeficientes")


def plot_or_top5(tab: pd.DataFrame) -> None:
    t = tab.sort_values("OR (MLE)")
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(t))
    ax.errorbar(t["OR (MLE)"], y,
                xerr=[t["OR (MLE)"] - t["OR IC95% inf"], t["OR IC95% sup"] - t["OR (MLE)"]],
                fmt="o", color="k", capsize=4)
    ax.scatter(t["OR (MLE)"], y, s=60, zorder=3,
               color=[VERMELHO if v > 1 else AZUL for v in t["OR (MLE)"]])
    ax.axvline(1, color="grey", ls="--")
    ax.set_yticks(y, t.index)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio por +1 desvio padrão (escala log, IC 95%)")
    ax.set_title("(b) Odds ratios dos 5 maiores |β|")
    fig.tight_layout()
    save_figure(fig, "q9b_odds_ratios")


def plot_matriz(cm, acc) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(cm, display_labels=["BOM (0)", "MAU (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(f"(d) Matriz de confusão (corte {THRESHOLD})\nacurácia no teste = {acc:.1%}")
    fig.tight_layout()
    save_figure(fig, "q9d_matriz_confusao")


def plot_lpm(prob_lin: np.ndarray, prob_log: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(min(prob_lin.min(), 0) - 0.05, max(prob_lin.max(), 1) + 0.05, 40)
    ax.hist(prob_lin, bins=bins, alpha=0.6, color="#FF9800", label="Regressão linear (MQO)")
    ax.hist(prob_log, bins=bins, alpha=0.6, color=AZUL, label="Regressão logística")
    ax.axvspan(bins[0], 0, color="red", alpha=0.08)
    ax.axvspan(1, bins[-1], color="red", alpha=0.08, label="Fora de [0, 1]")
    ax.set_xlabel("Probabilidade prevista de DEFAULT (conjunto de teste)")
    ax.set_ylabel("Nº de empresas")
    ax.set_title("(e) Regressão linear pode prever 'probabilidades' fora de [0, 1]")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "q9e_linear_vs_logistica")


def plot_roc(curvas: dict, y_te) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for nome, prob in curvas.items():
        fpr, tpr, _ = roc_curve(y_te, prob)
        ax.plot(fpr, tpr, lw=2, label=f"{nome} (AUC = {roc_auc_score(y_te, prob):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Aleatório (0,500)")
    ax.set_xlabel("Taxa de falsos positivos (bons recusados)")
    ax.set_ylabel("Taxa de verdadeiros positivos (maus detectados)")
    ax.set_title("(f) Curvas ROC no teste")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "q9f_roc")


def taxa(df, cond) -> str:
    return f"{df.loc[cond, 'DEFAULT'].mean():.0%} (n={int(cond.sum())})"


def run() -> dict:
    header("QUESTÃO 9 - Regressão logística (PEJOTA)")
    df = load_pejota()
    split = make_split(df)
    print(f"Observações: {len(df)} | % MAU: {df['DEFAULT'].mean():.1%}")
    print(f"Treino: {len(split.y_tr)} | Teste: {len(split.y_te)}")

    # (a)
    logit = make_logit(split.X_tr).fit(split.X_tr, split.y_tr)
    nomes = feature_names(logit)
    coef = pd.Series(logit.named_steps["modelo"].coef_[0], index=nomes)
    res_sm = ajustar_statsmodels(logit, split.X_tr, split.y_tr)
    (TABLE_DIR / "q9a_summary_statsmodels.txt").write_text(str(res_sm.summary()), encoding="utf-8")
    print(f"\n(a) {len(nomes)} preditores, padronizados com média/desvio do treino")
    print(f"Pseudo R² (McFadden) = {res_sm.prsquared:.3f} | LR p-valor = {res_sm.llr_pvalue:.1e}")
    print("Coeficientes positivos (aumentam P(DEFAULT)):",
          ", ".join(coef[coef > 0].sort_values(ascending=False).index))
    print("Coeficientes negativos (reduzem P(DEFAULT)):",
          ", ".join(coef[coef < 0].sort_values().index))
    print("Significativos a 5% (MLE):",
          ", ".join(res_sm.pvalues.drop("const")[lambda p: p < 0.05].index))
    plot_coeficientes(coef)

    # (b)
    tab = tabela_or(coef, res_sm)
    save_table(tab.round(4), "q9b_odds_ratios_todos")
    top5 = tab.loc[coef.abs().nlargest(5).index]
    save_table(top5.round(4), "q9b_odds_ratios_top5")
    print("\n(b) Top 5 por |β|:")
    print(top5.round(3).to_string())
    plot_or_top5(top5)
    print("Taxa de inadimplência observada (base completa):")
    print(f"  REST_TIPO_1 = 0: {taxa(df, df['REST_TIPO_1'] == 0)} | >= 3: {taxa(df, df['REST_TIPO_1'] >= 3)}")
    print(f"  REST_TIPO_3 = 0: {taxa(df, df['REST_TIPO_3'] == 0)} | >= 1: {taxa(df, df['REST_TIPO_3'] >= 1)}")
    print(f"  corr(REST_TIPO_1, REST_TIPO_3) = {df['REST_TIPO_1'].corr(df['REST_TIPO_3']):.2f}")
    print(f"  IFIN4 = 0: {taxa(df, df['IFIN4'] == 0)} | > 0: {taxa(df, df['IFIN4'] > 0)}")
    print(f"  IDADE <= 22 meses: {taxa(df, df['IDADE'] <= 22)} | > 44 meses: {taxa(df, df['IDADE'] > 44)}")

    # (c)
    prob_tr = logit.predict_proba(split.X_tr)[:, 1]
    prob_te = logit.predict_proba(split.X_te)[:, 1]
    pred_te = (prob_te >= THRESHOLD).astype(int)
    acc_tr = accuracy_score(split.y_tr, prob_tr >= THRESHOLD)
    acc_te = accuracy_score(split.y_te, pred_te)
    base = DummyClassifier(strategy="most_frequent").fit(split.X_tr, split.y_tr)
    acc_base = accuracy_score(split.y_te, base.predict(split.X_te))
    print(f"\n(c) Acurácia treino = {acc_tr:.1%} | teste = {acc_te:.1%} | baseline = {acc_base:.1%}")

    # (d)
    cm = confusion_matrix(split.y_te, pred_te)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n(d) Matriz [[VN, FP], [FN, VP]] = {cm.tolist()}")
    print(f"Bons recusados (FP) = {fp} | Maus aprovados (FN) = {fn}")
    print(f"Sensibilidade = {tp / (tp + fn):.1%} | Especificidade = {tn / (tn + fp):.1%} | "
          f"Precisão (MAU) = {tp / (tp + fp):.1%}")
    print(classification_report(split.y_te, pred_te, target_names=["BOM (0)", "MAU (1)"]))
    plot_matriz(cm, acc_te)

    # (e) Ilustração: modelo de probabilidade linear
    from sklearn.pipeline import Pipeline
    lpm = Pipeline([("prep", make_preprocessor(split.X_tr)), ("modelo", LinearRegression())])
    lpm.fit(split.X_tr, split.y_tr)
    p_lin = lpm.predict(split.X_te)
    fora = (p_lin < 0) | (p_lin > 1)
    print(f"\n(e) Regressão linear: {fora.sum()} de {len(p_lin)} previsões no teste fora de [0, 1] "
          f"(mín = {p_lin.min():.3f}, máx = {p_lin.max():.3f})")
    plot_lpm(p_lin, prob_te)

    # (f) Comparação com as árvores da lista anterior
    linhas = [{"modelo": "Regressão logística", "acc_teste": acc_te,
               "AUC_teste": roc_auc_score(split.y_te, prob_te)}]
    curvas = {"Logística": prob_te}
    for p in PROFUNDIDADES_ARVORE:
        arv = make_tree(split.X_tr, p).fit(split.X_tr, split.y_tr)
        pa = arv.predict_proba(split.X_te)[:, 1]
        linhas.append({"modelo": f"Árvore profundidade {p}",
                       "acc_teste": accuracy_score(split.y_te, pa >= 0.5),
                       "AUC_teste": roc_auc_score(split.y_te, pa)})
        curvas[f"Árvore prof. {p}"] = pa
    linhas.append({"modelo": "Baseline", "acc_teste": acc_base, "AUC_teste": 0.5})
    comp = pd.DataFrame(linhas).set_index("modelo")
    save_table(comp.round(4), "q9f_logistica_vs_arvores")
    print("\n(f) Comparação no teste:")
    print(comp.round(3).to_string())
    plot_roc(curvas, split.y_te)

    return {"df": df, "split": split, "logit": logit, "res_sm": res_sm, "coef": coef,
            "prob_te": prob_te, "comp": comp}
