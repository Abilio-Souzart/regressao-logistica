"""Pipelines de pré-processamento e modelos.

Tudo que é aprendido com dados (mediana da imputação, média e desvio da
padronização, categorias de UF) é ajustado apenas no treino.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import C_LOGIT, COLS_CATEGORICAS, SEED


def signed_log(x):
    """sign(x) * log(1 + |x|): comprime caudas longas e aceita valores negativos e zero."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(X: pd.DataFrame, transformacao: str = "padrao") -> ColumnTransformer:
    """transformacao: 'padrao' (imputação + padronização, como no notebook),
    'log' (imputação + log com sinal + padronização) ou 'nenhuma' (só imputação)."""
    numericas = [c for c in X.columns if c not in COLS_CATEGORICAS]
    passos = [SimpleImputer(strategy="median")]
    if transformacao == "log":
        passos.append(FunctionTransformer(signed_log, feature_names_out="one-to-one"))
    if transformacao in ("padrao", "log"):
        passos.append(StandardScaler())
    # UF também é padronizada no modelo 'padrao', como no notebook original
    cat = OneHotEncoder(drop="first", sparse_output=False)
    cat_pipe = make_pipeline(cat, StandardScaler()) if transformacao != "nenhuma" else cat
    return ColumnTransformer(
        [("num", make_pipeline(*passos), numericas), ("cat", cat_pipe, COLS_CATEGORICAS)],
        verbose_feature_names_out=False,
    )


def make_logit(X: pd.DataFrame, transformacao: str = "padrao", C: float = C_LOGIT) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(X, transformacao)),
        ("modelo", LogisticRegression(C=C, max_iter=5000, random_state=SEED)),
    ])


def make_tree(X: pd.DataFrame, max_depth: int) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(X, "nenhuma")),
        ("modelo", DecisionTreeClassifier(max_depth=max_depth, random_state=SEED)),
    ])


def feature_names(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())


def transformar(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Matriz já pré-processada, com nomes de colunas (útil para o statsmodels)."""
    return pd.DataFrame(pipe.named_steps["prep"].transform(X),
                        columns=feature_names(pipe), index=X.index)
