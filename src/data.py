"""Leitura e limpeza da base PEJOTA e divisão treino/teste."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import COLS_EXCLUIDAS, DATA_PATH, SEED, TEST_SIZE

COLS_COM_BRANCOS = ["IFIN1", "IFIN2", "IFIN4", "IFIN5", "IFIN7", "IFIN8"]


def load_pejota() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="PEJOTA")
    df.columns = df.columns.str.strip()                     # "IDADE " -> "IDADE"
    df[COLS_COM_BRANCOS] = (df[COLS_COM_BRANCOS]
                            .replace(r"^\s*$", np.nan, regex=True)
                            .apply(pd.to_numeric, errors="coerce"))
    df["UF"] = df["UF"].astype(str).str.strip()
    df["DEFAULT"] = (df["STATUS"].astype(str).str.strip() == "MAU").astype(int)
    return df


def features_e_alvo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    faltando = [c for c in COLS_EXCLUIDAS if c not in df.columns]
    if faltando:
        raise KeyError(f"Colunas não encontradas: {faltando}")
    return df.drop(columns=COLS_EXCLUIDAS), df["DEFAULT"]


@dataclass
class Split:
    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: pd.Series
    y_te: pd.Series


def make_split(df: pd.DataFrame, seed: int = SEED) -> Split:
    """Mesma divisão da lista de árvores (70/30, estratificada, semente 42)."""
    X, y = features_e_alvo(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    return Split(X_tr, X_te, y_tr, y_te)
