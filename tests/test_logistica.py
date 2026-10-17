"""Testes do pré-processamento e de propriedades usadas na análise."""
import numpy as np
import pytest

from src.data import features_e_alvo, load_pejota, make_split
from src.preprocessing import feature_names, make_logit, signed_log
from src.q10_iris import ajustar, carregar


@pytest.fixture(scope="module")
def split():
    return make_split(load_pejota())


def test_identificador_e_alvo_fora_dos_preditores():
    X, y = features_e_alvo(load_pejota())
    assert not {"COD_CLI", "STATUS", "DEFAULT"} & set(X.columns)
    assert y.mean() == pytest.approx(0.5)


def test_imputacao_e_padronizacao_so_com_treino(split):
    pipe = make_logit(split.X_tr).fit(split.X_tr, split.y_tr)
    num = pipe.named_steps["prep"].named_transformers_["num"]
    cols = [c for c in split.X_tr.columns if c != "UF"]
    assert np.allclose(num.named_steps["simpleimputer"].statistics_, split.X_tr[cols].median())
    assert len(feature_names(pipe)) == 21


def test_signed_log():
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    y = signed_log(x)
    assert y[2] == 0
    assert np.allclose(y, -y[::-1])                 # simétrica
    assert np.all(np.diff(y) > 0)                   # preserva a ordem


def test_iris_separacao_perfeita():
    X, y = carregar()
    col = X.columns[0]
    assert X.loc[y == 1, col].max() < X.loc[y == 0, col].min()


def test_iris_coeficiente_cresce_sem_regularizacao():
    """Com separação perfeita, |b1| cresce sem limite quando a penalização diminui."""
    X, y = carregar()
    b1 = [abs(ajustar(X, y, C)[1]) for C in (0.1, 1, 10, 100)]
    assert all(a < b for a, b in zip(b1, b1[1:]))
