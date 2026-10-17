"""Caminhos e parâmetros globais."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "base_pejota.xls"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

SEED = 42
TEST_SIZE = 0.30
N_FOLDS = 10
THRESHOLD = 0.50
C_LOGIT = 1.0                 # regularização L2 padrão do scikit-learn (igual ao notebook)
PROFUNDIDADES_ARVORE = [2, 3, 4]   # árvores da lista anterior usadas na comparação

COLS_EXCLUIDAS = ["COD_CLI", "STATUS", "DEFAULT"]
COLS_CATEGORICAS = ["UF"]

# Custo relativo dos erros (análise complementar): aprovar um mau pagador
# custa CUSTO_FN vezes mais do que recusar um bom.
CUSTO_FN = 5
CUSTO_FP = 1


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
