"""Executa a análise completa de regressão logística.

Uso:
    python main.py                    # tudo
    python main.py --sem-complementos # só os itens do enunciado
"""
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")

from src import q9_complementos, q9_inadimplencia, q10_iris  # noqa: E402
from src.config import FIG_DIR, TABLE_DIR, ensure_dirs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sem-complementos", action="store_true",
                        help="não roda as análises complementares da Questão 9")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()

    q9 = q9_inadimplencia.run()
    if not args.sem_complementos:
        q9_complementos.run(q9)
    q10_iris.run()

    print(f"\nTabelas: {TABLE_DIR}\nFiguras: {FIG_DIR}")


if __name__ == "__main__":
    main()
