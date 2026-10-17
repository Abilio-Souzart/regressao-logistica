import matplotlib.pyplot as plt
import pandas as pd

from .config import FIG_DIR, TABLE_DIR


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    df.to_csv(TABLE_DIR / f"{name}.csv", index=index)


def save_figure(fig, name: str, dpi: int = 110) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def header(title: str) -> None:
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)
