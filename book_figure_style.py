from matplotlib import font_manager
from matplotlib import pyplot as plt


PALETTE = {
    "navy": "#1B365D",
    "blue": "#0072B2",
    "teal": "#00866A",
    "orange": "#C76D00",
    "magenta": "#A03A78",
    "red": "#B83B3B",
    "gray": "#5C6670",
    "light_gray": "#D9DEE5",
    "near_white": "#F4F6F8",
    "white": "#FFFFFF",
}

THEME_STATUS = "APPROVED_FOR_BOOK_WIDE_USE"
THEME_APPROVED_AT = "2026-10-02"


def selected_font():
    for family in ["Times New Roman", "Times", "DejaVu Serif"]:
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Serif"


def apply_book_figure_style():
    family = selected_font()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 11.5,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "axes.edgecolor": PALETTE["gray"],
            "axes.labelcolor": PALETTE["navy"],
            "axes.titlecolor": PALETTE["navy"],
            "axes.titleweight": "bold",
            "xtick.color": PALETTE["gray"],
            "ytick.color": PALETTE["gray"],
            "text.color": PALETTE["navy"],
            "grid.color": PALETTE["light_gray"],
            "grid.linestyle": ":",
            "grid.linewidth": 0.8,
            "legend.facecolor": PALETTE["white"],
            "legend.edgecolor": PALETTE["light_gray"],
            "legend.framealpha": 0.96,
            "savefig.facecolor": PALETTE["white"],
            "savefig.transparent": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return family


def finish_axis(axis, grid=True):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(PALETTE["gray"])
    axis.spines["bottom"].set_color(PALETTE["gray"])
    if grid:
        axis.grid(True, alpha=0.75)
    axis.set_axisbelow(True)


def save_figure(figure, output_root, stem):
    figure.savefig(output_root / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=PALETTE["white"], transparent=False)
    figure.savefig(output_root / f"{stem}.pdf", bbox_inches="tight", facecolor=PALETTE["white"], transparent=False)
