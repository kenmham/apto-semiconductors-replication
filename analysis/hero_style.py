import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


FRONTIER = "#0072B2"
TYPICAL  = "#E69F00"
NULL     = "#8C8C8C"
SIGNAL   = "#D55E00"
ACCENT   = "#009E73"
ACCENT2  = "#CC79A7"
INK      = "#222222"
FAINT    = "#CFCFCF"


SIZES = {
    "poster": dict(dpi=300, base=15, scale=1.55),
    "slide":  dict(dpi=200, base=13, scale=1.0),
}


def apply(size):

    s = SIZES[size]
    b = s["base"] * s["scale"]
    plt.rcParams.update({
        "figure.dpi": s["dpi"],
        "savefig.dpi": s["dpi"],
        "font.family": "DejaVu Sans",
        "font.size": b,
        "axes.titlesize": b * 1.18,
        "axes.titlepad": b * 0.9,
        "axes.labelsize": b * 1.02,
        "xtick.labelsize": b * 0.9,
        "ytick.labelsize": b * 0.9,
        "legend.fontsize": b * 0.92,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9 * s["scale"],
        "axes.grid": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "lines.linewidth": 2.4 * s["scale"],
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return s


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _natural(x, _pos=None):

    if x == 0:
        return "0"
    neg = x < 0
    x = abs(x)
    units = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1, "")]
    for thr, suf in units:
        if x >= thr:
            v = x / thr
            s = f"{v:.0f}{suf}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}{suf}"
            return ("-" if neg else "") + s

    small = [(1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p")]
    for thr, suf in small:
        if x >= thr:
            v = x / thr
            s = f"{v:.0f}{suf}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}{suf}"
            return ("-" if neg else "") + s
    return ("-" if neg else "") + f"{x:.1e}"


NAT = FuncFormatter(_natural)


def log_natural(ax, which="y"):

    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_formatter(NAT)


def new(size, w, h):
    s = apply(size)
    fig, ax = plt.subplots(figsize=(w * s["scale"], h * s["scale"]))
    return fig, ax


def render(build, stem, outdir="images", sizes=("poster", "slide")):


    os.makedirs(outdir, exist_ok=True)
    written = []
    for size in sizes:
        fig = build(size)
        path = os.path.join(outdir, f"{stem}_{size}.png")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(path)
    return written
