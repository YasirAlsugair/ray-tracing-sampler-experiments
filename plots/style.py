"""Shared matplotlib style. Warm palette matching the explainer video
(RT gold, HMC red, accent blue). No em dashes anywhere in labels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RT = "#F4B942"      # ray tracing  (gold)
HMC = "#E0533D"     # HMC          (red)
BLUE = "#6E8BE8"    # accent
TRUTH = "#444444"   # truth / reference
GREY = "#9aa0a6"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLOR = {"rt": RT, "hmc": HMC}
LABEL = {"rt": "Ray tracing", "hmc": "HMC"}
