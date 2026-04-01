from pathlib import Path

from msean import load_config
from msean.generation import gen
from msean.visualization.plots import plot_degree_dist

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")

    G = gen(cfg)
    plot_degree_dist(G, show=True)