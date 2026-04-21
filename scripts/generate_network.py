from pathlib import Path

from msean import load_config, save_config
from msean.generation import gen
from msean.measurements.properties import get_degree_dist, get_degree_dist_layers, get_embeddedness
from msean.io.save import prepare_run_directory

if __name__ == "__main__":
    # Load configuration from configs/default.yaml
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")

    # Save configuration used to runs/ subfolder
    paths = prepare_run_directory(cfg)
    save_config(cfg, paths["config"])

    # Generate model
    G, layers = gen(cfg)

    # Generate and save plots

    # Degree distribution
    deg_fig = get_degree_dist(G, show=cfg.output.plots.show)
    if cfg.output.plots.save:
        deg_fig.savefig(paths["plots"] / "degree_distribution.png")

    # Degree distribution per layer
    deg_layer_fig = get_degree_dist_layers(layers, cfg.network.layer_labels, show=cfg.output.plots.show)
    if cfg.output.plots.save:
        deg_layer_fig.savefig(paths["plots"] / "degree_distribution_layers.png")

    # Embededness
    embed_fig = get_embeddedness(G, show=cfg.output.plots.show)
    if cfg.output.plots.save:
        embed_fig.savefig(paths["plots"] / "embededness.png")

    # TODO: Save degree distribution DataFrame in a csv file?