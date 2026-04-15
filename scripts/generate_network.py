from pathlib import Path

from msean import load_config, save_config
from msean.generation import gen
from msean.visualization.plots import plot_degree_dist, plot_degree_dist_layers, plot_embeddedness
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
    if cfg.output.plots.degree_dist:
        deg_fig, deg_df = plot_degree_dist(G, show=cfg.output.plots.show)
        deg_df.to_csv(paths["csv_files"] / "degree_distribution.csv")
        if cfg.output.plots.save:
            deg_fig.savefig(paths["plots"] / "degree_distribution.png")

    # Degree distribution per layer
    if cfg.output.plots.degree_dist_layers:
        deg_layer_fig, deg_layer_dfs = plot_degree_dist_layers(layers, cfg.network.layer_labels, show=cfg.output.plots.show)
        for i in range(len(layers)):
            deg_layer_dfs[i].to_csv(paths["csv_files"] / f"degree_distribution_{cfg.network.layer_labels[i]}.csv")
        if cfg.output.plots.save:
            deg_layer_fig.savefig(paths["plots"] / "degree_distribution_layers.png")

    # Embededness
    if cfg.output.plots.embededness:
        embed_fig,embed_df = plot_embeddedness(G, show=cfg.output.plots.show)
        embed_df.to_csv(paths["csv_files"] / "embededness.csv")
        if cfg.output.plots.save:
            embed_fig.savefig(paths["plots"] / "embededness.png")