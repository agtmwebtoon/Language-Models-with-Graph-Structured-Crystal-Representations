# visualize_embedding.py
# ------------------------------------------------------------
# Usage:
#   CUDA_VISIBLE_DEVICES=1 python visualize_embedding.py --cfg runspec.json
#
# - cfg는 src/utils/config.py의 Config.build()로 로드/override
# - viz 파라미터는 cfg.visualization.* 에서만 읽음
# - latest run 선택은 GraphTextCLIPVisualizer 내부에서 처리한다고 가정
# ------------------------------------------------------------

import argparse
import os

# Set environment variables before importing transformers
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.utils.config import Config
from src.visualization import GraphTextCLIPVisualizer, VizRunConfig
from src.visualization.plots import TSNEConfig


def build_viz_run_config_from_cfg(cfg: Config) -> VizRunConfig:
    """
    Map your Config.VisualizationConfig fields to VizRunConfig/TSNEConfig.
    Your VisualizationConfig:
      - n_samples_tsne
      - tsne_perplexity
      - tsne_seed
      - similarity_batch_size
      - (output_dir is set by visualizer/latest-run-paths)
    """
    v = cfg.visualization

    # 필드명은 config.py 그대로 사용
    n_samples = int(getattr(v, "n_samples_tsne", 800))
    sim_batch = int(getattr(v, "similarity_batch_size", 64))
    perplexity = float(getattr(v, "tsne_perplexity", 30))
    seed = int(getattr(v, "tsne_seed", 42))

    loader_batch_size = int(getattr(v, "loader_batch_size", 128))
    connect_k = int(getattr(v, "connect_k", 200))
    seed_subset = int(getattr(v, "seed_subset", 123))

    tsne_cfg = TSNEConfig(perplexity=perplexity, seed=seed, connect_k=connect_k)

    return VizRunConfig(
        n_samples=n_samples,
        sim_batch=sim_batch,
        tsne=tsne_cfg,
        seed_subset=seed_subset,
        loader_batch_size=loader_batch_size,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to cfg json")
    args = ap.parse_args()

    cfg = Config.build(args.cfg)

    run_cfg = build_viz_run_config_from_cfg(cfg)

    viz = GraphTextCLIPVisualizer(cfg, run_cfg=run_cfg)
    out = viz.run()
    print(out)


if __name__ == "__main__":
    main()
