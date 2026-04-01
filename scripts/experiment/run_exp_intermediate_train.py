"""Phase 3: Train BridgeAnchors K=128 on top-3 intermediate layer pairs.

Patches the config to point to the correct embedding files.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LAYER_PAIRS = [
    {"name": "b9_l10", "dino_block": 9, "mpnet_layer": 10, "cka": 0.586},
    {"name": "b9_l11", "dino_block": 9, "mpnet_layer": 11, "cka": 0.552},
    {"name": "b10_l10", "dino_block": 10, "mpnet_layer": 10, "cka": 0.514},
]


def main() -> None:
    exp_dir = PROJECT_ROOT / "experiments" / "exp_intermediate_layer"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
        base_cfg = yaml.safe_load(f)

    for pair in LAYER_PAIRS:
        name = pair["name"]
        blk = pair["dino_block"]
        lay = pair["mpnet_layer"]

        logger.info("=" * 60)
        logger.info("Training on DINOv2 Block %d × MPNet Layer %d (CKA=%.3f)",
                    blk, lay, pair["cka"])
        logger.info("=" * 60)

        # Create a temporary config with patched embedding paths
        cfg = base_cfg.copy()
        cfg["data"] = dict(base_cfg["data"])
        cfg["data"]["img_emb_path"] = f"data/embeddings/cls/coco_train_img_dino_block{blk}.pt"
        cfg["data"]["txt_emb_path"] = f"data/embeddings/cls/coco_train_txt_mpnet_layer{lay}.pt"
        cfg["eval"] = dict(base_cfg["eval"])
        cfg["eval"]["flickr_img_emb_path"] = f"data/embeddings/cls/flickr30k_test_img_dino_block{blk}.pt"
        cfg["eval"]["flickr_txt_emb_path"] = f"data/embeddings/cls/flickr30k_test_txt_mpnet_layer{lay}.pt"

        tmp_config = exp_dir / f"config_{name}.yaml"
        with open(tmp_config, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)

        # Run training
        log_path = exp_dir / f"train_{name}.log"
        cmd = [
            sys.executable, "-m", "src.train",
            "--config", str(tmp_config),
            "--model", "bridge_anchors",
            "--num-anchors", "128",
            "--seed", "42",
            "--experiment-name", f"exp_intlayer_{name}",
        ]

        logger.info("Running: %s", " ".join(cmd))
        with open(log_path, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                   cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            logger.error("Training failed for %s (exit %d). See %s", name, result.returncode, log_path)
        else:
            logger.info("Training complete for %s. Log: %s", name, log_path)

    logger.info("All intermediate layer training runs complete.")


if __name__ == "__main__":
    main()
