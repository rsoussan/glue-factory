import argparse
from disk import DISK
from utils import load_image, rbd
from torchvision.utils import save_image
import viz2d
import torch
import sys
import random
import numpy as np
import os
from pathlib import Path
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from gluefactory.models.matchers.lightglue import LightGlue
from gluefactory.geometry.depth import sample_depth
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def filter_matches_by_score(matches: torch.Tensor, scores: torch.Tensor, th: float):
    """
    Split matches into valid and invalid based on score threshold.

    Args:
        matches: Tensor of shape [N, 2], each row = (kp0_idx, kp1_idx)
        scores: Tensor of shape [N], confidence score for each match
        th: float, threshold

    Returns:
        valid_matches: Tensor [M, 2], scores above threshold
        valid_scores: Tensor [M]
        invalid_matches: Tensor [K, 2], scores below or equal threshold
        invalid_scores: Tensor [K]
    """
    mask = scores > th
    valid_matches = matches[mask]
    valid_scores = scores[mask]
    invalid_matches = matches[~mask]
    invalid_scores = scores[~mask]
    return valid_matches, valid_scores, invalid_matches, invalid_scores

def load_and_check_model(model, ckpt_path, key="model", strict=False, map_location="cpu"):
    """
    Load a checkpoint into a model and verify which layers matched.
    
    Args:
        model: torch.nn.Module
        ckpt_path: str, path to checkpoint (.pth file)
        key: str, key in checkpoint dict containing state_dict
        strict: bool, enforce exact key match
        map_location: str or torch.device
    
    Returns:
        model: loaded model
        report: dict with missing/unexpected/matching keys and param counts
    """
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = checkpoint[key] if key in checkpoint else checkpoint

    # Load with requested strictness
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    # Compare keys
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())
    matching = sorted(list(model_keys & ckpt_keys))

    # Count parameters
    model_params = sum(p.numel() for p in model.parameters())
    ckpt_params = sum(v.numel() for v in state_dict.values())

    # Report
    report = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "matching_keys": matching,
        "model_param_count": model_params,
        "ckpt_param_count": ckpt_params,
        "all_layers_match": len(missing) == 0 and len(unexpected) == 0 and model_params == ckpt_params
    }

    # Print summary
    print("Model params:", model_params)
    print("Checkpoint params:", ckpt_params)

    if missing:
        print("\nMissing keys:")
        for k in missing:
            print(f"  {k} | model shape: {tuple(model_state[k].shape)}")

    if unexpected:
        print("\nUnexpected keys:")
        for k in unexpected:
            print(f"  {k} | checkpoint shape: {tuple(state_dict[k].shape)}")

    print(f"\nMatching keys ({len(matching)}):")
    for k in matching:
        print(f"  {k} | model shape: {tuple(model_state[k].shape)} | checkpoint shape: {tuple(state_dict[k].shape)}")

    return model, report

def get_kp_depth(keypoints, depth):
    print(f"keytpoints shape: {keypoints.shape}, depth shape: {depth.shape}")
    d, valid = sample_depth(keypoints, depth)
    return d

# --- Updated plot_images that accepts ax ---
def plot_images(imgs, titles=None, cmaps="gray", ax=None):
    """
    Plot a set of images horizontally.
    Args:
        imgs: list of NumPy RGB (H, W, 3) or PyTorch RGB (3, H, W) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        ax: list of matplotlib axes to draw on. If None, creates new fig/axes.
    """
    imgs = [
        (
            img.permute(1, 2, 0).cpu().numpy()
            if (isinstance(img, torch.Tensor) and img.dim() == 3)
            else img
        )
        for img in imgs
    ]
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if ax is None:
        fig, ax = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
        if n == 1:
            ax = [ax]
    else:
        if n == 1:
            ax = [ax]

    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])

    return ax


def main(input_dir):
    # Inits
    seed = 42
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup extractor and matcher
    extractor = DISK(max_num_keypoints=2048).eval().to(device)
    conf = LightGlue.default_conf
    conf["input_dim"] = 128
    conf["filter_threshold"] = 0
    conf["weights"] = "/usr/local/home/rsoussan/glue-factory/outputs/training/bartlett/ryan_tmp_disk+lg_megadepth/checkpoint_best.pth"
    matcher = LightGlue(conf).eval().to(device)

    input_dir = Path(input_dir)
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    avg_valid_percents = []

    with PdfPages("matches_report.pdf") as pdf:
        for subdir in sorted(subdirs, key=lambda d: int(d.name)):
            print(f"Processing {subdir}")

            # Load data
            image0 = load_image(str(subdir / "image0.png"))
            image1 = load_image(str(subdir / "image1.png"))
            depth0 = torch.load(str(subdir / "depth0.pt"))
            depth1 = torch.load(str(subdir / "depth1.pt"))

            feats0 = extractor.extract(image0.to(device))
            feats1 = extractor.extract(image1.to(device))

            data = {
                "keypoints0": feats0["keypoints"],
                "keypoints1": feats1["keypoints"],
                "descriptors0": feats0["descriptors"],
                "descriptors1": feats1["descriptors"],
            }
            data["depth_keypoints0"] = get_kp_depth(data["keypoints0"], depth0.to(device))
            data["depth_keypoints1"] = get_kp_depth(data["keypoints1"], depth1.to(device))
            data["overlap_0to1"] = 0.3

            w = h = image0.shape[1]
            data["view0"] = {"image_size": [w, h], "image": image0}
            data["view1"] = {"image_size": [w, h], "image": image1}

            # Predict
            pred = matcher(data)
            matches = pred["matches"][0]
            scores = pred["scores"][0]

            kpts0 = data["keypoints0"].squeeze(0)
            kpts1 = data["keypoints1"].squeeze(0)

            # Filter matches
            match_threshold = 0.01
            valid_matches, valid_scores, invalid_matches, invalid_scores = filter_matches_by_score(
                matches, scores, match_threshold
            )
            avg_valid_percents.append(
                100 * len(valid_matches) / len(matches) if len(matches) > 0 else 0
            )

            # Extract images for plotting
            img0 = data["view0"]["image"][0].cpu()
            img1 = data["view1"]["image"][0].cpu()

            # --- New page with valid + invalid stacked ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # Valid
            plot_images([img0, img1], ax=axes[0])
            if len(valid_matches) > 0:
                m_kpts0, m_kpts1 = kpts0[valid_matches[..., 0]], kpts1[valid_matches[..., 1]]
                viz2d.plot_matches(m_kpts0, m_kpts1, axes=(axes[0][0], axes[0][1]), color="lime", lw=0.2)
            axes[0][0].set_title(f"{subdir.name} –Threshold: {match_threshold}")
            axes[0][1].set_title(f"{subdir.name} – Image1 (Valid Matches: {len(valid_matches)})")

            # Images 
            plot_images([img0, img1], ax=axes[1])

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Invalid 
            plot_images([img0, img1], ax=axes[0])
            if len(invalid_matches) > 0:
                m_kpts0, m_kpts1 = kpts0[invalid_matches[..., 0]], kpts1[invalid_matches[..., 1]]
                viz2d.plot_matches(m_kpts0, m_kpts1, axes=(axes[0][0], axes[0][1]), color="lime", lw=0.2)
            axes[0][0].set_title(f"{subdir.name} –Threshold: {match_threshold}")
            axes[0][1].set_title(f"{subdir.name} – Image1 Invalid Matches: {len(invalid_matches)})")

            # Images 
            plot_images([img0, img1], ax=axes[1])

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


        # --- Histogram page ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(avg_valid_percents, bins=np.arange(0, 110, 10), color="skyblue", edgecolor="black")
        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_xlabel("Average Valid Match % (binned by 10%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Avg Valid Match % Across Subdirectories")
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to input directory containing subfolders")
    args = parser.parse_args()
    main(args.input_dir)
