from disk import DISK
from utils import load_image, rbd
from torchvision.utils import save_image
import viz2d
import torch
import sys
import random
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from gluefactory.models.matchers.lightglue_pretrained import LightGlue as LightGluePretrained
from gluefactory.models.matchers.lightglue import LightGlue 
from gluefactory.geometry.depth import sample_depth
from scipy.spatial.transform import Rotation as R

import torch

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

def balance_keypoints_by_scores(features0, features1):
    """
    Ensures features0 and features1 have the same number of keypoints/descriptors/scores
    by removing the lowest-scoring ones from the larger set.
    Operates in-place on the passed dicts.
    """
    scores0 = features0['scores']
    scores1 = features1['scores']
    n0, n1 = len(scores0), len(scores1)

    if n0 == n1:
        return  # already balanced

    if n0 > n1:
        num_keep = n1
        keep_idx = torch.topk(scores0, num_keep, largest=True).indices
        for k in ['keypoints', 'descriptors', 'scores']:
            features0[k] = features0[k][keep_idx]
    else:
        num_keep = n0
        keep_idx = torch.topk(scores1, num_keep, largest=True).indices
        for k in ['keypoints', 'descriptors', 'scores']:
            features1[k] = features1[k][keep_idx]

def crop_top_black_rows(img, keypoints, threshold_ratio=0.02):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    non_black_counts = np.count_nonzero(gray > 0, axis=1)
    row_threshold = threshold_ratio * w

    valid_rows = np.where(non_black_counts >= row_threshold)[0]
    if len(valid_rows) > 0:
        crop_top = int(valid_rows[0])
        cropped_img = img[crop_top:, :]
        cropped_keypoints = keypoints.copy()
        cropped_keypoints[:, 1] -= crop_top
    else:
        cropped_img = img
        cropped_keypoints = keypoints.copy()

    return cropped_img, cropped_keypoints


def save_matches_in_warped_view(
    img0, img1, kpts0, kpts1, K, R, save_path="warped_matches.png"
):
    """
    Warps img1 into img0's view using rotation R and shared intrinsics K.
    Then, projects and draws matching keypoints in this warped view,
    scaling and shifting the warped result to fit the original image size.
    """

    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    img0, img1 = to_numpy(img0), to_numpy(img1)
    h, w = img0.shape[:2]

    threshold_ratio = 0.5
    img0, kpts0 = crop_top_black_rows(img0, kpts0, threshold_ratio)
    img1, kpts1 = crop_top_black_rows(img1, kpts1, threshold_ratio)

    # Compute homography for pure rotation
    H = K @ R @ np.linalg.inv(K)

    # Warp the corners of the image to find extents
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [w, h, 1],
                        [0, h, 1]], dtype=np.float32).T  # 3x4
    warped_corners = H @ corners
    warped_corners /= warped_corners[2, :]
    warped_corners = warped_corners[:2, :].T  # 4x2

    # Compute bounding box of warped image
    min_xy = warped_corners.min(axis=0)
    max_xy = warped_corners.max(axis=0)
    warped_size = max_xy - min_xy

    # Compute scale to fit warped image back into (w, h)
    scale = min(w / warped_size[0], h / warped_size[1])
    tx, ty = -min_xy * scale  # translation to shift into view
    S = np.array([[scale, 0, tx],
                  [0, scale, ty],
                  [0, 0, 1]], dtype=np.float32)

    # Apply the scaled+translated homography
    H_adj = S @ H
    warped_img0 = cv2.warpPerspective(img0, H_adj, (w, h))
    warped_img1 = cv2.warpPerspective(img1, H_adj, (w, h))

    # Warp keypoints
    def warp_kpts(kpts):
        kpts_h = np.concatenate([kpts, np.ones((len(kpts), 1))], axis=1)
        kpts_w = (H_adj @ kpts_h.T).T
        return kpts_w[:, :2] / kpts_w[:, 2:]

    kpts0_warped = warp_kpts(kpts0)
    kpts1_warped = warp_kpts(kpts1)

    # Convert to cv2.KeyPoint for drawing
    kps0_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts0_warped]
    kps1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts1_warped]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kpts0))]

    matched_vis = cv2.drawMatches(
        warped_img0, kps0_cv, warped_img1, kps1_cv, matches, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(save_path, matched_vis)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("FATAL ERROR: CUDA is required but is not available.")
        sys.exit(1)
    device = torch.device('cuda')
    torch.set_default_device(device)
    seed = 42
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    
    # Load args 
    parser = argparse.ArgumentParser(description="LightGlue matcher.")
    parser.add_argument("--default_lg", action="store_true", help="Use default (pretrained) version of LightGlue")
    parser.add_argument("--match_threshold", "-m", type=float, default=0.5, help="Matching threshold")
    args = parser.parse_args()

    # Setup extractor and matcher
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    # Load data
    feats0 = torch.load('0/features_0.pt')
    feats1 = torch.load('1/features_1.pt')
    features0 = feats0['features']
    features1 = feats1['features']
    if features0 != features1:
        print(f"Features 0 {features0} differs from features 1 {features1}, exiting.")
        sys.exit(1)
    # Need each image to have the same number of keypoints to work with trained LightGlue instance due to masking
    balance_keypoints_by_scores(feats0, feats1)

    image0 = feats0['image']
    image1 = feats1['image']
    w = h = image0.shape[1]
    data = {"keypoints0": feats0['keypoints'].unsqueeze(0), "keypoints1": feats1['keypoints'].unsqueeze(0), "descriptors0": feats0['descriptors'].unsqueeze(0), "descriptors1": feats1['descriptors'].unsqueeze(0)}
    print(f"keypoints0 shape: {data['keypoints0'].shape}, depth shape: {feats0['depth'].shape}")
    print(f"keypoints1 shape: {data['keypoints1'].shape}, depth shape: {feats1['depth'].shape}")
    data["depth_keypoints0"] = get_kp_depth(data["keypoints0"], feats0['depth'].to(device).unsqueeze(0)) 
    data["depth_keypoints1"] = get_kp_depth(data["keypoints1"], feats1['depth'].to(device).unsqueeze(0)) 
    data['overlap_0to1'] = 0.3
    data["view0"] = {"image_size": [w, h]}
    data["view1"] = {"image_size": [w, h]}
    data["view0"]['image'] = image0
    data["view1"]['image'] = image1

    # Setup LightGlue
    matcher = None
    if args.default_lg:
        conf = LightGluePretrained.default_conf
        conf['features'] = features0
        matcher = LightGluePretrained(conf).eval().to(device)
    else:
        if features0 != 'disk':
            print(f"Custom LightGlue training only supports disk features, {features0} selected.")
        conf = LightGlue.default_conf
        #conf['features'] = features0
        conf['input_dim'] = 128
        conf['filter_threshold'] = 0
        conf['weights'] = '/usr/local/home/rsoussan/glue-factory/outputs/training/bartlett/depth_only_combined_string_encoding/checkpoint_best.pth'
        matcher = LightGlue(conf).eval().to(device)

    # Predict
    pred = matcher(data) 

    # Save data
    matches = pred["matches"][0]
    scores = pred["scores"][0]
    kpts0 = data["keypoints0"].squeeze(0)
    kpts1 = data["keypoints1"].squeeze(0)

    # Filter matches
    matches, scores, invalid_matches, invalid_scores = filter_matches_by_score(matches, scores, args.match_threshold)
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    image0 = data['view0']['image'][0].cpu()
    image1 = data['view1']['image'][0].cpu()


    print(f'Matches: {matches.shape[0]}')
    print(f'Invalid Matches: {invalid_matches.shape[0]}')

    rotation = R.from_euler('zyx', [0, 0, 55], degrees=True).as_matrix()                        
    save_matches_in_warped_view(image0, image1, m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy(), feats0['intrinsics'], rotation, save_path="warped_matches.png")

    # Save valid matches
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Matches: {matches.shape[0]}', fs=20)
    viz2d.save_plot("valid_matches.png")

    # Save invalid matches
    m_kpts0, m_kpts1 = kpts0[invalid_matches[..., 0]], kpts1[invalid_matches[..., 1]]
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Invalid Matches: {invalid_matches.shape[0]}', fs=20)
    viz2d.save_plot("invalid_matches.png")
