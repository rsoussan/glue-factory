import torch

from ..utils.tensor import batch_to_device
from .viz2d import cm_RdGn, plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches


def make_match_figures(pred_, data_, n_pairs=2):
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]

    images, kpts, matches, mcolors = [], [], [], []
    gt_images, gt_matches = [], []
    heatmaps = []

    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    gtm0 = pred["gt_matches0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)

        # Predicted matches (original logic unchanged)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = gtm0[i][valid] == m0[i][valid]
        mcolors.append(cm_RdGn(correct).tolist())

        # Ground-truth matches
        gt_valid = gtm0[i] > -1
        gtm0_xy = kp0[i][gt_valid].numpy()
        gtm1_xy = kp1[i][gtm0[i][gt_valid]].numpy()
        gt_images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        gt_matches.append((gtm0_xy, gtm1_xy))

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

    # --- Plot ---
    all_images = [*images, *gt_images]
    fig, axes = plot_image_grid(all_images, return_fig=True, set_lim=True)

    # Predicted matches (keep code identical to original)
    for i in range(n_pairs):
        ax0, ax1 = axes[i]
        if len(heatmaps) > 0:
            plot_heatmaps(heatmaps[i], axes=[ax0, ax1], a=1.0)
        #plot_keypoints(kpts[i], axes=[ax0, ax1], colors="royalblue")
        plot_matches(
            *matches[i],
            color=mcolors[i],
            axes=[ax0, ax1],
            a=0.5,
            lw=0.3,
            ps=0.0,
        )
    print(f"matches: {len(matches)}, gt matches: {len(gt_matches)}")

    # Ground-truth matches (new extra row)
    for i in range(n_pairs):
        ax0, ax1 = axes[i + n_pairs]
        #plot_keypoints(kpts[i], axes=[ax0, ax1], colors="royalblue")
        plot_matches(
            *gt_matches[i],
            color="yellow",
            axes=[ax0, ax1],
            a=0.6,
            lw=0.1,
            ps=0.0,
        )

    return {"matching": fig}

def make_match_figures_orig(pred_, data_, n_pairs=2):
    # print first n pairs in batch
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    gtm0 = pred["gt_matches0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = gtm0[i][valid] == m0[i][valid]

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]

    return {"matching": fig}
