"""
Simply load images from a folder or nested folders (does not have any split),
and apply homographic adaptations to it. Yields an image pair without border
artifacts.
"""

import argparse
import logging
import shutil
import tarfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import zipfile
from omegaconf import OmegaConf
from tqdm import tqdm

from ..geometry.homography import (
    compute_homography,
    sample_homography_corners,
    warp_points,
)
from ..models.cache_loader import CacheLoader, pad_local_features
from ..settings import DATA_PATH
from ..utils.image import read_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .augmentations import IdentityAugmentation, augmentations
from .base_dataset import BaseDataset


"""
ETH3D multi-view benchmark, used for line matching evaluation.
"""

import os

from ..geometry.wrappers import Camera, Pose
from ..utils.image import ImagePreprocessor, load_image
from .utils import scale_intrinsics

logger = logging.getLogger(__name__)

def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = HomographyDataset(conf)
    loader = dataset.get_data_loader("train")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


def sample_homography(img, depth, conf: dict, size: list):
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["depth"] = cv2.warpPerspective(depth, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data

def read_cameras(camera_file, scale_factor=None):
    """Read the camera intrinsics from a file in COLMAP format."""
    with open(camera_file, "r") as f:
        raw_cameras = f.read().rstrip().split("\n")
    raw_cameras = raw_cameras[3:]
    cameras = []
    for c in raw_cameras:
        data = c.split(" ")
        fx, fy, cx, cy = np.array(list(map(float, data[4:])))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        if scale_factor is not None:
            K = scale_intrinsics(K, np.array([scale_factor, scale_factor]))
        cameras.append(Camera.from_calibration_matrix(K).float())
    return cameras


def qvec2rotmat(qvec):
    """Convert from quaternions to rotation matrix."""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


class ETH3DHomographyDataset(BaseDataset):
    default_conf = {
        "data_dir": "ETH3D_undistorted",
        "grayscale": False,
        "downsize_factor": 8,
        "min_covisibility": 500,
        "batch_size": 1,
        "two_view": True,
        "min_overlap": 0.5,
        "max_overlap": 1.0,
        "sort_by_overlap": False,
        "seed": 0,
        # Homography config
        "right_only": False,  # image0 is orig (rescaled), image1 is right
        "reseed": False,
        "homography": {
            "difficulty": 0.8,
            "translation": 1.0,
            "max_angle": 60,
            "n_angles": 10,
            "patch_shape": [640, 480],
            "min_convexity": 0.05,
        },
        "photometric": {
            "name": "dark",
            "p": 0.75,
            # 'difficulty': 1.0,  # currently unused
        },
    }

    def _init(self, conf):
        self.grayscale = conf.grayscale
        self.downsize_factor = conf.downsize_factor

        # Set random seeds
        np.random.seed(conf.seed)
        torch.manual_seed(conf.seed)

        # Auto-download the dataset
        if not (DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading the ETH3D dataset...")
            self.download_eth3d()

        # Form pairs of images from the multiview dataset
        self.img_dir = DATA_PATH / conf.data_dir
        self.data = []
        for folder in self.img_dir.iterdir():
            img_folder = Path(folder, "images", "dslr_images_undistorted")
            depth_folder = Path(folder, "ground_truth_depth/undistorted_depth")
            depth_ext = ".png"
            names = [img.name for img in img_folder.iterdir()]
            names.sort()

            # Read intrinsics and extrinsics data
            cameras = read_cameras(
                str(Path(folder, "dslr_calibration_undistorted", "cameras.txt")),
                1 / self.downsize_factor,
            )
            name_to_cam_idx = {name: {} for name in names}
            with open(
                str(Path(folder, "dslr_calibration_jpg", "images.txt")), "r"
            ) as f:
                raw_data = f.read().rstrip().split("\n")[4::2]
            for raw_line in raw_data:
                line = raw_line.split(" ")
                img_name = os.path.basename(line[-1])
                name_to_cam_idx[img_name]["dist_camera_idx"] = int(line[-2])
            T_world_to_camera = {}
            image_visible_points3D = {}
            with open(
                str(Path(folder, "dslr_calibration_undistorted", "images.txt")), "r"
            ) as f:
                lines = f.readlines()[4:]  # Skip the header
                raw_poses = [line.strip("\n").split(" ") for line in lines[::2]]
                raw_points = [line.strip("\n").split(" ") for line in lines[1::2]]
            for raw_pose, raw_pts in zip(raw_poses, raw_points):
                img_name = os.path.basename(raw_pose[-1])
                # Extract the transform from world to camera
                target_extrinsics = list(map(float, raw_pose[1:8]))
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = qvec2rotmat(target_extrinsics[:4])
                pose[:3, 3] = target_extrinsics[4:]
                T_world_to_camera[img_name] = pose
                name_to_cam_idx[img_name]["undist_camera_idx"] = int(raw_pose[-2])
                # Extract the visible 3D points
                point3D_ids = [id for id in map(int, raw_pts[2::3]) if id != -1]
                image_visible_points3D[img_name] = set(point3D_ids)

            # Extract the covisibility of each image
            num_imgs = len(names)
            n_covisible_points = np.zeros((num_imgs, num_imgs))
            for i in range(num_imgs - 1):
                for j in range(i + 1, num_imgs):
                    visible_points3D1 = image_visible_points3D[names[i]]
                    visible_points3D2 = image_visible_points3D[names[j]]
                    n_covisible_points[i, j] = len(
                        visible_points3D1 & visible_points3D2
                    )

            # Keep only the pairs with enough covisibility
            valid_pairs = np.where(n_covisible_points >= conf.min_covisibility)
            valid_pairs = np.stack(valid_pairs, axis=1)

            self.data += [
                {
                    "view0": {
                        "name": names[i][:-4],
                        "img_path": str(Path(img_folder, names[i])),
                        "depth_path": str(Path(depth_folder, names[i][:-4]))
                        + depth_ext,
                        "camera": cameras[name_to_cam_idx[names[i]]["dist_camera_idx"]],
                        "T_w2cam": Pose.from_4x4mat(T_world_to_camera[names[i]]),
                    },
                    "view1": {
                        "name": names[j][:-4],
                        "img_path": str(Path(img_folder, names[j])),
                        "depth_path": str(Path(depth_folder, names[j][:-4]))
                        + depth_ext,
                        "camera": cameras[name_to_cam_idx[names[j]]["dist_camera_idx"]],
                        "T_w2cam": Pose.from_4x4mat(T_world_to_camera[names[j]]),
                    },
                    "T_world_to_ref": Pose.from_4x4mat(T_world_to_camera[names[i]]),
                    "T_world_to_target": Pose.from_4x4mat(T_world_to_camera[names[j]]),
                    "T_0to1": Pose.from_4x4mat(
                        np.float32(
                            T_world_to_camera[names[j]]
                            @ np.linalg.inv(T_world_to_camera[names[i]])
                        )
                    ),
                    "T_1to0": Pose.from_4x4mat(
                        np.float32(
                            T_world_to_camera[names[i]]
                            @ np.linalg.inv(T_world_to_camera[names[j]])
                        )
                    ),
                    "n_covisible_points": n_covisible_points[i, j],
                }
                for (i, j) in valid_pairs
            ]

        # Homography info 
        aug_conf = conf.photometric
        aug_name = aug_conf.name
        assert (
            aug_name in augmentations.keys()
        ), f'{aug_name} not in {" ".join(augmentations.keys())}'
        self.photo_augment = augmentations[aug_name](aug_conf)
        self.left_augment = (
            IdentityAugmentation() if conf.right_only else self.photo_augment
        )
        self.img_to_tensor = IdentityAugmentation()

        # TODO: make sure to rotate features!
        #if conf.load_features.do:
        #    self.feature_loader = CacheLoader(conf.load_features)

        # Print some info
        print("[Info] Successfully initialized dataset")
        print("\t Name: ETH3D")
        print("----------------------------------------")

    def download_eth3d(self):
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "ETH3D_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://cvg-data.inf.ethz.ch/SOLD2/SOLD2_ETH3D_undistorted/"
        zip_name = "ETH3D_undistorted.zip"
        zip_path = tmp_dir / zip_name
        torch.hub.download_url_to_file(url_base + zip_name, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        shutil.move(tmp_dir / zip_name.split(".")[0], data_dir)

    def get_dataset(self, split):
        return self

#    def _transform_keypoints(self, features, data):
#        """Transform keypoints by a homography, threshold them,
#        and potentially keep only the best ones."""
#        # Warp points
#        features["keypoints"] = warp_points(
#            features["keypoints"], data["H_"], inverse=False
#        )
#        h, w = data["image"].shape[1:3]
#        valid = (
#            (features["keypoints"][:, 0] >= 0)
#            & (features["keypoints"][:, 0] <= w - 1)
#            & (features["keypoints"][:, 1] >= 0)
#            & (features["keypoints"][:, 1] <= h - 1)
#        )
#        features["keypoints"] = features["keypoints"][valid]
#
#        # TODO: enable loading features!!!
#        # Threshold
#        if self.conf.load_features.thresh > 0:
#            valid = features["keypoint_scores"] >= self.conf.load_features.thresh
#            features = {k: v[valid] for k, v in features.items()}
#
#        # Get the top keypoints and pad
#        n = self.conf.load_features.max_num_keypoints
#        if n > -1:
#            inds = np.argsort(-features["keypoint_scores"])
#            features = {k: v[inds[:n]] for k, v in features.items()}
#
#            if self.conf.load_features.force_num_keypoints:
#                features = pad_local_features(
#                    features, self.conf.load_features.max_num_keypoints
#                )
#
#        return features



    def _read_image(self, img_path):
        img = load_image(img_path, grayscale=self.grayscale)
        shape = img.shape[-2:]
        # instead of INTER_AREA this does bilinear interpolation with antialiasing
        img_data = ImagePreprocessor({"resize": max(shape) // self.downsize_factor})(
            img
        )
        return img_data

    def read_depth(self, depth_path):
        if self.downsize_factor != 8:
            raise ValueError(
                "Undistorted depth only available for low res"
                + " images(downsize_factor = 8)."
            )
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth_img = depth_img.astype(np.float32) / 256

        return depth_img

    def _read_view(self, img_tensor, depth, H_conf, ps, left=False):
        img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        data = sample_homography(img, depth, H_conf, ps)
        # visualize these!!
        if left:
            data["image"] = self.left_augment(data["image"], return_tensor=True)
            
        else:
            data["image"] = self.photo_augment(data["image"], return_tensor=True)

        gs = data["image"].new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        if self.conf.grayscale:
            data["image"] = (data["image"] * gs).sum(0, keepdim=True)
        # TODO: enable this!
        #if self.conf.load_features.do:
        #    features = self.feature_loader({k: [v] for k, v in data.items()})
        #    features = self._transform_keypoints(features, data)
        #    data["cache"] = features
        return data

    def __getitem__(self, idx):
        """Returns the data associated to a pair of images (reference, target)
        that are co-visible."""
        data = self.data[idx]
        # Load the images
        view0 = data.pop("view0")
        view1 = data.pop("view1")
        view0 = {**view0, **self._read_image(view0["img_path"])}
        view1 = {**view1, **self._read_image(view1["img_path"])}
        view0["scales"] = np.array([1.0, 1]).astype(np.float32)
        view1["scales"] = np.array([1.0, 1]).astype(np.float32)

        # Load the depths
        view0["depth"] = self.read_depth(view0["depth_path"])
        view1["depth"] = self.read_depth(view1["depth_path"])

        outputs = {
            **data,
            "view0": view0,
            "view1": view1,
            "name": f"{view0['name']}_{view1['name']}",
        }

        # Homography
        ps = self.conf.homography.patch_shape

        left_conf = omegaconf.OmegaConf.to_container(self.conf.homography)
        if self.conf.right_only:
            left_conf["difficulty"] = 0.0

        # Only use view0 image
        img = view0["image"]
        depth = view0["depth"]
        data0 = self._read_view(img, depth, left_conf, ps, left=True)
        data1 = self._read_view(img, depth, self.conf.homography, ps, left=False)

        H = compute_homography(data0["coords"], data1["coords"], [1, 1])

        size = img.shape[:2][::-1]
        outputs["original_image_size"] = np.array(size)
        outputs["H_0to1"] = H.astype(np.float32)
        outputs["idx"] = idx
        outputs["view0"] = view0 
        outputs["view1"] = view1 
        return outputs

    def __len__(self):
        return len(self.data)
