import numpy as np
import sys
import os
import argparse
import torch
import cv2
from torchvision.utils import save_image
from PIL import Image
import math
from typing import Union, Tuple 
from disk import DISK
from sift import SIFT
from superpoint import SuperPoint
from scipy.spatial.transform import Rotation as R
from gluefactory.geometry.depth import sample_normals_from_depth
from dataclasses import dataclass

def mean_normal(normals: np.ndarray) -> np.ndarray:
    # Align normals so they point roughly in the same hemisphere
    reference_vector = [0.0, -1.0, 0.0]
    aligned_normals = np.copy(normals)
    dots = aligned_normals @ reference_vector
    # Flip unaligned normals
    aligned_normals[dots < 0] *= -1  

    # Return mean normal
    mean_normal = np.mean(aligned_normals, axis=0)
    print(f"mean normal: {mean_normal}")
    mean_normal /= np.linalg.norm(mean_normal) + 1e-8
    return mean_normal


def patch_mean_normal(normal_patch: np.ndarray, norm_tolerance: float = 1e-6) -> np.ndarray:
    is_finite_vector = np.all(np.isfinite(normal_patch), axis=-1)
    finite_normals = normal_patch[is_finite_vector] 
    if finite_normals.size == 0:
        print(f"No finite normals in patch.")
        return np.zeros(3, dtype=np.float32)

    # Filter valid (close to unit norm) vectors
    norms = np.linalg.norm(finite_normals, axis=-1)
    is_non_zero_norm = norms > norm_tolerance
    is_unit_norm = np.abs(norms - 1.0) <= norm_tolerance
    # TODO: put back check for unit norm? what percent of norms are finite but not unit norms? (AA)
    is_valid_normal = is_non_zero_norm #& is_unit_norm
    valid_normals = finite_normals[is_valid_normal]

    # Normalize valid norms
    valid_normals = valid_normals/(np.expand_dims(norms[is_valid_normal], -1))
    
    # Get mean norm
    if valid_normals.size > 0:
        print("mean:", np.mean(valid_normals, axis=0), "max:", np.max(valid_normals, axis=0), "min:", np.min(valid_normals, axis=0), "std:", np.std(valid_normals, axis=0))
        return mean_normal(valid_normals)
    else:
        print(f"No valid normals!") 
        return np.zeros(3, dtype=np.float32)

def scale_intrinsics(K: np.ndarray, patch_scale_factor: float) -> np.ndarray:
    K_scaled = K.copy()
    scale = 1.0 / patch_scale_factor
    # Scale focal lengths 
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    # Scale principal points
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return K_scaled.astype(np.float32)

def save_tensor_as_image(tensor: torch.Tensor, filename: str, normalize_range='min_max'):
    """Saves a float tensor as an image."""
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] > 3:
        tensor = tensor[0]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

    tensor = tensor.float()

    if normalize_range == 'min_max':
        cleaned_tensor = torch.nan_to_num(tensor, nan=0.0)
        min_val = cleaned_tensor.min()
        max_val = cleaned_tensor.max()
        if max_val == min_val:
            normalized_tensor = torch.full_like(cleaned_tensor, 0.5)
        else:
            normalized_tensor = (cleaned_tensor - min_val) / (max_val - min_val)
    elif normalize_range == '-1_1': 
        normalized_tensor = (tensor + 1.0) / 2.0
        normalized_tensor = torch.nan_to_num(normalized_tensor, nan=0.0)
    elif normalize_range == '0_1':
        normalized_tensor = torch.nan_to_num(tensor, nan=0.0)
    else:
        raise ValueError("Invalid normalize_range. Use 'min_max', '-1_1', or '0_1'.")
    
    save_image(normalized_tensor.clamp(0.0, 1.0), filename)

def filter_bottom_percent_keypoints(keypoints, bottom_percent):
    responses = np.array([kp.response for kp in keypoints], dtype=float)
    threshold = np.quantile(responses, bottom_percent)
    filtered_keypoints = [kp for kp in keypoints if kp.response > threshold]
    return filtered_keypoints

def filter_keypoints_near_warped_boundary(
    keypoints,
    descriptors, 
    scores,
    H,
    patch_size,
    border_thresh=5,
    warped_patch=None,
    black_thresh=5,
    smooth_border=False,
    convex_fit=True,
    morph_kernel_size=5,
    save_path=None,
    draw_keypoints=False,
):
    """
    Removes keypoints near the visible warped patch boundary,
    tightening the border to exclude black edges and optionally smoothing it.

    Args:
        keypoints (list[cv2.KeyPoint]): Keypoints in warped frame.
        H (np.ndarray): 3x3 homography from original patch → warped image.
        patch_size (int): Original patch size.
        border_thresh (float): Distance (pixels) from border to reject.
        warped_patch (np.ndarray, optional): Warped patch image for analysis/drawing.
        black_thresh (int): Intensity threshold for black pixels.
        smooth_border (bool): Apply morphological smoothing to mask.
        convex_fit (bool): Use convex hull to ensure no valid region is cut off.
        morph_kernel_size (int): Kernel size for mask closing operation.
        save_path (str, optional): Path to save debug visualization.
        draw_keypoints (bool): Whether to draw kept/discarded keypoints.

    Returns:
        filtered_kpts (list[cv2.KeyPoint]): Keypoints kept after filtering.
        tightened_poly (np.ndarray): Polygon (Nx1x2) of the tightened boundary.
    """
    # --- 1. Warp original patch corners ---
    corners = np.array([
        [0, 0, 1],
        [patch_size - 1, 0, 1],
        [patch_size - 1, patch_size - 1, 1],
        [0, patch_size - 1, 1],
    ]).T
    warped_corners = H @ corners
    warped_corners /= warped_corners[2, :]
    warped_corners = warped_corners[:2, :].T.astype(np.float32)
    warped_poly = warped_corners.reshape((-1, 1, 2))

    tightened_poly = warped_poly  # fallback

    if warped_patch is not None:
        # --- 2. Compute valid mask (ignore black areas) ---
        gray = cv2.cvtColor(warped_patch, cv2.COLOR_BGR2GRAY) if warped_patch.ndim == 3 else warped_patch
        mask = (gray > black_thresh).astype(np.uint8) * 255

        # --- 3. Optional smoothing ---
        if smooth_border:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # --- 4. Find the largest valid contour ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # --- 5. Fit convex or raw polygon ---
            if convex_fit:
                tightened_poly = cv2.convexHull(largest_contour)
            else:
                # Optional smoothing: approximate polygon with fewer vertices
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                tightened_poly = cv2.approxPolyDP(largest_contour, epsilon, True)
        else:
            tightened_poly = warped_poly

    # --- 6. Compute keypoint distances from tightened boundary ---
    distances = np.array([
        cv2.pointPolygonTest(tightened_poly, tuple(kp.pt), measureDist=True)
        for kp in keypoints
    ])

    keep_mask = distances > border_thresh
    filtered_kpts = [kp for kp, keep in zip(keypoints, keep_mask) if keep]
    filtered_descriptors = descriptors[keep_mask]
    filtered_scores = scores[keep_mask]
    discarded_kpts = [kp for kp, keep in zip(keypoints, keep_mask) if not keep]

    # --- 7. Visualization ---
    if warped_patch is not None and save_path is not None:
        vis = warped_patch.copy()
        if vis.dtype != np.uint8:
            vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Draw boundaries
        cv2.polylines(vis, [np.int32(warped_poly)], True, (0, 0, 255), 2)  # red = original warped boundary
        cv2.polylines(vis, [np.int32(tightened_poly)], True, (0, 255, 0), 2)  # green = tightened visible boundary

        if draw_keypoints:
            vis = cv2.drawKeypoints(
                vis, filtered_kpts, None, (0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            vis = cv2.drawKeypoints(
                vis, discarded_kpts, None, (0, 0, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

        cv2.imwrite(save_path, vis)

    return filtered_kpts, filtered_descriptors, filtered_scores

# TODO: clean all this up! unify rgsw data output to match megadepth data output! remove conditional processing!!
def load_data(data_path, prefix='0'):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"PyTorch data file not found at: {data_path}")

    data = torch.load(data_path)
    depth_tensor_original = {}
    depth_name = 'depth' + prefix
    view_name = 'view' + prefix
    if depth_name in data:
        depth_tensor_original = data[depth_name]
    else:
        depth_tensor_original = data[view_name]['depth']

    try:
        rgb_tensor_original = data[view_name]['image']
        camera_object = data[view_name]['camera']
    except KeyError as e:
        raise KeyError(f"Missing expected key in .pt file: {e}.")

    if rgb_tensor_original.shape[0] == 1:
        rgb_tensor_original = rgb_tensor_original.squeeze(0) 
    try:
        K_torch = camera_object.calibration_matrix()
        K = K_torch.cpu().numpy().astype(np.float32)
       
        if K.shape == (1, 3, 3):
            K = K.squeeze(0) 
        if K.shape != (3, 3):
            raise ValueError(f"Calibration matrix returned shape {K.shape}, expected (3, 3).")
            
    except Exception as e:
         raise RuntimeError(f"Error extracting calibration matrix or method not found: {e}")

    FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"Loaded Intrinsics: FX={FX:.2f}, FY={FY:.2f}, CX={CX:.2f}, CY={CY:.2f}")

    rgb_tensor_processed = rgb_tensor_original.float().cpu()
    # Normalize RGB tensor to C, H, W format
    if rgb_tensor_processed.ndim == 3 and rgb_tensor_processed.shape[2] == 3:
        rgb_tensor_processed = rgb_tensor_processed.permute(2, 0, 1)

    if rgb_tensor_processed.max() <= 1.0:
        rgb_img_np = (rgb_tensor_processed * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    else:
        rgb_img_np = rgb_tensor_processed.to(torch.uint8).numpy()

    if rgb_img_np.shape[0] == 3: 
        rgb_img_np = np.transpose(rgb_img_np, (1, 2, 0)) # H, W, C
    if rgb_img_np.shape[2] == 3:
         # TODO: why bgr???
         rgb_img_np = cv2.cvtColor(rgb_img_np, cv2.COLOR_RGB2BGR)

    depth_tensor_processed = depth_tensor_original.float().cpu()
    if depth_tensor_processed.ndim == 3 and depth_tensor_processed.shape[0] == 1:
        depth_tensor_processed = depth_tensor_processed.squeeze(0) # H, W

   
    depth_img_np = depth_tensor_processed.numpy()

#     # TODO: pass height and width as args!!! is this actually needed???
#    target_size_tuple = (IMAGE_HEIGHT, IMAGE_WIDTH)
#    # Resize data if it doesn't match the target resolution
#    if depth_img_np.shape[:2] != target_size_tuple:
#        print(f"Resizing input data from {depth_img_np.shape[:2]} to {target_size_tuple}")
#        rgb_img_np = cv2.resize(rgb_img_np, (IMAGE_WIDTH, IMAGE_HEIGHT))
#        rgb_tensor_original = torch.from_numpy(rgb_img_np).permute(2,0,1).float() / 255.0
#
#        depth_img_np = cv2.resize(depth_tensor_processed.numpy(), (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
#        depth_tensor_processed = torch.from_numpy(depth_img_np).float()
#    else:
#         rgb_tensor_original = rgb_tensor_processed # Already in C, H, W format 
#
    return rgb_img_np, depth_img_np, rgb_tensor_original, depth_tensor_processed, K

def save_data(keypoints, descriptors, filename):
    kp_array = torch.tensor([kp.pt for kp in keypoints], dtype=torch.float32)
    desc_tensor = torch.tensor(descriptors, dtype=torch.float32)
    scores_tensor = torch.tensor([kp.response for kp in keypoints], dtype=torch.float32)
    torch.save({
        "keypoints": kp_array,
        "descriptors": desc_tensor,
        "scores": scores_tensor
    }, filename)

def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that rotates vec1 to vec2 (Rodrigues' formula)."""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-6:
        return np.identity(3, dtype=np.float32) 

    K_skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float32)
    
    R = np.identity(3, dtype=np.float32) + K_skew + K_skew @ K_skew * ((1 - c) / (s ** 2))
    return R

def clamp_rotation_rpy(R_in: np.ndarray, max_rpy: np.ndarray) -> np.ndarray:                                                                                                                                                                                  
    """                                                                                                                                                                                                                         
    Clamp the roll, pitch, yaw of a rotation matrix (Z-Y-X convention) using degree thresholds.                                                                                                                                               
                                                                                                                                                                     
    Args:                                                                                                                                                                                                                                                     
        R_in (np.ndarray): Input 3x3 rotation matrix.                                                                                                                                                                                                                           
        max_rpy (np.ndarray): Max thresholds [max_roll, max_pitch, max_yaw] in degrees.                                                                                                                                                                                         
                                                                                                                                                                     
    Returns:                                                                          
        np.ndarray: 3x3 rotation matrix with thresholded rpy values.                                                                                                 
    """                                                                           
    # --- 1. Convert rotation matrix to rpy (Z-Y-X convention) ---                
    r = R.from_matrix(R_in)                                                       
    # scipy returns [z, y, x] for 'zyx'; reverse to get roll, pitch, yaw          
    roll, pitch, yaw = r.as_euler('zyx', degrees=True)[::-1]                                                                                                         
    rpy = np.array([roll, pitch, yaw])                                            
                                                                                      
    # --- 2. Clamp each angle magnitude to threshold, preserving sign ---             
    clamped_rpy = np.sign(rpy) * np.minimum(np.abs(rpy), max_rpy)                         
                                                                                               
    # --- 3. Convert back to rotation matrix ---                                               
    R_out = R.from_euler('zyx', clamped_rpy[::-1], degrees=True).as_matrix()                        
    print(f"original rpy: {rpy} \n clamped_rpy: {clamped_rpy}")                                             
                                                                                                          
    return R_out 

def rotation_matrix_from_rpy(rpy):
    # Reverse rpy to ypr due to ZYX convention
    return R.from_euler('zyx', rpy[::-1], degrees=True).as_matrix()                        

def get_normals(depth_image: np.ndarray, K: np.ndarray, device) -> np.ndarray:
    depth_tensor = torch.from_numpy(depth_image).float().to(device)
    H, W = depth_tensor.shape

    # Prepare sample locations tensor [1, H*W, 2] 
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    sample_locations = torch.stack([x, y], dim=-1).float().reshape(1, H * W, 2) # [1, H*W, 2]

    # Add batch dimension to depth tensor and K
    depth_tensor_batch = depth_tensor.unsqueeze(0) # [1, H, W]
    K_batch = torch.from_numpy(K).float().to(device).unsqueeze(0) # [1, 3, 3]

    # Sample normals 
    normals, _ = sample_normals_from_depth(sample_locations, depth_tensor_batch, K_batch)
    
    # Convert normals to np 
    # Remove batch dim
    normals = normals.squeeze(0).cpu() # [H*W, 3]
    normals_np = normals.reshape(H, W, 3).numpy().astype(np.float32)
    return normals_np


def calculate_critical_roll_angle(K: np.ndarray) -> float:
    """
    Calculates the critical roll angle (geometric horizon) in degrees.

    This angle represents the pure roll (rotation around the X-axis) at which 
    the patch projection denominator (perspective scaling factor w') becomes 
    zero, leading to an infinite projection and image flip.

    Mathematical Proof:
    The instability occurs when the homography term H_geo(2, 2) is zero.
    Since H_geo(2, 2) = (R * K_inv)(2, 2), and K_inv(1, 2) = -cv/fv:
    
    H_geo(2, 2) = R(2, 1) * K_inv(1, 2) + R(2, 2) * K_inv(2, 2) = 0
    
    For a pure roll angle theta (R_roll(theta)):
    R(2, 1) = sin(theta)
    R(2, 2) = cos(theta)
    
    Substituting:
    sin(theta) * (-cv/fv) + cos(theta) * 1 = 0
    sin(theta) * (cv/fv) = cos(theta)
    tan(theta) = fv / cv
    
    The critical angle theta is arctan(fv / cv).

    Args:
        K (np.ndarray): The 3x3 camera intrinsic matrix.

    Returns:
        float: The critical roll angle in degrees.
    """
    # K is structured as:
    # [[fu, 0, cu],
    #  [0, fv, cv],
    #  [0, 0, 1]]
    
    fv = K[1, 1]  # Vertical focal length
    cv = K[1, 2]  # Vertical principal point offset (y-coordinate)
    
    if np.isclose(cv, 0):
        # If the principal point is perfectly centered, the critical angle is 90 degrees.
        print("WARNING: Principal point offset (cv) is near zero. Critical angle is 90 degrees.")
        return 90.0

    # Calculate the ratio tan(theta) = fv / cv
    critical_tan_ratio = fv / cv
    
    # Calculate the angle
    critical_roll_radians = np.arctan(critical_tan_ratio)
    critical_roll_degrees = np.degrees(critical_roll_radians)
    
    return critical_roll_degrees


@dataclass
class PatchData:
    rgb_patch: np.ndarray           # (H, W, 3)
    warped_patch: np.ndarray        # (H, W, 3)
    x: int                          # top-left x of patch in global image
    y: int                          # top-left y of patch in global image
    homography: np.ndarray          # (3, 3) warp homography matrix
    mean_normal: np.ndarray         # (1, 3) mean normal
    warped_keypoints: list 


class PatchWarper:
    def __init__(self, K: np.ndarray, patch_size: int, patch_size_factor: int, max_warped_dim_multiplier: int = 3, border_mode: int = cv2.BORDER_CONSTANT, pitch_degrees: Union[float, None] = None):
        self.original_K = K
        self.patch_size = patch_size
        self.patch_size_factor = patch_size_factor
        self.max_warped_dim = int(self.patch_size * max_warped_dim_multiplier)
        self.border_mode = border_mode
        self.target_normal = np.array([0, 0, -1], dtype=np.float32)

        self.patch_corners_homogeneous = np.array([
            [0, 0, 1],                # Top-Left
            [patch_size, 0, 1],       # Top-Right
            [0, patch_size, 1],       # Bottom-Left
            [patch_size, patch_size, 1] # Bottom-Right
        ], dtype=np.float32).T # Shape (3, 4)

        self.R_fixed = None 
        if pitch_degrees is not None:
            # In the camera frame, pitching is around the x axis, so technically its roll
            self.R_fixed = rotation_matrix_from_rpy([pitch_degrees, 0, 0])

    def _calculate_warped_dims_and_shift(self, homography: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
        warped_corners_homogeneous = homography @ self.patch_corners_homogeneous 
        z_coords = warped_corners_homogeneous[2, :]
        # Guard against zero or near-zero division (singularity)
        z_coords = np.where(np.abs(z_coords) < 1e-6, np.inf, z_coords)
        corners_warped_cartesian = warped_corners_homogeneous[:2, :] / z_coords 

        # Check for invalid warping
        if np.any(np.isinf(corners_warped_cartesian)) or np.any(np.isnan(corners_warped_cartesian)):
            print("Warning: Warping resulted in NaN/Inf coordinates. Returning default patch size.")
            warped_width = self.patch_size 
            warped_height = self.patch_size
            min_x, min_y = 0.0, 0.0
            return (warped_width, warped_height), np.array([min_x, min_y], dtype=np.float32)

        # Find the bounding box of the warped corners
        min_x = corners_warped_cartesian[0, :].min()
        max_x = corners_warped_cartesian[0, :].max()
        min_y = corners_warped_cartesian[1, :].min()
        max_y = corners_warped_cartesian[1, :].max()

        # Calculate the required output dimensions (raw)
        raw_width = int(np.ceil(max_x - min_x))
        raw_height = int(np.ceil(max_y - min_y))
        
        # Limit patch size to prevent extremely large patches
        max_dim = self.max_warped_dim
        scale_w = min(1.0, max_dim / raw_width) if raw_width > 0 else 1.0
        scale_h = min(1.0, max_dim / raw_height) if raw_height > 0 else 1.0
        # Use the most restrictive (smallest) scale factor to ensure both dimensions fit
        scale_factor = min(scale_w, scale_h)

        # Calculate final output dimensions 
        # Apply the scaling to the raw extent and then convert to the final output integer size.
        scaled_width = raw_width * scale_factor
        scaled_height = raw_height * scale_factor
        scaled_width = max(1, int(np.ceil(scaled_width)))
        scaled_height = max(1, int(np.ceil(scaled_height)))
        
        # Construct the combined Shift and Scale Matrix (H_shift_scale)
        # This matrix performs:
        # 1. Translation: -min_x, -min_y (to move the corner to 0,0)
        # 2. Scaling: * scale_factor (to shrink the patch to fit the max_dim)
        H_shift_scale = np.array([
            [scale_factor, 0, -min_x * scale_factor], 
            [0, scale_factor, -min_y * scale_factor], 
            [0, 0, 1]
        ], dtype=np.float32)

        return (scaled_width, scaled_height), H_shift_scale

    def get_warping_params(self, x: float, y: float, R) -> Union[Tuple[np.ndarray, Tuple[int, int], np.ndarray], Tuple[None, None, None]]:
        """
        Calculates H_geo, warped_dims, and H_final for the current patch.
        Returns H_geo, (W, H), H_final
        """
      
        # Adjust principal point based on patch location  
        K = self.original_K.copy()
        K[0, 2] -= x # cx
        K[1, 2] -= y # cy
        #K = scale_intrinsics(K, self.patch_size_factor) 
 
        K_inv = np.linalg.inv(K)
        H_geo = K @ R @ K_inv
        
        # This call now performs the size capping and determines the necessary scale/shift matrix
        warped_dims, H_shift_scale = self._calculate_warped_dims_and_shift(H_geo)
        H_final = H_shift_scale @ H_geo
        return warped_dims, H_final

    def get_rotation(self, mean_normal):
        if self.R_fixed is not None:
            return self.R_fixed
        valid = np.any(~np.isclose(mean_normal, 0, atol=1e-6)) and np.isfinite(mean_normal).all()
        if not valid:
            print("Invalid mean normal, not applying rotation.")
            return np.eye(3)
        rotation = rotation_matrix_from_vectors(mean_normal, self.target_normal)
        return clamp_rotation_rpy(rotation, [55, 55, 55])    

    def warp_patch(self, rgb_patch, x, y, mean_normal):
        R = self.get_rotation(mean_normal)
        warped_dims, H_final = self.get_warping_params(x, y, R)
        if H_final is None:
            print(f"H final is none!")
            return None, None 

        warped_patch = cv2.warpPerspective(
            rgb_patch,
            H_final, 
            warped_dims,
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode
        )
        return warped_patch, H_final

def detect_features_and_unwarp(
    feature_detector,
    image,
    H,
    x,
    y,
    patch_size, 
    unwarped_keypoints_all,
    descriptors_all, 
    warped_keypoints,
    filter_boundary_keypoints = True
):
    device="cuda" if torch.cuda.is_available() else "cpu"
    # Convert image to Torch format 
    img_torch = torch.from_numpy(image).float() / 255.0
    if img_torch.ndim == 2:
        img_torch = img_torch.unsqueeze(0)  # (1,H,W)
    elif img_torch.ndim == 3:
        img_torch = img_torch.permute(2, 0, 1)  # (C,H,W)
    img_torch = img_torch.to(device)

    # Extract features
    with torch.no_grad():
        pred = feature_detector.extract(img_torch)

    keypoints = pred["keypoints"][0].cpu().numpy()      # (N, 2)
    descriptors = pred["descriptors"][0].cpu().numpy()  # (N, D)
    scores = pred["keypoint_scores"][0].cpu().numpy()   # (N,)

    # No detections, return empty
    if keypoints.shape[0] == 0:
        unwarped_pred = {
            "keypoints": torch.zeros((0, 2)),
            "keypoint_scores": torch.zeros((0,)),
            "descriptors": torch.zeros((0, descriptors.shape[1]) if descriptors.size else (0, 256)),
        }
        return [], np.zeros((0, descriptors.shape[1])), unwarped_pred

    # Save warped keypoints
    for i, (keypoint_x, keypoint_y) in enumerate(keypoints):
        keypoint = cv2.KeyPoint(
            x=float(keypoint_x),
            y=float(keypoint_y),
            size=1.0,
            angle=-1,
            response=float(scores[i]),
            octave=0
        )
        warped_keypoints.append(keypoint)

    if filter_boundary_keypoints:
        warped_keypoints, descriptors, scores = filter_keypoints_near_warped_boundary(warped_keypoints, descriptors, scores, H, patch_size, border_thresh=5, warped_patch=image, save_path=f"patch_keypoints_{x}_{y}.png", draw_keypoints=True)
        keypoints = np.array([kp.pt for kp in warped_keypoints], dtype=np.float32)

    # Add new descriptors
    descriptors_all.append(descriptors)

    # Unwarp keypoints 
    H_inv = np.linalg.inv(H)
    keypoints_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
    unwarped_keypoints_homogeneous = (H_inv @ keypoints_homogeneous.T).T
    # Normalize
    z = unwarped_keypoints_homogeneous[:, 2:3]
    z[np.abs(z) < 1e-6] = 1.0
    unwarped_keypoints = unwarped_keypoints_homogeneous[:, :2] / z
    # Add global offset
    global_shift = np.array([x, y])
    unwarped_shifted_keypoints = unwarped_keypoints + global_shift

    for i, (keypoint_x, keypoint_y) in enumerate(unwarped_shifted_keypoints):
        keypoint = cv2.KeyPoint(
            x=float(keypoint_x),
            y=float(keypoint_y),
            size=1.0,
            angle=-1,
            response=float(scores[i]),
            octave=0
        )
        unwarped_keypoints_all.append(keypoint)

    # Torch predictions (TODO: are these used by anyone??)
    unwarped_pred = {
        "keypoints": torch.from_numpy(unwarped_shifted_keypoints).float(),
        "keypoint_scores": torch.from_numpy(scores).float(),
        "descriptors": torch.from_numpy(descriptors).float(),
    }

    return unwarped_pred

def detect_features_from_patches(rgb_img, normals, patch_warper):
    """
    Extracts features after warping patches based on mean normal or fixed pitch rotation.
    """
    H, W = rgb_img.shape[:2]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_detector = DISK(max_num_keypoints=int(2048/(PATCH_SIZE_FACTOR*PATCH_SIZE_FACTOR))).eval().to(device) 
    #feature_detector = SuperPoint(max_num_keypoints=int(2048/(PATCH_SIZE_FACTOR*PATCH_SIZE_FACTOR))).eval().to(device) 
    keypoints_all = []
    descriptors_all = []
    patches = []

    for y in range(0, H - patch_warper.patch_size + 1, patch_warper.patch_size):
        for x in range(0, W - patch_warper.patch_size + 1, patch_warper.patch_size):
            y_end = y + patch_warper.patch_size
            x_end = x + patch_warper.patch_size
            rgb_patch = rgb_img[y:y_end, x:x_end]
            normal_patch = normals[y:y_end, x:x_end]
            mean_normal = patch_mean_normal(normal_patch) 
            warped_patch, H = patch_warper.warp_patch(rgb_patch, x, y, mean_normal)
            # TODO: don't convert to grayscale? -> make this optional!! (BB) 
            #gray_warped_patch = cv2.cvtColor(warped_patch, cv2.COLOR_BGR2GRAY)
            warped_keypoints = []
            detect_features_and_unwarp(
                feature_detector,
                warped_patch,
                H,
                x,
                y,
                patch_warper.patch_size, 
                keypoints_all,
                descriptors_all, 
                warped_keypoints
            )
            patches.append(PatchData(rgb_patch, warped_patch, x, y, H, mean_normal, warped_keypoints))
    descriptors_all = np.vstack(descriptors_all)
    return keypoints_all, descriptors_all, patches

def create_keypoint_image(keypoints, rgb_image, patch_size):
    h, w = rgb_image.shape[:2]
    keypoint_image = rgb_image.copy()
    keypoint_image = cv2.drawKeypoints(image=keypoint_image, keypoints=keypoints, outImage=None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Add patch rectangles
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            cv2.rectangle(keypoint_image, (x, y), (x + patch_size, y + patch_size), (255, 0, 0), 1)
    return keypoint_image
 

def create_warped_patches_mosaic_image(patches, h, w, draw_keypoints = False, filter_keypoints_bottom_percent=None):
    mosaic_image = np.zeros((h, w, 3), dtype=np.uint8)
    for patch in patches:
        x = patch.x
        y = patch.y
        # Assumes square patches
        patch_size = patch.rgb_patch.shape[0]
        y_end = y + patch_size
        x_end = x + patch_size
       
        warped_patch = patch.warped_patch.copy() 
        if draw_keypoints:
            keypoints = patch.warped_keypoints
            if filter_keypoints_bottom_percent is not None:
                keypoints = filter_bottom_percent_keypoints(keypoints, filter_keypoints_bottom_percent)  
            warped_patch = cv2.drawKeypoints(image=warped_patch, keypoints=keypoints, outImage=None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        resized_warped_patch = cv2.resize(
            warped_patch, 
            (patch_size, patch_size), 
            interpolation=cv2.INTER_LINEAR
        )
        mosaic_image[y:y_end, x:x_end] = resized_warped_patch
    return mosaic_image


def create_normals_arrow_image(patches, rgb_img: np.ndarray) -> np.ndarray:
    """Draws color-coded normal arrows over the rgb image."""
    H, W = rgb_img.shape[:2]
    # Assumes square patches
    patch_size = patches[0].rgb_patch.shape[0]
    normals_arrow_img = rgb_img.copy() 
    arrow_thickness = 1          
    arrow_thickness_outline = arrow_thickness + 2 
    arrow_visual_scalar = patch_size * 1.0 
    arrow_tip_length = 0.15 

    for patch in patches:
        x = patch.x
        y = patch.y
        y_end = y + patch_size
        x_end = x + patch_size
        center_x = x + patch_size // 2
        center_y = y + patch_size // 2
       
        nx, ny, nz = patch.mean_normal 
        # Convert normal vector to BGR color space for visualization
        arrow_color_r = int((nx + 1) / 2 * 255)
        arrow_color_g = int((ny + 1) / 2 * 255)
        arrow_color_b = int((nz + 1) / 2 * 255)
        arrow_color_main = (arrow_color_b, arrow_color_g, arrow_color_r) 

        end_x = int(center_x + nx * arrow_visual_scalar)
        end_y = int(center_y + ny * arrow_visual_scalar)

        cv2.arrowedLine(img=normals_arrow_img, pt1=(center_x, center_y), pt2=(end_x, end_y), color=(0, 0, 0), thickness=arrow_thickness_outline, tipLength=arrow_tip_length)
        cv2.arrowedLine(img=normals_arrow_img, pt1=(center_x, center_y), pt2=(end_x, end_y), color=arrow_color_main, thickness=arrow_thickness, tipLength=arrow_tip_length)
            
    return normals_arrow_img


def create_original_vs_warped_patches_mosaic_image(patches, patch_size, max_warped_dim, h, w):
    """
    Creates a visualization where each original patch is shown next to its geometrically warped version.
    Includes warped patch dimensions in the visualization and draws the mean normal on the original patch.
    """
    padding_x, padding_y = 2, 2
    max_row_height = max(patch_size, max_warped_dim) 
    # Display original and warped patch side by side
    total_patch_width_display = patch_size + max_warped_dim + 2 * padding_x 
    # Add extra space for text below
    total_patch_height_display = max_row_height + 2 * padding_y + 20 

    num_patches_x = w // patch_size
    num_patches_y = h // patch_size

    output_width = num_patches_x * total_patch_width_display + padding_x
    output_height = num_patches_y * total_patch_height_display + padding_y

    comparison_mosaic = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    comparison_mosaic.fill(50) 
    
    arrow_visual_scalar = patch_size * 0.4 
    arrow_thickness = 1 
    arrow_thickness_outline = 3
    arrow_tip_length = 0.2
    
    for n, patch in enumerate(patches):
        x = patch.x
        y = patch.y
        y_end = y + patch_size
        x_end = x + patch_size

        rgb_patch_original = patch.rgb_patch.copy()
        mean_normal = patch.mean_normal

        # Draw Mean Normal Arrow on Original Patch
        # TODO: add function for this???
        if mean_normal is not None and np.linalg.norm(mean_normal) > 1e-6:
            nx, ny, nz = mean_normal
            arrow_color_r = int((nx + 1) / 2 * 255)
            arrow_color_g = int((ny + 1) / 2 * 255)
            arrow_color_b = int((nz + 1) / 2 * 255)
            arrow_color_main = (arrow_color_b, arrow_color_g, arrow_color_r) 
            
            center_x = patch_size // 2
            center_y = patch_size // 2
            
            end_x = int(center_x + nx * arrow_visual_scalar)
            end_y = int(center_y + ny * arrow_visual_scalar)

            pt1 = (center_x, center_y)
            pt2 = (end_x, end_y)
            
            # Draw Outline (Black)
            cv2.arrowedLine(img=rgb_patch_original, pt1=pt1, pt2=pt2, color=(0, 0, 0), 
                            thickness=arrow_thickness_outline, tipLength=arrow_tip_length)
            # Draw Main color
            cv2.arrowedLine(img=rgb_patch_original, pt1=pt1, pt2=pt2, color=arrow_color_main, 
                            thickness=arrow_thickness, tipLength=arrow_tip_length)
        
        # 4. Place original patch (with normal arrow)
        w = int(math.sqrt(len(patches)))
        i = n // w
        j = n % w 
        display_start_x_orig = j * total_patch_width_display + padding_x
        display_start_y_orig = i * total_patch_height_display + padding_y
        comparison_mosaic[display_start_y_orig : display_start_y_orig + patch_size, 
                          display_start_x_orig : display_start_x_orig + patch_size] = rgb_patch_original

        # 5. Place warped patch (display dimensions are the capped warped_dims)
        display_start_x_warped = display_start_x_orig + patch_size + padding_x
        display_start_y_warped = display_start_y_orig
        
        warped_h, warped_w, _ = patch.warped_patch.shape
        if patch.warped_patch.size > 0:
            comparison_mosaic[display_start_y_warped : display_start_y_warped + warped_h, 
                              display_start_x_warped : display_start_x_warped + warped_w] = patch.warped_patch
        
        # 6. Add dimension text (now safe from UnboundLocalError)
        dims_text = f"W:{warped_w} H:{warped_h}"
        text_pos = (display_start_x_warped, display_start_y_warped + max_row_height + 15)
        cv2.putText(comparison_mosaic, dims_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 7. Draw borders
        cv2.rectangle(comparison_mosaic, (display_start_x_orig, display_start_y_orig), (display_start_x_orig + patch_size, display_start_y_orig + patch_size), (0, 255, 0), 1) 
        if patch.warped_patch.size > 0:
            cv2.rectangle(comparison_mosaic, (display_start_x_warped, display_start_y_warped), (display_start_x_warped + warped_w, display_start_y_warped + warped_h), (255, 0, 0), 1) 

    return comparison_mosaic

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("FATAL ERROR: CUDA is required but is not available.")
        sys.exit(1)
    device = torch.device('cuda')
    torch.set_default_device(device)
     
    parser = argparse.ArgumentParser(description="Patch-based feature extractor.")
    parser.add_argument("data_path", help="Path to the PyTorch .pt data file.")
    parser.add_argument("--fixed_pitch", type=float, default=None, help="If provided, use a fixed pitch rotation (degrees) instead of rotations derived from depth normals.")
    parser.add_argument("--data_prefix", type=str, default='0', help="Data to load, can be 0 or 1.")
    args = parser.parse_args()

    if args.fixed_pitch is not None:
        print(f"Using fixed pitch: {args.fixed_pitch}")

    # Load data 
    try:
        rgb_image, depth_image, rgb_tensor, depth_tensor, K = load_data(args.data_path, prefix=args.data_prefix)
    except Exception as e:
        print(f"FATAL ERROR: Could not load data from {args.pt_path}. Reason: {e}")
        sys.exit(1)
   
    h, w = rgb_image.shape[:2] 
    # Constants
    PATCH_SIZE_FACTOR = 4 
    # Assumes square image. TODO: account for non square images...
    PATCH_SIZE = w // PATCH_SIZE_FACTOR # This should be 64
    MAX_WARPED_DIM_MULTIPLIER = 3 
    BORDER_MODE = cv2.BORDER_CONSTANT 
        
    normals = get_normals(depth_image, K, device)
    patch_warper = PatchWarper(K, PATCH_SIZE, PATCH_SIZE_FACTOR, MAX_WARPED_DIM_MULTIPLIER, BORDER_MODE, args.fixed_pitch)
    keypoints, descriptors, patches = detect_features_from_patches(rgb_image, normals, patch_warper)
    save_data(keypoints, descriptors, f"features_{args.data_prefix}.pt")
    print("\n--- Feature Extraction Summary ---")
    print(f"Total Keypoints Detected: {len(keypoints)}")
    if descriptors is not None:
        print(f"Total Descriptors Shape: {descriptors.shape}")

# TODO: move visualization code to new file!!! 
    # Save raw image and depth image
    print("\n--- Saving Images for Visualization ---")
    save_image(rgb_tensor.permute(2,0,1) if rgb_tensor.ndim == 3 and rgb_tensor.shape[2] == 3 else rgb_tensor, 'rgb.png')
    save_tensor_as_image(depth_tensor, 'depths.png', normalize_range='min_max')
   
    # Save normals image 
    normals_tensor = torch.from_numpy(normals).permute(2, 0, 1).float()
    save_tensor_as_image(normals_tensor, 'normals.png', normalize_range='-1_1')
    normals_arrow_image = create_normals_arrow_image(patches, rgb_image)
    cv2.imwrite('normals_arrow.png', normals_arrow_image)

    # Save keypoint image
    keypoint_image = create_keypoint_image(keypoints, rgb_image, PATCH_SIZE)
    cv2.imwrite("keypoints.png", keypoint_image)
    
    # Save warped patches mosaic
    warped_patches_mosaic_image = create_warped_patches_mosaic_image(patches, h, w)
    cv2.imwrite("warped_patches_mosaic.png", warped_patches_mosaic_image)

    # Save warped patches mosaic with keypoints
    warped_patches_mosaic_with_keypoints_image = create_warped_patches_mosaic_image(patches, h, w, draw_keypoints=True)
    cv2.imwrite("warped_patches_mosaic_with_keypoints.png", warped_patches_mosaic_with_keypoints_image)

    # Save warped patches mosaic with filtered keypoints
    warped_patches_mosaic_with_filtered_keypoints_image = create_warped_patches_mosaic_image(patches, h, w, draw_keypoints=True, filter_keypoints_bottom_percent=0.8)
    cv2.imwrite("warped_patches_mosaic_with_filtered_keypoints.png", warped_patches_mosaic_with_filtered_keypoints_image)

    # Save original vs warped patches mosaic
    original_vs_warped_patches_mosaic_image = create_original_vs_warped_patches_mosaic_image(patches, PATCH_SIZE, patch_warper.max_warped_dim, h, w)
    cv2.imwrite('original_vs_warped_patches_mosaic.png', original_vs_warped_patches_mosaic_image)

