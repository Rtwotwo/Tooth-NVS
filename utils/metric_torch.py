# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import math
import open3d as o3d
from tqdm import tqdm
from typing import Tuple

def write_evaluation_results(target_dir: str, images_count: int, auc_results: dict, pcd_results: dict, inference_time_s: float, peak_mem_mb: float) -> None:
    result_file = os.path.join(target_dir, "eval_results.txt")
    with open(result_file, "w") as f:
        f.write(f"Image Count: {images_count},\n")
        f.write(f"Relative Rotation Error (degrees): {auc_results['rel_rangle_deg']},\n")
        f.write(f"Relative Translation Error (degrees): {auc_results['rel_tangle_deg']},\n")
        f.write(f"Racc_5: {auc_results['Racc_5']},\n")
        f.write(f"Racc_15: {auc_results['Racc_15']},\n")
        f.write(f"Tacc_5: {auc_results['Tacc_5']},\n")
        f.write(f"Tacc_15: {auc_results['Tacc_15']},\n")
        f.write(f"AUC at 30 degrees: {auc_results['Auc_30']},\n")
        f.write(f"Accuracy Mean: {pcd_results['accuracy_mean']},\n")
        f.write(f"Completeness Mean: {pcd_results['completeness_mean']},\n")
        f.write(f"Chamfer Distance: {pcd_results['chamfer_distance']},\n")
        f.write(f"Inference Time: {inference_time_s},\n")
        f.write(f"Peak Memory Usage (MB): {peak_mem_mb},\n")


@torch.inference_mode()
def evaluate_pcd(
        pcd_gt, 
        points_3d, 
        depth_conf,
        images,
        images_gt_updated, 
        original_coords, 
        vggt_fixed_resolution=518,
        conf_thresh=1.0):
    
    pcd_xyz_gt_list = []
    pcd_rgb_gt_list = []
    pcd_xyz_sampled_list = []
    pcd_rgb_sampled_list = []

    for k, idx in tqdm(enumerate(list(images_gt_updated.keys())), desc="Evaluating Points..."):

        point3D_ids_gt = images_gt_updated[idx].point3D_ids
        mask_gt = (point3D_ids_gt >= 0) & (images_gt_updated[idx].xys[:, 0] >= 0) & (images_gt_updated[idx].xys[:, 1] >= 0) & \
                    (images_gt_updated[idx].xys[:, 0] < original_coords[k, -2].item()) & (images_gt_updated[idx].xys[:, 1] < original_coords[k, -1].item())

        xys_gt = images_gt_updated[idx].xys[mask_gt]
        pcd_rgb_gt = np.stack([pcd_gt[id].rgb for id in point3D_ids_gt[mask_gt]], axis=0)
        pcd_xyz_gt = np.stack([pcd_gt[id].xyz for id in point3D_ids_gt[mask_gt]], axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_xyz_gt)
        distances = np.array(pcd.compute_nearest_neighbor_distance())
        avg_dist = np.mean(distances)
        std_dev_dist = np.std(distances)

        mask_distance = distances < avg_dist + 2 * std_dev_dist
        xys_gt = xys_gt[mask_distance]
        pcd_xyz_gt = pcd_xyz_gt[mask_distance]
        pcd_rgb_gt = pcd_rgb_gt[mask_distance]

        # transform xys_gt to the coordinate on points_3d, which is (N, H, W, 3)
        xys_gt_scaled = np.zeros_like(xys_gt)
        pcd_xyz_sampled = np.zeros_like(pcd_xyz_gt)
        pcd_conf_sampled = np.zeros_like(pcd_rgb_gt[:, 0])  # Assuming confidence is a single channel
        pcd_rgb_sampled = np.zeros_like(pcd_rgb_gt)
        original_coords_arr = original_coords.cpu().numpy()
        resize_ratio = (original_coords_arr[:, -2:].max() / vggt_fixed_resolution)

        xys_gt_scaled[:, 0] = xys_gt[:, 0] / resize_ratio + original_coords_arr[k, 0]
        xys_gt_scaled[:, 1] = xys_gt[:, 1] / resize_ratio + original_coords_arr[k, 1]

        xys_gt_scaled[:, 0] = np.clip(xys_gt_scaled[:, 0], 0, points_3d.shape[2] - 1)
        xys_gt_scaled[:, 1] = np.clip(xys_gt_scaled[:, 1], 0, points_3d.shape[1] - 1)
        
        pcd_xyz_sampled = points_3d[k, xys_gt_scaled[:, 1].astype(int), xys_gt_scaled[:, 0].astype(int)]
        pcd_conf_sampled = depth_conf[k, xys_gt_scaled[:, 1].astype(int), xys_gt_scaled[:, 0].astype(int)]
        pcd_rgb_sampled = images[k, :, xys_gt_scaled[:, 1].astype(int), xys_gt_scaled[:, 0].astype(int)].permute(1, 0).cpu().numpy() * 255

        conf_mask = pcd_conf_sampled > conf_thresh

        pcd_xyz_gt_list.append(pcd_xyz_gt[conf_mask])
        pcd_xyz_sampled_list.append(pcd_xyz_sampled[conf_mask])
        pcd_rgb_gt_list.append(pcd_rgb_gt[conf_mask])
        pcd_rgb_sampled_list.append(pcd_rgb_sampled[conf_mask])

    pcd_xyz_gt_array = np.concatenate(pcd_xyz_gt_list, axis=0)
    pcd_xyz_sampled_array = np.concatenate(pcd_xyz_sampled_list, axis=0)
    pcd_rgb_gt_array = np.concatenate(pcd_rgb_gt_list, axis=0) / 255.0
    pcd_rgb_sampled_array = np.concatenate(pcd_rgb_sampled_list, axis=0) / 255.0

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(pcd_xyz_gt_array)
    pcd_src.colors = o3d.utility.Vector3dVector(pcd_rgb_gt_array)

    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(pcd_xyz_sampled_array)
    pcd_tgt.colors = o3d.utility.Vector3dVector(pcd_rgb_sampled_array)

    trans_init, threshold  = np.eye(4), 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_tgt,
        pcd_src,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd_tgt.transform(reg_p2p.transformation)

    completeness = pcd_src.compute_point_cloud_distance(pcd_tgt)
    accuracy = pcd_tgt.compute_point_cloud_distance(pcd_src)

    accuracy_mean = np.mean(accuracy)  # to be written to txt file
    completeness_mean = np.mean(completeness)  # to be written to txt file
    chamfer_distance = np.mean(np.concatenate([accuracy, completeness]))  # to be written to txt file

    print(f"    --  Accuracy Mean: {accuracy_mean:.4f}")
    print(f"    --  Completeness Mean: {completeness_mean:.4f}")
    print(f"    --  Chamfer Distance: {chamfer_distance:.4f}")

    results = {
        'accuracy_mean': accuracy_mean,
        'completeness_mean': completeness_mean,
        'chamfer_distance': chamfer_distance,
    }

    return results

@torch.inference_mode()
def evaluate_auc(pred_se3, gt_se3, device, return_aligned=False):

    camera_centers_gt = - (gt_se3[:, :3, :3].cpu().numpy().transpose(0, 2, 1) @ gt_se3[:, 3, :3][..., None].cpu().numpy()).squeeze(-1)
    camera_centers_pred = - (pred_se3[:, :3, :3].cpu().numpy().transpose(0, 2, 1) @ pred_se3[:, 3, :3][..., None].cpu().numpy()).squeeze(-1)
    c, R, t = umeyama(camera_centers_pred.T, camera_centers_gt.T)
    
    print("    --  Umeyama Scale: ", c)
    print("    --  Umeyama Rotation: \n", R)
    print("    --  Umeyama Translation: \n", t)

    ext_transform = np.eye(4)
    ext_transform[:3, :3] = R
    ext_transform[:3, 3:] = t
    ext_transform = np.linalg.inv(ext_transform)

    pred_aligned = np.zeros((pred_se3.shape[0], 4, 4))
    pred_aligned[:, :3, :3] = pred_se3[:, :3, :3].cpu().numpy()
    pred_aligned[:, :3, 3] = pred_se3[:, 3, :3].cpu().numpy() * c
    pred_aligned[:, 3, 3] = 1.0
    pred_aligned = np.einsum('bmn,bnk->bmk', pred_aligned, ext_transform[None])

    pred_se3_aligned = torch.eye(4, device=device).unsqueeze(0).repeat(len(pred_se3), 1, 1)
    pred_se3_aligned[:, :3, :3] = torch.tensor(pred_aligned[:, :3, :3], device=device)
    pred_se3_aligned[:, 3, :3] = torch.tensor(pred_aligned[:, :3, 3], device=device)

    rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_se3_aligned, gt_se3, device)
    print(f"    --  Pair Rot   Error (Deg) of Vanilla: {rel_rangle_deg.mean():10.2f}")
    print(f"    --  Pair Trans Error (Deg) of Vanilla: {rel_tangle_deg.mean():10.2f}")

    rError = rel_rangle_deg.cpu().numpy()
    tError = rel_tangle_deg.cpu().numpy()
    Racc_5 = np.mean(rError < 5) * 100
    Tacc_5 = np.mean(tError < 5) * 100
    Racc_15 = np.mean(rError < 15) * 100
    Tacc_15 = np.mean(tError < 15) * 100

    Auc_30 = calculate_auc_np(rError, tError, max_threshold=30)
    print(f"    --  AUC at 30: {Auc_30:.4f}")

    results = {
        'rel_rangle_deg': rel_rangle_deg.mean(),
        'rel_tangle_deg': rel_tangle_deg.mean(),
        'Racc_5': Racc_5,
        'Tacc_5': Tacc_5,
        'Racc_15': Racc_15,
        'Tacc_15': Tacc_15,
        'Auc_30': Auc_30,
    }

    if return_aligned:
        return results, pred_se3_aligned, c, R, t
    else:
        return results
    
def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    
    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

def camera_to_rel_deg(pred_se3, gt_se3, device, batch_size=1):
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.

    Args:
    - pred_se3: Predicted camera represented as a batch of 4x4 SE3 transformation matrices.
    - gt_se3: Ground truth camera represented as a batch of 4x4 SE3 transformation matrices.
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    - rel_rotation_angle_deg, rel_translation_angle_deg: Relative rotation and translation angles in degrees.
    """

    with torch.no_grad():
        # Generate pairwise indices to compute relative poses
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, gt_se3.shape[0] // batch_size)
        pair_idx_i1 = pair_idx_i1.to(device)

        # Compute relative camera poses between pairs
        # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
        # This is possible because of SE3
        # Calculate in two direction to eliminate the bias of the order of camera pairs
        relative_pose_gt_fw = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
        relative_pose_pred_fw = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

        relative_pose_gt_bw = closed_form_inverse(gt_se3[pair_idx_i2]).bmm(gt_se3[pair_idx_i1])
        relative_pose_pred_bw = closed_form_inverse(pred_se3[pair_idx_i2]).bmm(pred_se3[pair_idx_i1])

        relative_pose_gt = torch.cat((relative_pose_gt_fw, relative_pose_gt_bw), dim=0)
        relative_pose_pred = torch.cat((relative_pose_pred_fw, relative_pose_pred_bw), dim=0)

        # Compute the difference in rotation and translation
        # between the ground truth and predicted relative camera poses
        rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
        rel_tangle_deg = translation_angle(relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3])

    return rel_rangle_deg, rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram))


def calculate_auc(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Define histogram bins
    bins = torch.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(max_errors, bins=max_threshold + 1, min=0, max=max_threshold)

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()


def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2


def closed_form_inverse(se3):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.
    """
    R = se3[:, :3, :3]
    T = se3[:, 3:, :3]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # Compute the left part of the inverse transformation
    left_bottom = -T.bmm(R_transposed)
    left_combined = torch.cat((R_transposed, left_bottom), dim=1)

    # Keep the right-most column as it is
    right_col = se3[:, :, 3:].detach().clone()
    inverted_matrix = torch.cat((left_combined, right_col), dim=-1)

    return inverted_matrix


def rotation_angle(rot_gt, rot_pred, batch_size=None):
    # rot_gt, rot_pred (B, 3, 3)
    rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-4)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def compute_ARE(rotation1, rotation2):
    if isinstance(rotation1, torch.Tensor):
        rotation1 = rotation1.cpu().detach().numpy()
    if isinstance(rotation2, torch.Tensor):
        rotation2 = rotation2.cpu().detach().numpy()

    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    error = theta * 180 / np.pi
    return np.minimum(error, np.abs(180 - error))


def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)


def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.

    More specifically::

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)