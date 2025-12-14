import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch.nn.functional as F
import torchvision.transforms.functional as TF

from rich import print as rich_print

class RichLogger:
    def __init__(self, default_tag="3RGS", default_style="bold blue"):
        self.default_tag = default_tag
        self.default_style = default_style
        self._log_styles = {
            "3RGS": "bold green",
            "GUI": "bold magenta",
            "Eval": "bold red",
        }

    def add_style(self, tag: str, style: str):
        """Add or update a style for a tag."""
        self._log_styles[tag] = style

    def get_style(self, tag: str) -> str:
        return self._log_styles.get(tag, self.default_style)

    def log(self, *args, tag: str = None):
        tag = tag or self.default_tag
        style = self.get_style(tag)
        rich_print(f"[{style}]{tag}:[/{style}]", *args)

logger = RichLogger()

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False, edges=None, min_map=None, max_map=None, abs=False, colorbar=True, title_font_size=20):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    # Evaluate rotation error between estimated and ground truth camera poses
    def compute_rotation_error(R1, R2):
        """
        Compute rotation error in degrees between two rotation matrices
        """
        # R1,R2: [N,3,3]
        R_diff = torch.matmul(R1, R2.transpose(-2,-1))
        # Convert to axis-angle
        tr = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1) 
        theta = torch.acos(torch.clamp((tr - 1) / 2, -1, 1))
        return theta * 180 / torch.pi # Convert to degrees
    # Alternative implementation
    def rotation_distance(R1, R2, eps=1e-7):
        R_diff = R1@R2.transpose(-2,-1)
        trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
        angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()  # Returns radians
        return angle * 180 / torch.pi
    a = torch.tensor(traj_est_aligned.poses_se3).float()
    b = torch.tensor(traj_ref.poses_se3).float()
    est_Rs = torch.stack([c[:3,:3] for c in a])
    gt_Rs = torch.stack([c[:3,:3] for c in b])
    #rot_errors = compute_rotation_error(est_Rs, gt_Rs)
    rot_errors = rotation_distance(est_Rs, gt_Rs)
    mean_rot_error = rot_errors.mean().item()
    median_rot_error = rot_errors.median().item()
    print(f"Rotation Error - Mean: {mean_rot_error:.2f}°, Median: {median_rot_error:.2f}°")
    # Plot histogram of rotation errors
    plt.figure(figsize=(10, 6))
    plt.hist(rot_errors.cpu().numpy(), bins=30)
    plt.xlabel('Rotation Error (degrees)')
    plt.ylabel('Count')
    plt.title(f'Rotation Error Distribution at Step {label}\nMean: {mean_rot_error:.2f}°, Median: {median_rot_error:.2f}°')
    plt.savefig(os.path.join(plot_dir, f'rotation_error_hist_step{label}.png'))
    plt.close()

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    #ape_stats['rmse_per_frame'] = ape_metric.get_statistic('rmse_per_frame').tolist()
    ape_stats['rot_error_frame'] = rot_errors.numpy().tolist()
    logger.log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    err = np.abs(ape_metric.error) if abs else ape_metric.error
    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure(figsize=(12, 8))  # Make figure larger with 12x8 inches size
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE: {ape_stat:.3f}", font={'size': title_font_size})
    # close the axis if not colorbar
    if not colorbar:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    fixed_traj_colormap(
        ax,
        traj_est_aligned,
        err,
        plot_mode,
        min_map=ape_stats["min"] if min_map is None else min_map,
        max_map=ape_stats["max"] if max_map is None else max_map,
        colorbar=colorbar,
    )
    ax.legend(fontsize=title_font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=150, bbox_inches="tight", pad_inches=0)  # Increased DPI for better quality
    plt.close(fig)  

    return ape_stats



def eval_ate(poses_est, poses_gt, save_dir, iterations, final=False, monocular=False, dir_name = 'plot',edges=None):
    trj_data = dict()
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []


    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for uid, (pose_est, pose_gt) in enumerate(zip(poses_est, poses_gt)):
        pose_est = pose_est[0].detach().cpu().numpy()
        pose_gt = pose_gt[0].detach().cpu().numpy()
        trj_id.append(uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, dir_name)
    os.makedirs(plot_dir, exist_ok=True)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    #wandb.logger.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


from evo.tools.plot import colored_line_collection, set_aspect_equal_3d
from matplotlib import cm
def fixed_traj_colormap(ax, traj, array, plot_mode, min_map, max_map, title="", colorbar=True):
    """
    color map a path/trajectory in xyz coordinates according to
    an array of values
    :param ax: plot axis
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param array: Nx1 array of values used for color mapping
    :param plot_mode: PlotMode
    :param min_map: lower bound value for color mapping
    :param max_map: upper bound value for color mapping
    :param title: plot title
    """
    pos = traj.positions_xyz
    norm = matplotlib.colors.Normalize(vmin=min_map, vmax=max_map, clip=True)
    mapper = cm.ScalarMappable(
        norm=norm,
        cmap=SETTINGS.plot_trajectory_cmap)  # cm.*_r is reversed cmap
    mapper.set_array(array)
    colors = [mapper.to_rgba(a) for a in array]
    line_collection = colored_line_collection(pos, colors, plot_mode)
    ax.add_collection(line_collection)
    ax.autoscale_view(True, True, True)
    if plot_mode == PlotMode.xyz:
        ax.set_zlim(
            np.amin(traj.positions_xyz[:, 2]),
            np.amax(traj.positions_xyz[:, 2]))
        if SETTINGS.plot_xyz_realistic:
            set_aspect_equal_3d(ax)
    fig = plt.gcf()
    if colorbar:
        cbar = fig.colorbar(
            #mapper, ticks=[min_map, (max_map - (max_map - min_map) / 2), max_map])
            # fix the colorbar
            mapper, ticks=[min_map, (max_map - (max_map - min_map) / 2), max_map], ax=plt.gca())
        cbar.ax.set_yticklabels([
            "{0:0.3f}".format(min_map),
            "{0:0.3f}".format(max_map - (max_map - min_map) / 2),
            "{0:0.3f}".format(max_map)
        ])
    if title:
        ax.legend(frameon=True)
        plt.title(title)