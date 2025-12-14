import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import utils.colmap as colmap_utils
from utils.umeyama import umeyama
from utils.metric_torch import evaluate_auc
from argparse import ArgumentParser


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--data_dir_pred', type=str, help='path of pred bin files', default=None)
    parser.add_argument('--data_dir_gt', type=str, help='path of gt bin files', default=None)
    parser.add_argument('--device', type=str, help='device to use', default='cuda:0')
    args = parser.parse_args(sys.argv[1:])

    results_rel_rangle_deg = []
    results_rel_tangle_deg = []
    results_Auc_30 = []

    print(f"Processing scene: {args.data_dir_pred}")
    sparse_dir_gt = os.path.join(args.data_dir_gt, "sparse", "0")
    images_dir = os.path.join(args.data_dir_gt, "images")

    cameras_gt = colmap_utils.read_cameras_binary(os.path.join(sparse_dir_gt, "cameras.bin"))
    images_gt = colmap_utils.read_images_binary(os.path.join(sparse_dir_gt, "images.bin"))
    pcd_gt = colmap_utils.read_points3D_binary(os.path.join(sparse_dir_gt, "points3D.bin"))
    images_gt_keys = sorted(list(images_gt.keys()))
    images_gt = {id: images_gt[id] for id in list(images_gt_keys)}

    sparse_dir_pred = os.path.join(args.data_dir_pred, "sparse", "0")
    cameras_pred = colmap_utils.read_cameras_binary(os.path.join(sparse_dir_pred, "cameras.bin"))
    images_pred = colmap_utils.read_images_binary(os.path.join(sparse_dir_pred, "images.bin"))
    images_pred_updated = {id: images_pred[id] for id in list(images_gt.keys())}
    pcd_pred = colmap_utils.read_points3D_binary(os.path.join(sparse_dir_pred, "points3D.bin"))

    # print(f"GT's intrinsics: {cameras_gt[1].params}")
    # print(f"Pred's intrinsics: {cameras_pred[1].params}")
    # diff = cameras_gt[1].params - cameras_pred[1].params
    # print("Intrinsic Difference", np.linalg.norm(diff[:2] / cameras_gt[1].params[2:]))

    translation_gt = torch.tensor([image.tvec for image in images_gt.values()], device=args.device)
    rotation_gt = torch.tensor([colmap_utils.qvec2rotmat(image.qvec) for image in images_gt.values()], device=args.device)
    gt_se3 = torch.eye(4, device=args.device).unsqueeze(0).repeat(len(images_gt), 1, 1)
    gt_se3[:, :3, :3] = rotation_gt
    gt_se3[:, 3, :3] = translation_gt

    translation_pred = torch.tensor([image.tvec for image in images_pred_updated.values()], device=args.device)
    rotation_pred = torch.tensor([colmap_utils.qvec2rotmat(image.qvec) for image in images_pred_updated.values()], device=args.device)
    pred_se3 = torch.eye(4, device=args.device).unsqueeze(0).repeat(len(images_pred_updated), 1, 1)
    pred_se3[:, :3, :3] = rotation_pred
    pred_se3[:, 3, :3] = translation_pred

    auc_results, _, c, R, t = evaluate_auc(pred_se3, gt_se3, args.device, return_aligned=True)

    result_file = os.path.join(args.data_dir_pred, f"pose_results.txt")
    with open(result_file, "w") as f:
        f.write(f"Image Count: {len(gt_se3)},\n")
        f.write(f"Relative Rotation Error (degrees): {auc_results['rel_rangle_deg']},\n")
        f.write(f"Relative Translation Error (degrees): {auc_results['rel_tangle_deg']},\n")
        f.write(f"AUC at 30 degrees: {auc_results['Auc_30']},\n")
        f.write(f"Racc_5: {auc_results['Racc_5']},\n")
        f.write(f"Racc_15: {auc_results['Racc_15']},\n")
        f.write(f"Tacc_5: {auc_results['Tacc_5']},\n")
        f.write(f"Tacc_15: {auc_results['Tacc_15']},\n")
    
    print("[INFO] Pose evaluation results saved to {}".format(result_file))
        