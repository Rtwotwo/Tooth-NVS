"""
Author: Redal
Date: 2025-12-08
Todo: use pycolmap to convert video into point cloud
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import os
import sys
import cv2
import argparse
import subprocess
import pycolmap
ENV = os.environ.copy()


def VideotoFrames(video_path:str, 
                  output_path:str, 
                  skip_frames:int=1
                  )->None:
    """将视频数据源转换成图片帧,设置skip_frames参数作为跳帧参数处理"""
    assert skip_frames>0, f"视频抽帧skip_frames参数必须大于0!"
    video_cap = cv2.VideoCapture(video_path)
    count_num = 0
    os.makedirs(output_path, exist_ok=True)
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if ret:
            if count_num % skip_frames==0:
                file_path = os.path.join(output_path, f"{count_num//skip_frames}.jpg")
                cv2.flip(frame, 0) # 垂直翻转图片
                cv2.imwrite(file_path, frame)
                print(f"[INFO] 已保存图片至{file_path}")
        else: break
        count_num += 1
    # 释放摄像头资源
    video_cap.release()
    

def PycolmapSFM(args, images_dir:str, 
                database_path:str, 
                points_path:str, 
                dense_path:str=None,
                camera_model:str="SIMPLE_PINHOLE"
                )->None:
    """运行pycolmap的SFM重建系数稀疏点云数据
    images_dir: 抽帧后保存图片文件夹位置, camera_model:相机模型参数
    database_path: 数据库文件位置,由用户决定其名称,后缀格式为.db
    output_path: 稀疏点云数据保存位置"""
    pycolmap.extract_features(database_path=database_path, 
                              image_path=images_dir,
                              camera_model=camera_model)
    # 进行特征匹配并使用增量式SFM
    os.makedirs(points_path, exist_ok=True)
    pycolmap.match_exhaustive(database_path=database_path)
    reconstruction = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=images_dir,
                    output_path=points_path)
    print(f"[INFO] SFM稀疏点云重建任务已完成,点云数据保存在{points_path}!")
    # 进行稠密重建以获得点云数据.ply
    os.makedirs(dense_path, exist_ok=True)
    cmd_undistort = [
        "colmap", "image_undistorter",
        "--image_path", images_dir,
        "--input_path", os.path.join(points_path, "0"),
        "--output_path", dense_path,
        "--output_type", "COLMAP"]
    print(f"[INFO] Running: {' '.join(cmd_undistort)}")
    result_undistort = subprocess.run(cmd_undistort, capture_output=True, 
                    text=True, encoding='utf-8', errors='replace', env=ENV)
    # 使用COLMAP命令行执行稠密重建
    cmd1 = ["colmap", "patch_match_stereo",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.max_image_size", "2000"]
    print(f"[INFO] Running: {' '.join(cmd1)}")
    result1 = subprocess.run(cmd1, capture_output=True, text=True, env=ENV)
    # 创建点云存储路径,生成.ply点云数据
    plyname = os.path.basename(args.video_path).split(".")[0] + ".ply"
    fused_ply = os.path.join(dense_path, plyname)
    cmd2 = ["colmap", "stereo_fusion",
        "--workspace_path", dense_path,
        "--output_path", fused_ply]
    print(f"[INFO] Running: {' '.join(cmd2)}")
    result2 = subprocess.run(cmd2, capture_output=True, text=True, env=ENV)


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./logs/tooth_11.mp4", help="Video path")
    parser.add_argument("--skip_frames", type=int, default=1, help="Skip frames")
    parser.add_argument("--frames_path", type=str,  default="./logs/data/tooth_11/images", help="Output path")
    parser.add_argument("--points_path", type=str, default="./logs/data/tooth_8/points", help="Output path")
    parser.add_argument("--dense_path", type=str, default="./logs/data/tooth_8/dense", help="Output path")
    parser.add_argument("--database_path", type=str, default="./logs/data/tooth_8/database.db", help="Database path")
    args = parser.parse_args()
    # 进行pycolmap三维重建任务
    VideotoFrames(video_path=args.video_path, 
                  output_path=args.frames_path, 
                  skip_frames=args.skip_frames)
    # PycolmapSFM(args, images_dir=args.frames_path, 
    #              database_path=args.database_path, 
    #               dense_path=args.dense_path, 
    #               points_path=args.points_path)


if __name__ == "__main__":
    main()