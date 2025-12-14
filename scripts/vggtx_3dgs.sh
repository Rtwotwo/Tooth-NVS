#!/bin/bash
export LD_LIBRARY_PATH="/data1/local_userdata/liuqi/anaconda3/envs/vggtx/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/data2/dataset/Redal/work_tooth_nvs/submodules/simple-knn:$PYTHONPATH"
python="/data1/local_userdata/liuqi/anaconda3/envs/vggtx/bin/python"
ROOT_DIR="/data2/dataset/Redal/work_tooth_nvs/data"
LOG_DIR="/data2/dataset/Redal/work_tooth_nvs/logs/data_vggt_x"
OUT_DIR="/data2/dataset/Redal/work_tooth_nvs/logs/data_3dgs"
# TOOTH_ID=(0 1 2 3 4 5)
TOOTH_ID=(0)
GPU=5


# 1.提取牙齿的点云,相机参数,图像等信息
for id in "${TOOTH_ID[@]}"; do
    scene_dir="${ROOT_DIR}/tooth_${id}"
    echo "[INFO] 正在处理的场景: ${scene_dir}, 时间: $(date +"%Y-%m-%d %H:%M:%S")"

    CUDA_VISIBLE_DEVICES=${GPU} python demo_colmap.py \
        --scene_dir "${scene_dir}" \
        --shared_camera \
        --use_ga
done
echo "[INFO] 所有场景处理完成!"


# 2.使用Gaussian Splatting进行NVS
# for id in "${TOOTH_ID[@]}"; do 
#     scene_dir="${LOG_DIR}/tooth_${id}"
#     output_dir="${OUT_DIR}/tooth_${id}"
#     mkdir -p "${output_dir}"
#     echo "[INFO] 正在训练的场景: ${scene_dir}, 时间: $(date +"%Y-%m-%d %H:%M:%S")"

#     CUDA_VISIBLE_DEVICES=${GPU} python /data2/dataset/Redal/work_tooth_nvs/clones/gaussian-splatting/train.py \
#         -s "${scene_dir}" \
#         -m "${output_dir}" \
#         --eval \
#         --disable_viewer
#     CUDA_VISIBLE_DEVICES=${GPU} python /data2/dataset/Redal/work_tooth_nvs/clones/gaussian-splatting/render.py \
#         -s "${scene_dir}" \
#         -m "${output_dir}" \
#         --skip_train \
#         --eval
#     CUDA_VISIBLE_DEVICES=${GPU} python /data2/dataset/Redal/work_tooth_nvs/clones/gaussian-splatting/metrics.py \
#         -m "${output_dir}" 
# done
echo "[INFO] 所有场景渲染测试完成!"
