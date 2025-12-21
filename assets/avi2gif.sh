#!/bin/bash
MAX_SIZE_K=5000

for i in {0..5}; do
    input="./failure/tooth_${i}.mp4"
    output="./failure/tooth_${i}.gif"
    palette="./failure/palette_${i}.png"
    echo "正在处理 $input → $output ..."

    # 生成全局调色板
    ffmpeg -y -i "$input" -vf "palettegen=stats_mode=diff" "$palette"
    # 使用调色板生成 GIF,限制文件大小
    ffmpeg -y -i "$input" -i "$palette" -lavfi "[0:v][1:v]paletteuse=dither=bayer:bayer_scale=2" -fs "${MAX_SIZE_K}k" "$output"
    rm -f "$palette"
    echo "完成：$output"
done
echo "所有视频处理完毕！"