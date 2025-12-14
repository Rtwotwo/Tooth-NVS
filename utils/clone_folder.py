import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--src_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--tgt_dir", type=str, required=True, help="Directory to clone the scene images to")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    for item in os.listdir(args.src_dir):
        src = os.path.join(args.src_dir, item)
        dst = os.path.join(args.tgt_dir, item)
        if os.path.exists(dst):
            continue

        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for file in os.listdir(src):
                os.symlink(os.path.abspath(os.path.join(src, file)), os.path.abspath(os.path.join(dst, file)))
        else:
            os.symlink(os.path.abspath(src), os.path.abspath(dst))