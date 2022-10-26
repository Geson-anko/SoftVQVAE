# Please run in the project root folder.
# Usage:
# $ python scripts/upload_logs.py -d /Volume/NAS
#
# If there is same folder name or file name in the target folder,
# this doesn't copy them into target folder.

import os
import argparse
from pathlib import Path
import shutil


def copy_logs(args) -> None:
    dist_path = Path(args.dist)
    src_path = Path(args.source)

    log_names = os.listdir(src_path)
    for n in log_names:
        src_p = src_path / n
        dist_p = dist_path / n
        if not os.path.exists(dist_p):
            print(f"copy: {src_p} -> {dist_p}")
            if os.path.isdir(src_p):
                shutil.copytree(src_p, dist_p)
            else:
                shutil.copyfile(src_p, dist_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dist", type=str, required=True, help="The distination of logs")
    parser.add_argument(
        "-s", "--source", type=str, required=False, default="./logs", help="The location of log folder."
    )

    copy_logs(parser.parse_args())
