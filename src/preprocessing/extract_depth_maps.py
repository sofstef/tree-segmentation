import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

import argparse


##### file reading #####
SAMPLE_PREFIX = "Capture_Sample_"

def extract_data(filename):
    # Read in the data
    data = np.genfromtxt(filename, delimiter=",")
    depth_range = data[:, 2]

    return data[:, 0], data[:, 1], depth_range.astype(np.float64)


def save_depths(infile, save_dir, width, height):

    print("extracting ...")
    x, y, depth = extract_data(infile)
    depth = np.reshape(depth, (width, height))
    file_name = infile.split('/')[-1]
    
    print("drawing ...")
    plt.imsave(save_dir + file_name + ".jpg", depth)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     palette = copy.copy(plt.cm.viridis)
#     palette.set_bad("black")
#     ax.matshow(depth, cmap=palette)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     plt.tight_layout()
    

#     plt.savefig(save_dir + file_name + ".jpg", bbox_inches='tight', pad_inches = 0)
    print("done!")
    print("file saved as " + file_name + ".jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert depth files to readable view"
    )
    parser.add_argument(
        "--raw_dir",
        help="Path to raw sample files.",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
    )
    parser.add_argument(
        "--width",
        help="Expected ARCore image width",
        type=int,
    )
    parser.add_argument(
        "--height",
        help="Expected ARCore image height",
        type=int,
    )
    args = parser.parse_args()
    RAW_DIR = args.raw_dir
    SAVE_DIR = args.save_dir
    
    # fetch samples from directory
    file_names = os.listdir(RAW_DIR)
    file_names = [
        name
        for name in file_names
        if name.startswith(SAMPLE_PREFIX)
        and not (name.endswith(".txt") or name.endswith(".jpeg"))
    ]
    if len(file_names) < 1:
        print(colored("Error: No samples found in directory {}".format(RAW_DIR), "red"))
        exit(1)
    for name in file_names:
        name = RAW_DIR + name
        save_depths(name, args.save_dir, args.width, args.height)
