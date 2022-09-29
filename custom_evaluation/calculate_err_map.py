import argparse
import cv2
import numpy as np
import os
import matplotlib as plt
import seaborn as sns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate deltaL, deltaC and deltaE between pairs of images with the same name")
    parser.add_argument('src_file', type=str, help="First file")
    parser.add_argument('dest_file', type=str, help="Second file")
    parser.add_argument('output', type=str, help='name of the output file')
    args = parser.parse_args()

    src_file = args.src_file
    dest_file = args.dest_file
    output = args.output

    if os.path.exists(src_file) and os.path.exists(dest_file):
        s_path = os.path.abspath(src_file)
        d_path = os.path.abspath(dest_file)
        
        s_image = cv2.imread(s_path).astype(np.float32) / 255
        t_image = cv2.imread(d_path).astype(np.float32) / 255
        
        err = np.square(np.subtract(s_image,t_image)).mean(axis=2)


        plt.figure.Figure(figsize = (10,7))
        sns_plot = sns.heatmap(err, cmap="flare", vmin=np.min(err), vmax = np.max(err))
        plt.pyplot.savefig(output, dpi=400)

        
