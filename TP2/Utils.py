# ##############################################################################
#  Aprendizagem Profunda, TP2 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################
import os
import shutil

import imageio
import numpy as np
from matplotlib import pyplot as plt


def create_gif(images_frames, gif_name, time=0.2):
    filenames = os.listdir(images_frames)
    filenames.sort()
    print(filenames)
    with imageio.get_writer(gif_name, mode='I', duration=time) as writer:
        for filename in filenames:
            image = imageio.imread(f"{gif_name}/{gif_name}")
            writer.append_data(image)


def plot_statistics(data, title, path):
    # data to be plotted
    x = np.arange(1, len(data) + 1)

    # plotting
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel(title)
    plt.plot(x, data, color="green")
    plt.savefig(path)
    plt.close()


def create_folders(training_path):
    if os.path.exists(training_path):
        # Recursively delete all subfolders and sub files under the folder
        shutil.rmtree(training_path)
    os.makedirs(training_path)