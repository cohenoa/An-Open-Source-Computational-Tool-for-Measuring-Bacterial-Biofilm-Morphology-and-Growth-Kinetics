import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec

from macros import *


def create_figure1(df_images, output_dir):
    fig1a = plt.figure(constrained_layout=True, figsize=(10, 5))
    spec1a = gridspec.GridSpec(ncols=3, nrows=2, figure=fig1a)

    files = {0: 'Plate 4 CHX 1 cm day 1 GFP_RGB_GFP_22_38.tif',
             1: 'Plate 4 CHX 1 cm day 2 GFP_RGB_GFP_22_38.tif',
             2: 'Plate 4 CHX 1 cm day 3 GFP_RGB_GFP_22_38.tif'}

    files_top_row = {0: 'plate-4-day-1.tif',
             1: 'plate-4-day-2.tif',
             2: 'plate-4-day-3.tif'}

    for k in files:
        fig1a.add_subplot(spec1a[0, k])
        plt.title('Day {}'.format(k + 1))

        image = cv2.imread(files_top_row[k])
        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        plt.imshow(image)
        plt.axis('off')

        fig1a.add_subplot(spec1a[1, k])
        image = df_images.loc[files[k]]['NormalizedImage']
        image = np.dstack((image, image, image))
        outer_contour_obj = df_images.loc[files[k]]['OuterContourObj']
        image = cv2.drawContours(image, [outer_contour_obj], -1, 1, 6)
        plt.imshow(image)
        plt.axis('off')
    plt.savefig(os.path.join(output_dir, '1a.png'))

    plt.figure(figsize=(7, 5))
    ax = sns.pointplot(x="Day", y="OuterContourArea", data=df_images, hue='DistanceFromCHXStr',
                       hue_order=['1.0', '1.5', '2.0', 'Control'], capsize=.05)
    plt.ylabel("Periphery Contour Area ($\mu m^{2}$)", fontsize=AXIS_FONT_SIZE)
    plt.xlabel("Day", fontsize=AXIS_FONT_SIZE)
    ax.get_legend().set_title(DISTANCE_LEGEND_TITLE)
    # ax.get_legend().set
    ax.tick_params(labelsize=AXIS_TICK_SIZE)
    plt.setp(ax.get_legend().get_title(), fontsize=LEGEND_TITLE_FONT_SIZE)  # for legend title
    plt.setp(ax.get_legend().get_texts(), fontsize=LEGEND_TEXT_FONT_SIZE)  # for legend text
    plt.yticks([1E7, 2E7, 3E7, 4E7, 5E7, 6E7, 7E7, 8E7, 9E7])  # added to match Figure 6

    plt.savefig(os.path.join(output_dir, '1b.png'))
