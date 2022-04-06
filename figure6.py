import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from macros import *


def create_figure6(df_images):
    # convertion of distance 3 to 'Control'
    df_images['DistanceFromCHXStr'] = df_images['DistanceFromCHX'].apply(lambda x: str(x))
    df_images.loc[df_images['DistanceFromCHXStr'] == '3.0', 'DistanceFromCHXStr'] = 'Control'

    df_images['OuterContourArea'] *= df_images['MicrometersInPixels'] * df_images['MicrometersInPixels']
    df_images['OuterContourPerimeter'] *= df_images['MicrometersInPixels']
    df_images['InnerContourArea'] *= df_images['MicrometersInPixels'] * df_images['MicrometersInPixels']
    df_images['InnerContourPerimeter'] *=  df_images['MicrometersInPixels']

    fig6 = plt.figure(constrained_layout=True, figsize=(5, 5))
    spec6 = gridspec.GridSpec(ncols=3, nrows=3, figure=fig6)

    fig6.add_subplot(spec6[0, 0])
    image_05cm_day1 = df_images.loc['Plate 120 CHX 0.5 cm day 1 GFP_RGB_GFP_34_96.tif']['RawImage3D']# * 100 / 70
    image_05cm_day1 = cv2.normalize(image_05cm_day1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imshow(image_05cm_day1)
    plt.axis('off')

    fig6.add_subplot(spec6[0, 1])
    image_05cm_day2 = df_images.loc['Plate 120 CHX 0.5 cm day 2 GFP_RGB_GFP_34_96.tif']['RawImage3D']# * 100 / 70
    image_05cm_day2 = cv2.normalize(image_05cm_day2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imshow(image_05cm_day2)
    plt.axis('off')

    fig6.add_subplot(spec6[0, 2])
    image_05cm_day3 = df_images.loc['Plate 120 CHX 0.5 cm day 3 GFP_RGB_GFP_20_63.tif']['RawImage3D']# * 300 / 100
    image_05cm_day3 = cv2.normalize(image_05cm_day3.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imshow(image_05cm_day3)
    plt.axis('off')

    fig6.add_subplot(spec6[1:3, 0:3])
    colors = ['#2d8b27','#969696']
    sns.set_palette(sns.color_palette(colors))

    ax = sns.pointplot(x='Day', y='OuterContourArea', data=df_images, hue='DistanceFromCHXStr', hue_order=['0.5', 'Control'], capsize=.05)
    plt.ylabel("Periphery Contour Area ($\mu m^{2}$)", fontsize=AXIS_FONT_SIZE)
    plt.xlabel('Day', fontsize=AXIS_FONT_SIZE)
    ax.legend(loc='upper left', title=DISTANCE_LEGEND_TITLE)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)
    plt.setp(ax.get_legend().get_title(), fontsize=LEGEND_TITLE_FONT_SIZE)
    plt.setp(ax.get_legend().get_texts(), fontsize=LEGEND_TEXT_FONT_SIZE)
    plt.yticks([1E7, 2E7, 3E7, 4E7, 5E7, 6E7, 7E7, 8E7, 9E7])
    plt.savefig('./Figures/6.png')