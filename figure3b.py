
import matplotlib.pyplot as plt
import seaborn as sns

from macros import *


def create_figure3b(df_images):

    colors = ['#9ECAE1', '#4292C6', '#084594', '#969696']
    sns.set_palette(sns.color_palette(colors))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = sns.pointplot(ax=axes[0], x="Day", y="inner_horizontal_radii_ratio",
                       data=df_images[df_images['Day'] != 1], hue='DistanceFromCHXStr',
                       hue_order=['1.0', '1.5', '2.0', 'Control'], capsize=.05)
    axes[0].hlines(1, -0.1, 1.1, color='gray', linestyles='dashed')
    axes[0].set_ylabel("Horizontal Control/Exposed Radii Ratio", fontsize=AXIS_FONT_SIZE)
    axes[0].set_xlabel("Day", fontsize=AXIS_FONT_SIZE)
    axes[0].legend(loc='upper left', title=DISTANCE_LEGEND_TITLE)
    axes[0] = ax.set(ylim=(0.7, 1.5))

    ax.tick_params(labelsize=AXIS_TICK_SIZE)
    plt.setp(ax.get_legend().get_title(), fontsize=LEGEND_TITLE_FONT_SIZE) # for legend title
    plt.setp(ax.get_legend().get_texts(), fontsize=LEGEND_TEXT_FONT_SIZE) # for legend text

    ax = sns.pointplot(ax=axes[1], x="Day", y="inner_vertical_radii_ratio",
                       data=df_images[df_images['Day'] != 1], hue='DistanceFromCHXStr',
                       hue_order=['1.0', '1.5', '2.0', 'Control'], capsize=.05)
    axes[1].hlines(1, -0.1, 1.1, color='gray', linestyles='dashed')
    axes[1].set_ylabel("Vertical Control/Exposed Radii Ratio", fontsize=AXIS_FONT_SIZE)
    axes[1].set_xlabel("Day", fontsize=AXIS_FONT_SIZE)
    axes[1].legend(loc='upper left', title=DISTANCE_LEGEND_TITLE)
    axes[1] = ax.set(ylim=(0.7, 1.5))
    ax.tick_params(labelsize=AXIS_TICK_SIZE)
    plt.setp(ax.get_legend().get_title(), fontsize=LEGEND_TITLE_FONT_SIZE) # for legend title
    plt.setp(ax.get_legend().get_texts(), fontsize=LEGEND_TEXT_FONT_SIZE) # for legend text
    plt.savefig('./Figures/3b.png')
