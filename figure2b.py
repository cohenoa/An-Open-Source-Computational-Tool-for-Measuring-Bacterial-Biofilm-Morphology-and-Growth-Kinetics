import matplotlib.pyplot as plt
import seaborn as sns

from macros import *


def create_figure2b(df_images, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = sns.pointplot(ax=axes[0], x="Day", y="horizontal_radii_ratio", data=df_images, hue='DistanceFromCHXStr',
                       hue_order=['1.0', '1.5', '2.0', 'Control'], capsize=.05)
    axes[0].set_ylabel("Horizontal Unexposed/Exposed Radii Ratio", fontsize=AXIS_FONT_SIZE)
    axes[0].set_xlabel("Day", fontsize=AXIS_FONT_SIZE)
    axes[0].set(ylim=(0.7, 1.5))
    axes[0].tick_params(labelsize=AXIS_TICK_SIZE)
    axes[0].hlines(1, -0.2, 2.2, color='gray', linestyles='dashed')
    axes[0].legend(loc='upper left', title=DISTANCE_LEGEND_TITLE)

    plt.setp(ax.get_legend().get_title(), fontsize=LEGEND_TITLE_FONT_SIZE) # for legend title
    plt.setp(ax.get_legend().get_texts(), fontsize=LEGEND_TEXT_FONT_SIZE) # for legend text

    ax = sns.pointplot(ax=axes[1], x="Day", y="vertical_radii_ratio", data=df_images, hue='DistanceFromCHXStr',  hue_order=['1.0', '1.5', '2.0', 'Control'],  capsize=.05)
    axes[1].set_ylabel("Vertical Unexposed/Exposed Radii Ratio", fontsize=AXIS_FONT_SIZE)
    axes[1].set_xlabel("Day", fontsize=AXIS_FONT_SIZE)
    axes[1].set(ylim=(0.7, 1.5))
    axes[1].hlines(1, -0.2, 2.2, color='gray', linestyles='dashed')
    axes[1].tick_params(labelsize=AXIS_TICK_SIZE)
    axes[1].legend(loc='upper left', title=DISTANCE_LEGEND_TITLE)

    plt.setp(ax.get_legend().get_title(), fontsize=LEGEND_TITLE_FONT_SIZE) # for legend title
    plt.setp(ax.get_legend().get_texts(), fontsize=LEGEND_TEXT_FONT_SIZE) # for legend text
    plt.savefig(os.path.join(output_dir, 'Figure_2b.png'))
