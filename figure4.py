import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from augmentation_utils import get_intensities_orientation
from macros import *
from utils import *


def get_intensities(image, outer_contour, inner_contour, center, day):
    if day != 3:
        return None, None, None, None

    # inner core
    outer_mask, inner_mask = get_masks(image, outer_contour, inner_contour)
    inner_image = cv2.bitwise_and(image, inner_mask)

    left_inner_mask = inner_mask[:, 0:int(center[0])]
    left_image = inner_image[:, 0:int(center[0])]
    mean_inner_left_intensity = np.mean(left_image[np.where(left_inner_mask == 1)])

    right_inner_mask = inner_mask[:, int(center[0]):]
    right_image = inner_image[:, int(center[0]):]
    mean_inner_right_intensity = np.mean(right_image[np.where(right_inner_mask == 1)])

    # outer periphery
    outer_only_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
    outer_image = cv2.bitwise_and(image, outer_only_mask)

    left_outer_mask = outer_only_mask[:, 0:int(center[0])]
    left_outer_image = outer_image[:, 0:int(center[0])]
    mean_outer_left_intensity = np.mean(left_outer_image[np.where(left_outer_mask == 1)])

    right_outer_mask = outer_only_mask[:, int(center[0]):]
    right_outer_image = outer_image[:, int(center[0]):]
    mean_outer_right_intensity = np.mean(right_outer_image[np.where(right_outer_mask == 1)])

    return mean_outer_left_intensity, mean_outer_right_intensity, mean_inner_left_intensity, mean_inner_right_intensity


def fill_df_intensities(df_images):
    df_images['outer_left_intensity'] = \
        df_images.apply(lambda x: get_intensities(x['RawImage'], x['OuterContourObj'], x['InnerContourObj'], x['Center'], x['Day'])[0], axis=1)
    df_images['outer_right_intensity'] = \
        df_images.apply(lambda x: get_intensities(x['RawImage'], x['OuterContourObj'], x['InnerContourObj'], x['Center'], x['Day'])[1], axis=1)
    df_images['inner_left_intensity'] = \
        df_images.apply(lambda x: get_intensities(x['RawImage'], x['OuterContourObj'], x['InnerContourObj'], x['Center'], x['Day'])[2], axis=1)
    df_images['inner_right_intensity'] = \
        df_images.apply(lambda x: get_intensities(x['RawImage'], x['OuterContourObj'], x['InnerContourObj'], x['Center'], x['Day'])[3], axis=1)

    df_images['outer_right_to_left_ratio_intensity'] = df_images['outer_right_intensity'] / df_images['outer_left_intensity']
    df_images['inner_right_to_left_ratio_intensity'] = df_images['inner_right_intensity'] / df_images['inner_left_intensity']


def fill_df_intensities_use_orientation(df_images):
    tqdm.pandas()

    df_images['outer_left_intensity'] = \
        df_images.progress_apply(lambda x: get_intensities_orientation(x['RawImage'], x['OuterContourObj'], x['InnerContourObj'], x['Center'],x['orientation'])[0], axis=1)
    df_images['outer_right_intensity'] = \
        df_images.progress_apply(lambda x: get_intensities_orientation(x['RawImage'], x['OuterContourObj'], x['InnerContourObj'], x['Center'], x['orientation'])[1], axis=1)
    df_images['inner_left_intensity'] = \
        df_images.progress_apply(lambda x: get_intensities_orientation(x['RawImage'], x['OuterContourObj'], x['InnerContourObj'], x['Center'], x['orientation'])[2], axis=1)
    df_images['inner_right_intensity'] = \
        df_images.progress_apply(lambda x: get_intensities_orientation(x['RawImage'], x['OuterContourObj'], x['InnerContourObj'], x['Center'], x['orientation'])[3], axis=1)

    df_images['outer_right_to_left_ratio_intensity'] = df_images['outer_right_intensity'] / df_images['outer_left_intensity']
    df_images['inner_right_to_left_ratio_intensity'] = df_images['inner_right_intensity'] / df_images['inner_left_intensity']
    print(len(df_images))


def create_figure4(df_day3, output_dir):
    sns.reset_defaults()
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='DistanceFromCHX', y="ratio_right_to_left_intensity", hue='Region type', data=df_day3, linewidth=1.5, width=0.6) #width=0.5, linewidth = 0.7
    # plt.legend(loc='lower right')
    ax.set_xticklabels(['1.0', '1.5', '2.0', 'Control'])
    ax.set_ylabel("Exposed/Control Pixel Intensity Ratio", fontsize=AXIS_FONT_SIZE)
    ax.set_xlabel("Distance from CHX (cm)", fontsize=AXIS_FONT_SIZE)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)
    plt.vlines(0.5, 0.9, 1.1, color='lightgray', linestyles='dashed')
    plt.vlines(1.5, 0.9, 1.1, color='lightgray', linestyles='dashed')
    plt.vlines(2.5, 0.9, 1.1, color='lightgray', linestyles='dashed')
    plt.hlines(1, -0.5, 3.5, color='lightgray', linestyles='dashed')

    plt.setp(ax.get_legend().get_title(), fontsize=LEGEND_TITLE_FONT_SIZE) # for legend title
    plt.setp(ax.get_legend().get_texts(), fontsize=LEGEND_TEXT_FONT_SIZE)  # for legend text
    sns.despine(offset=0, trim=False)

    plt.savefig(os.path.join(output_dir, 'Figure_4.png'))
















