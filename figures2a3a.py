import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from macros import *
from utils import get_ellipse_radii


def create_illustration(df_images, figure_number, output_dir, input_folder):
    index = 'Plate 4 CHX 1 cm day 3 GFP_RGB_GFP_22_38.tif'
    center = df_images.loc[index]['Center']
    center_x = int(center[0])
    img_file = os.path.join(input_folder, 'Day3/{}'.format(index))
    assert (os.path.exists(img_file))
    image = df_images.loc[index]['RawImage3D']
    distance_from_CHX_map = df_images.loc[index]['distance_from_CHX_map']
    norm_image = df_images.loc[index]['NormalizedImage']

    # Compute mask based on outer contour
    if figure_number == 2:
        contour = df_images.loc[index]['OuterContourObj']
        _, _, left_contour = get_ellipse_radii(df_images.loc[index]['NormalizedImage'], \
                                               df_images.loc[index]['OuterContourObj'], \
                                               df_images.loc[index]['Center'], 'LEFT')
        _, _, right_contour = get_ellipse_radii(df_images.loc[index]['NormalizedImage'],
                                                df_images.loc[index]['OuterContourObj'],
                                                df_images.loc[index]['Center'], 'RIGHT')
    elif figure_number == 3:
        contour = df_images.loc[index]['InnerContourObj']
        _, _, left_contour = get_ellipse_radii(df_images.loc[index]['NormalizedImage'], \
                                               df_images.loc[index]['InnerContourObj'], \
                                               df_images.loc[index]['Center'], 'LEFT')
        _, _, right_contour = get_ellipse_radii(df_images.loc[index]['NormalizedImage'], \
                                                df_images.loc[index]['InnerContourObj'], \
                                                df_images.loc[index]['Center'], 'RIGHT')

    # BEGIN

    whole_colony_mask = np.zeros(norm_image.shape)
    whole_colony_mask = cv2.drawContours(whole_colony_mask, [contour], -1, (1, 1, 1), cv2.FILLED)

    image_with_background = np.zeros((norm_image.shape[0], norm_image.shape[1], 3))
    image_with_background[whole_colony_mask == 1, 1] = norm_image[whole_colony_mask == 1]

    # Red channel
    image_with_background[whole_colony_mask == 0, 0] = \
        cv2.normalize(distance_from_CHX_map.astype('float'), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                      dtype=cv2.CV_32F)[
            whole_colony_mask == 0] / 2
    image_with_background[image_with_background < 0] = 0  # fixing normalization bug of cv2

    # convert to gray for visualization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
    image[:, :, 0] = image[:, :, 1]
    image[:, :, 2] = image[:, :, 1]
    image = cv2.drawContours(image, [contour], -1, RED, PLOT_WIDTH)
    image_with_background = cv2.drawContours(image_with_background, [contour], -1, RED, PLOT_WIDTH)

    # draw vertical line
    image = cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (255, 255, 0), thickness=2)

    image_left_with_mirror, left_contour_mirrored_with_ellipse = draw_half(image, center_x, contour, left_contour, 'LEFT')
    image_right_with_mirror, right_contour_mirrored_with_ellipse = draw_half(image, center_x, contour, right_contour, 'RIGHT')

    plot_figure(image_with_background,
                image_left_with_mirror,
                left_contour_mirrored_with_ellipse,
                image_right_with_mirror,
                right_contour_mirrored_with_ellipse, figure_number, output_dir)


def draw_half(image, center_x, original_contour, left_contour, type):

    height = image.shape[0]
    contour_mirrored = np.zeros(image.shape, dtype=np.int8)
    contour_mirrored = cv2.drawContours(contour_mirrored, [original_contour], -1, RED, PLOT_WIDTH)

    if type == 'LEFT':
        contour_mirrored[:, center_x:contour_mirrored.shape[1]] = 0
        mirror = contour_mirrored[:, center_x:0:-1]
        contour_mirrored[:, center_x:center_x + mirror.shape[1]] = mirror[:,:contour_mirrored.shape[1] - center_x]
        image_half = image[:, :center_x].copy()
        image_half_with_mirror = np.zeros(image.shape)
        image_half_with_mirror[:, :center_x, :] = image_half
    elif type == 'RIGHT':
        contour_mirrored[:, 0:center_x] = 0
        mirror = contour_mirrored[:, contour_mirrored.shape[1]:center_x:-1]  # fliplr operation
        left_half_width = center_x
        offset = mirror.shape[1] - left_half_width
        mirror = mirror[:, offset:]  # fetch end of right_contour_mirrored section
        contour_mirrored[:, center_x - mirror.shape[1]:center_x] = mirror
        image_half = image[:, center_x:].copy()
        image_half_with_mirror = np.zeros(image.shape)
        image_half_with_mirror[:, center_x:, :] = image_half


    LINE_COLOR = (255, 255, 0)
    image_half_with_mirror = cv2.line(image_half_with_mirror, (center_x, 0), (center_x, height), LINE_COLOR,
                                      thickness=2)
    image_half_with_mirror = cv2.addWeighted(image_half_with_mirror.astype('uint8'), 1, \
                                             contour_mirrored.astype('uint8'), 1.5, 0)

    # plot the contour with the ellipse
    ellipse = cv2.fitEllipse(left_contour)
    contour_mirrored_with_ellipse = np.zeros(image.shape)
    contour_mirrored_with_ellipse = cv2.ellipse(contour_mirrored_with_ellipse, ellipse, WHITE,
                                                     ELLIPSE_LINE_WIDTH)
    contour_mirrored_with_ellipse = cv2.addWeighted(contour_mirrored_with_ellipse.astype('uint8'), 0.6,
                                                         contour_mirrored.astype('uint8'), 1.5, 1)

    return image_half_with_mirror, contour_mirrored_with_ellipse


def plot_figure(image_with_background,
                image_left_with_mirror,
                left_contour_mirrored_with_ellipse,
                image_right_with_mirror,
                right_contour_mirrored_with_ellipse, figure_number, output_dir):
    # Plotting the figure
    figure = plt.figure(constrained_layout=True, figsize=(8, 4))
    spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=figure)
    figure.add_subplot(spec2[0:2, 0])
    plt.imshow(image_with_background)
    plt.axis('off')

    figure.add_subplot(spec2[0, 1])
    plt.imshow(image_left_with_mirror)
    plt.axis('off')

    figure.add_subplot(spec2[0, 2])
    plt.imshow(left_contour_mirrored_with_ellipse)
    plt.axis('off')

    figure.add_subplot(spec2[1, 1])
    plt.imshow(image_right_with_mirror)
    plt.axis('off')

    figure.add_subplot(spec2[1, 2])
    plt.imshow(right_contour_mirrored_with_ellipse)
    plt.axis('off')

    if figure_number == 2:
        plt.savefig(os.path.join(output_dir, 'Figure_2a.png'))
    elif figure_number == 3:
        plt.savefig(os.path.join(output_dir, 'Figure_3a.png'))
