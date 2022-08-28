from matplotlib import pyplot as plt
import argparse

from augmentation_utils import create_control_augmentations
from figure1 import create_figure1
from figure2b import create_figure2b
from figure3b import create_figure3b
from figure4 import create_figure4, fill_df_intensities_use_orientation
from figure5 import create_figure5
from figure6 import create_figure6
from figures2a3a import create_illustration
from stat_utils import figure_4_compute_pvalues, print_stats_table, print_num_repetitions
from utils import *
import seaborn as sns
from macros import *

pd.options.display.min_rows = 10000

sns.set_palette(sns.color_palette(colors))


def create_images_df(input_folder, wanted_distances=None, df_centers=None):
    # Iterate over the images in the folder
    list_dict = []
    for day in [1, 2, 3]:
        total_images = 0
        image_list = sorted(os.listdir(os.path.join(input_folder, 'Day' + str(day))))
        print('Total images in directory for day {}: {}'.format(day, len(image_list)))
        for image_file_name in image_list:
            assert (image_file_name.endswith('tif'))
            plate, distance, _, micrometers_in_pixels = parse_image_filename(image_file_name)

            if wanted_distances is not None and distance not in wanted_distances:
                # skip unwanted distances
                continue

            center = df_centers.loc[image_file_name]['Center'] if df_centers is not None else None
            total_images += 1

            image_file_path = os.path.join(input_folder, 'Day' + str(day), image_file_name)
            assert (os.path.exists(image_file_path))
            image_normalized, image_raw, image_uint, image_3D = read_image(image_file_path)

            # Compute outer contour
            min_thresh_outer, min_thresh_inner, erode_kernel_size, erode_iterations = get_params(image_file_name)

            contours_outer = get_all_contours(image_normalized, min_thresh_outer)
            outer_contour_obj, outer_contour_area, outer_contour_perimeter = \
                fetch_nth_contour(contours_outer, 1)
            # Create outer mask
            image_outer_mask = cv2.drawContours(np.zeros(image_normalized.shape), [outer_contour_obj], -1, 1, -1)

            # Compute inner contour
            inner_contour_obj, inner_contour_area, inner_contour_perimeter, image_contrast = None, None, None, None
            if day != 1:
                min_thresh_inner_original = min_thresh_inner

                lst = [min_thresh_inner]
                # lst = np.arange(0, 1, 0.005)

                for min_thresh_inner in lst:
                    current_image = image_normalized
                    contours_inner = get_all_contours(current_image, min_thresh_inner, True, int(erode_kernel_size),
                                                      int(erode_iterations))

                    try:
                        inner_contour_obj, inner_contour_area, inner_contour_perimeter = \
                            fetch_nth_contour(contours_inner, 2)
                    except:
                        # for thresholds for which no contours are found
                        continue

                    image_debug = image_normalized.copy() * 255
                    image_debug = np.dstack((image_debug, image_debug, image_debug))
                    image_debug = cv2.drawContours(image_debug, [inner_contour_obj], -1, (0, 0, 255), 2)
                    if center is not None:
                        center_x = int(center[0])
                        center_y = int(center[1])
                        height, width, _ = image_debug.shape
                        image_debug = cv2.line(image_debug, (center_x, 0), (center_x, height), (0, 255, 255),
                                               thickness=2)
                        image_debug = cv2.circle(image_debug, (center_x, center_y), 5, color=(0, 0, 255), thickness=3)

                    image_debug = cv2.drawContours(image_debug, [outer_contour_obj], -1, (0, 255, 0), 1)

                    str_distance = str(distance).replace('.', '_')
                    out_path = './THRESHOLD_IMAGES/Day{}/{}cm/Plate{}/'.format(day, str_distance, plate)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                    out_image_name = '{}.png'.format('{:.4f}'.format(min_thresh_inner))
                    cv2.imwrite(out_path + out_image_name, image_debug)

                    if abs(min_thresh_inner - min_thresh_inner_original) < 0.001:
                        chosen_out_path = './THRESHOLD_IMAGES/CHOSEN_Day{}/'.format(day)
                        if not os.path.exists(chosen_out_path):
                            os.makedirs(chosen_out_path)
                        # image_debug = cv2.putText(image_debug, str(min_thresh_inner),
                        #                           (center_x, center_y-200), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 1, 2)
                        cv2.imwrite(chosen_out_path + 'dist_{}cm_Plate{}.png'.format(str_distance, plate), image_debug)

            if inner_contour_obj is None and day != 1:
                print(image_file_name)

            image_dict = {'FileName': image_file_name,
                          'Day': day,
                          'DistanceFromCHX': distance,
                          'Plate': plate,
                          'MicrometersInPixels': micrometers_in_pixels,
                          'RawImage': image_raw,
                          'RawImage3D': image_3D,
                          'NormalizedImage': image_normalized,
                          'OuterMaskImage': image_outer_mask,
                          'OuterContourObj': outer_contour_obj,
                          'OuterContourArea': outer_contour_area,
                          'OuterContourPerimeter': outer_contour_perimeter,
                          'InnerContourObj': inner_contour_obj,
                          'InnerContourArea': inner_contour_area,
                          'InnerContourPerimeter': inner_contour_perimeter,
                          'Center': center

                          }
            debug = False
            if debug:
                print(image_file_name)
                inner_mask_debug = cv2.drawContours(np.zeros(image_raw.shape), [inner_contour_obj], -1, 1, cv2.FILLED)
                plt.imshow(inner_mask_debug)
                print('done with ', image_file_name)

            list_dict.append(image_dict)
        print('Total images used for day {}: {}'.format(day, total_images))

    df_images = pd.DataFrame(list_dict)
    df_images = df_images.set_index('FileName')

    return df_images


if __name__ == '__main__':
    # print('hello')
    # exit(1)
    # asking for two input arguments:
    # (a) Input folder, containing the images (e.g., tif fileS) and matching centers (csv files)
    # (b) Outout folder for generating the images presented in the manuscript.
    parser = argparse.ArgumentParser(description='Running biofilm image processing analysis.')
    # parser.add_argument('-i', '--input_folder', type=argparse.FileType('r'), required=True, help='Path of input folder.')
    parser.add_argument('-o', '--output_folder', required=True, help='Path of output folder.')

    args = parser.parse_args()
    # BASE_PATH = args.input
    # print(BASE_PATH)
    # exit(1)
    output_dir = args.output_folder
    output_dir = output_dir.strip()
    if not output_dir[-1] == '/': # TODO: check how to make this smarter
        output_dir += '/'

    # output_dir = './Figures3/'
    print('$'+output_dir+'$')
    # exit(1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_file = 'df_images.pickle'
    use_existing_df = False

    if use_existing_df:
        df_images = pd.read_pickle(df_file)
    else:
        df_centers = create_centers_df(input_folder=INPUT_FOLDER_CENTERS)
        df_images = create_images_df(input_folder=INPUT_FOLDER, wanted_distances=[1.0, 1.5, 2.0, 3.0],
                                     df_centers=df_centers)
        print('Finished creating df_images, joined with df_centers.')
        fill_df(df_images)
        df_images.to_pickle(df_file)

    print('-----------------------', 'Num Repetitions Stats', '--------------------')
    print_num_repetitions(df_images)

    print('-----------------------', 'Figure 1', '--------------------')
    create_figure1(df_images, output_dir)
    print_stats_table(df_images, 'OuterContourArea', [1, 2, 3])
    plt.close('all')
    print('-----------------------', 'DONE', '--------------------')


    print('-----------------------', 'Figure 2A', '--------------------')
    create_illustration(df_images, 2, output_dir)
    plt.close('all')
    print('-----------------------', 'DONE', '--------------------')

    print('-----------------------', 'Figure 2B', '--------------------')
    create_figure2b(df_images, output_dir)
    print_stats_table(df_images, 'horizontal_radii_ratio', [3])
    print_stats_table(df_images, 'vertical_radii_ratio', [3])
    plt.close('all')
    print('-----------------------', 'DONE', '--------------------')

    print('-----------------------', 'Figure 3A', '--------------------')
    create_illustration(df_images, 3, output_dir)
    plt.close('all')
    print('-----------------------', 'DONE', '--------------------')


    print('-----------------------', 'Figure 3B', '--------------------')
    create_figure3b(df_images, output_dir)
    print_stats_table(df_images, 'inner_horizontal_radii_ratio', [3])
    print_stats_table(df_images, 'inner_vertical_radii_ratio', [3])
    plt.close('all')

    df_images_day3 = df_images[df_images.Day == 3].copy()
    df_images_day3 = create_control_augmentations(df_images_day3)
    fill_df_intensities_use_orientation(df_images_day3)

    # group by file name in order to average over augmentations
    df_images_day3_before_grouping = df_images_day3.copy()
    average_result = df_images_day3.groupby('FileName').mean()
    df_images_day3 = df_images_day3.drop(df_images_day3[df_images_day3['DistanceFromCHX'] == 3.0].index)
    df_images_day3 = pd.concat([df_images_day3, average_result], axis=0)

    # Specify region type as either 'core' or 'periphery'
    df_images3_inner = df_images_day3.copy()
    df_images3_inner['ratio_right_to_left_intensity'] = df_images3_inner['inner_right_to_left_ratio_intensity']
    df_images3_inner['Region type'] = 'Core'

    df_images3_outer = df_images_day3.copy()
    df_images3_outer['ratio_right_to_left_intensity'] = df_images3_outer['outer_right_to_left_ratio_intensity']
    df_images3_outer['Region type'] = 'Periphery'
    df_day3 = pd.concat([df_images3_inner, df_images3_outer])
    print('-----------------------', 'DONE', '--------------------')


    print('-----------------------', 'Figure 4', '--------------------')
    create_figure4(df_day3, output_dir)
    figure_4_compute_pvalues(df_day3)
    print('-----------------------', 'DONE', '--------------------')

    print('-----------------------', 'Figure 5', '--------------------')
    create_figure5(df_images_day3_before_grouping, output_dir)
    #create_scatter(df_images_day3_before_grouping)
    print('-----------------------', 'DONE', '--------------------')

    print('-----------------------', 'Figure 6', '--------------------')
    df_images_halfs = create_images_df(input_folder=INPUT_FOLDER, wanted_distances=[0.5, 3.0])  # 3.0 is control
    create_figure6(df_images_halfs, output_dir)
    plt.close('all')
    print('-----------------------', 'DONE', '--------------------')