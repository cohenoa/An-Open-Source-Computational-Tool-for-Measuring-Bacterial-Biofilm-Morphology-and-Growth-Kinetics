import pandas as pd
import cv2
import os
import numpy as np
from utils import get_all_contours, read_image, fetch_nth_contour
import matplotlib.pyplot as plt
from matplotlib import gridspec

if __name__ == '__main__':
    df_file = 'df_images.pickle'
    use_existing_df = False
    df_images = pd.read_pickle(df_file)

    for day in [1, 2, 3]:
        print(day)
        a = df_images[(df_images.DistanceFromCHX == 3.0) & (df_images.Day == day)]['horizontal_radii_ratio'].mean()
        print(a)


    for day in [1, 2, 3]:
        print(day)
        a = df_images[(df_images.DistanceFromCHX == 3.0) & (df_images.Day == day)]['vertical_radii_ratio'].mean()
        print(a)

    print(len(df_images))
    exit(0)
    image_file_path = '/Dataset/Images//Day3//Plate 4 CHX 1 cm day 3 GFP_RGB_GFP_22_38.tif'
    norm_image, image_raw, image_uint, image_3D = read_image(image_file_path)
    min_thresh_outer, min_thresh_inner, erode_kernel_size, erode_iterations = 0.1,0.34,5,3

    contours_outer = get_all_contours(norm_image, min_thresh_outer)
    outer_contour_obj, outer_contour_area, outer_contour_perimeter = \
        fetch_nth_contour(contours_outer, 1)
    image_outer_mask = cv2.drawContours(np.zeros(norm_image.shape), [outer_contour_obj], -1, 1, -1)

    current_image = norm_image
    contours_inner = get_all_contours(current_image, min_thresh_inner, True, int(erode_kernel_size),
                                      int(erode_iterations))

    inner_contour_obj, inner_contour_area, inner_contour_perimeter = \
                            fetch_nth_contour(contours_inner, 2)
    image_inner_mask = cv2.drawContours(np.zeros(norm_image.shape), [inner_contour_obj], -1, 1, -1)
    figure = plt.figure(constrained_layout=True, figsize=(8, 4))
    spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=figure)

    whole_colony_mask = np.zeros(norm_image.shape)
    whole_colony_mask = cv2.drawContours(whole_colony_mask, [outer_contour_obj], -1, (1, 1, 1), cv2.FILLED)
    image_with_background = np.zeros((norm_image.shape[0], norm_image.shape[1], 3))
    image_with_background[whole_colony_mask == 1, 1] = norm_image[whole_colony_mask == 1]
    image_with_background = cv2.drawContours(image_with_background, [outer_contour_obj], -1, (1, 0, 0), 3)
    image_with_background = cv2.drawContours(image_with_background, [inner_contour_obj], -1, (1, 0, 0), 3)


    figure.add_subplot(spec2[0, 0])
    plt.imshow(image_with_background)
    plt.axis('off')

    figure.add_subplot(spec2[0, 1])
    plt.imshow(image_outer_mask, cmap='gray')
    plt.axis('off')

    figure.add_subplot(spec2[0, 2])
    plt.imshow(image_inner_mask, cmap='gray')
    plt.axis('off')

    plt.show()
    print('done')


# if __name__ == '__main__':
#
#
#     # image_file_path = 'C://Users//noa//PycharmProjects//BacillusCHX//Review//OtherImage' \
#     #         '//Vibrio fischeri Biofilm Formation Prevented by a Trio of Regulators//vibrio1.png'
#     base_path = 'C://Users//noa//PycharmProjects//BacillusCHX//Review//OtherImage//EColi7//'
#     for i in [1, 2] :#range(7, 10):
#         image_file_path = os.path.join(base_path, 'Capture{}.PNG'.format(i))
#         out_path = os.path.join(base_path, 'Outputs', 'Capture_{}'.format(i))
#         if not os.path.exists(out_path):
#             os.makedirs(out_path)
#         # min_thresh_inner = 0.3
#         for min_thresh_outer in [0.1]:# np.arange(0.1, 0.5, 0.1): # np.arange(0.05, 0.3, 0.05)
#             for min_thresh_inner in np.arange(0.7, 1, 0.05): # [0.7]
#                 for erode_kernel_size in range(1,5):
#                     for erode_iterations in range(0, 5):
#                         out_file = os.path.join(out_path, 'outer{:.4f}_inner{:.3f}_kernel{}_iter{}.jpg'.
#                                                  format(min_thresh_outer, min_thresh_inner, erode_kernel_size, erode_iterations))
#
#                         if os.path.exists(out_file):
#                             continue
#                         image_normalized, image_raw, image_uint, image_3D = read_image(image_file_path)
#                         contours_outer = get_all_contours(image_normalized, min_thresh_outer)
#                         outer_contour_obj, outer_contour_area, outer_contour_perimeter = \
#                             fetch_nth_contour(contours_outer, 1)
#                         # Create outer mask
#                         image_outer_mask = cv2.drawContours(np.zeros(image_normalized.shape), [outer_contour_obj], -1, 1, -1)
#
#                         # Compute inner contour
#                         # erode_kernel_size = 5
#                         # erode_iterations = 3
#                         # inner_contour_obj, inner_contour_area, inner_contour_perimeter, image_contrast = None, None, None, None
#
#                         # # DEBUG
#                         # kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
#                         # image_erode = cv2.erode(image_normalized, kernel, iterations=erode_iterations)
#                         # plt.imshow(image_erode)
#                         # plt.axis('off')
#                         # plt.show()
#                         # # DEBUG
#
#                         contours_inner = get_all_contours(image_normalized, min_thresh_inner, True, int(erode_kernel_size),
#                                                           int(erode_iterations))
#
#                         try:
#                             inner_contour_obj, inner_contour_area, inner_contour_perimeter = \
#                                 fetch_nth_contour(contours_inner, 2)  # NOTE! this is 2 in our code
#                         except:
#                             # for thresholds for which no contours are found
#                             print('EXCEPTION')
#                             continue
#
#                         image = image_normalized.copy()
#                         image = np.dstack((image, image, image))
#                         image = cv2.drawContours(image, [outer_contour_obj], -1, (255, 255, 0), 2)
#                         if inner_contour_obj is not None:
#                             image = cv2.drawContours(image, [inner_contour_obj], -1,  (255, 0, 0), 2)
#                         plt.title('{} erode_kernel_size={} erode_iterations={}'.
#                                   format(min_thresh_inner, erode_kernel_size, erode_iterations))
#                         plt.imshow(image)
#                         plt.axis('off')
#
#
#                         plt.savefig(out_file)
#                 # plt.show()