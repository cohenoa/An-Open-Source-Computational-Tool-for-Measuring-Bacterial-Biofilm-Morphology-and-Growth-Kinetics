import math
import os
import re

import cv2
import imutils
import numpy as np
import pandas as pd

from macros import DF_PARAMS


def parse_image_filename(image_file):
    pattern = '^Plate (\d+) CHX (\d+(\.\d)*) cm day (\d+) GFP_RGB_GFP_(\d+)_(\d+).tif$'

    res = re.search(pattern, image_file)
    assert (res is not None)

    plate = int(res.group(1))
    distance = float(res.group(2))
    day = int(res.group(4))
    micrometers_in_pixels = float(res.group(5) + '.' + res.group(6))

    return plate, distance, day, micrometers_in_pixels

def read_image(img_file: str):
  assert(os.path.exists(img_file))
  image = cv2.imread(img_file)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_3D = image.copy()
  image = image[:, :, 1]
  image_uint = image.copy()
  image = image / 255
  image_raw = image.copy()
  image_normalized = image.copy()
  image_normalized = cv2.normalize(image_normalized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  # cv2 epsilon number clipping
  image_normalized[image_normalized < 0] = 0
  return image_normalized, image_raw, image_uint, image_3D


def get_all_contours(image, min_thresh, to_erode=True, kernel_size=5, iterations=3):
  if to_erode:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image = cv2.erode(image, kernel, iterations=iterations)

  mask = cv2.inRange(image, min_thresh, 1)
  contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(contours)

  return contours


def fetch_nth_contour(contours, n=1):
    n -= 1
    nth_area = -1
    nth_contour = None

    # sort contours by area
    contour_areas = [cv2.contourArea(c) for c in contours]
    contour_areas_sorted = sorted(contour_areas, reverse=True)
    sorted_idx = np.argsort(contour_areas)[::-1]

    nth_area = contour_areas_sorted[n]
    nth_contour = contours[sorted_idx[n]]
    nth_perimeter = cv2.arcLength(nth_contour, True)

    return nth_contour, nth_area, nth_perimeter


def create_centers_df(input_folder):
  # create df of centers coordinates for each image:
  columns=['file_name', 'coordinate1', 'coordinate2']
  df_centers = pd.read_csv(os.path.join(input_folder, 'DAY1.csv'), header=None, names=columns)
  df_centers = pd.concat([df_centers, pd.read_csv(os.path.join(input_folder, 'DAY2.csv'), header=None, names=columns)])
  df_centers = pd.concat([df_centers, pd.read_csv(os.path.join(input_folder, 'DAY3.csv'), header=None, names=columns)])

  df_centers.coordinate1 = df_centers.coordinate1.astype('int64')
  df_centers.coordinate2 = df_centers.coordinate2.astype('int64')

  df_centers['Center'] = df_centers[['coordinate1', 'coordinate2']].apply(tuple, axis=1)
  df_centers = df_centers.drop(labels=['coordinate1', 'coordinate2'], axis=1)
  df_centers = df_centers.set_index('file_name')
  return df_centers


def compute_distance_gradient_image(image, distance, center, micrometers_in_pixels):
    CHX_dist_micrometer = distance * 1E4
    center = (int(center[0]), int(center[1]))
    p_origin = (int(center[0] + CHX_dist_micrometer / micrometers_in_pixels), int(center[1]))
    distance_gradient_image = np.linalg.norm(np.indices((image.shape[0], image.shape[1])).transpose((1,2,0)) -  np.array([p_origin[1],p_origin[0]]), axis=2)
    return distance_gradient_image


def get_ellipse_stats(contour):
  (xc,yc), (d1,d2), angle = cv2.fitEllipse(contour)
  a = max(d1,d2) / 2  # Major radius
  b = min(d1,d2) / 2  # Minor radius
  area = math.pi * a * b
  perimeter = math.pi * ( 3*(a+b) - math.sqrt( (3*a + b) * (a + 3*b) ) )
  return a, b, area, perimeter


def get_params(image_file_name):
  if not image_file_name in DF_PARAMS.index: # for 0.5 cm
    return 0.3, 0.3, 1, 0
  # assert(image_file_name in DF_PARAMS.index), '{} is missing from DF_PARAMS'.format(image_file_name)
  row = DF_PARAMS.loc[image_file_name]
  return row['min_thresh_outer'], row['min_thresh_inner'], row['erode_kernel_size'], row['erode_iterations']


def get_ellipse_radii(image, contour, center, half_type):
    if contour is None:  # e.g., day 1
        return -1, -1

    center_x = int(center[0])
    left_half_width = center_x
    right_half_width = image.shape[1] - left_half_width

    # half_image creation:
    half_image = np.zeros(image.shape, dtype=np.int8)
    half_image = cv2.drawContours(half_image, contour, -1, 1, 3)

    if half_type == 'LEFT':
        half_image[:, center_x:half_image.shape[1]] = 0
        mirror = half_image[:, center_x:0:-1]
        width_to_fill = half_image.shape[1] - center_x
        mirror = mirror[:, :width_to_fill]
        half_image[:, center_x:center_x + mirror.shape[1]] = mirror

    else:
        assert (half_type == 'RIGHT')
        half_image[:, 0:center_x] = 0
        mirror = half_image[:, half_image.shape[1]:center_x:-1]
        offset = mirror.shape[1] - left_half_width
        mirror = mirror[:, offset:]
        half_image[:, center_x - mirror.shape[1]:center_x] = mirror

    arr1, arr2 = np.where(half_image)
    current_contour = np.column_stack((arr2, arr1))
    current_contour_obj = current_contour.reshape((current_contour.shape[0], 1, current_contour.shape[1]))

    # fit ellipse to mirrored image
    try:
        (xc, yc), (horizontal_diameter, vertical_diameter), angle = cv2.fitEllipse(current_contour_obj)
    except:
        return None, None, None

    return horizontal_diameter / 2, vertical_diameter / 2, current_contour_obj


def get_masks(image, outer_contour, inner_contour):
    outer_mask = np.zeros(image.shape)
    outer_mask = cv2.drawContours(outer_mask, [outer_contour], -1, 1, cv2.FILLED)

    inner_mask = np.zeros(image.shape)
    inner_mask = cv2.drawContours(inner_mask, [inner_contour], -1, 1, cv2.FILLED)

    return outer_mask, inner_mask


def fill_df(df_images):
    df_images['distance_from_CHX_map'] = df_images.apply(lambda x: compute_distance_gradient_image(x['NormalizedImage'], x['DistanceFromCHX'], x['Center'], x['MicrometersInPixels']), axis=1)
    df_images['OuterContourArea'] *= df_images['MicrometersInPixels'] * df_images['MicrometersInPixels']
    df_images['OuterContourPerimeter'] *= df_images['MicrometersInPixels']
    df_images['InnerContourArea'] *= df_images['MicrometersInPixels'] * df_images['MicrometersInPixels']
    df_images['InnerContourPerimeter'] *=  df_images['MicrometersInPixels']
    df_images['DistanceFromCHXStr'] = df_images['DistanceFromCHX'].apply(lambda x: str(x))
    df_images.loc[df_images['DistanceFromCHXStr'] == '3.0', 'DistanceFromCHXStr'] = 'Control'

    # Add columns - vertical & horizontal radii for the inner contour (Figure 2)
    df_images['horizontal_left_radius'] = df_images.apply(lambda x: get_ellipse_radii(x['NormalizedImage'], x['OuterContourObj'], x['Center'], 'LEFT')[0], axis=1) #* df_images['MicrometersInPixels']
    df_images['vertical_left_radius'] = df_images.apply(lambda x: get_ellipse_radii(x['NormalizedImage'], x['OuterContourObj'], x['Center'], 'LEFT')[1], axis=1)  #* df_images['MicrometersInPixels']
    df_images['horizontal_right_radius'] = df_images.apply(lambda x: get_ellipse_radii(x['NormalizedImage'], x['OuterContourObj'], x['Center'], 'RIGHT')[0], axis=1) # * df_images['MicrometersInPixels']
    df_images['vertical_right_radius'] = df_images.apply(lambda x: get_ellipse_radii(x['NormalizedImage'], x['OuterContourObj'], x['Center'], 'RIGHT')[1], axis=1)  #* df_images['MicrometersInPixels']
    df_images['vertical_radii_ratio'] = df_images['vertical_left_radius'] / df_images['vertical_right_radius']
    df_images['horizontal_radii_ratio'] = df_images['horizontal_left_radius'] / df_images['horizontal_right_radius']

    # Computing vertical & horizontal radii for the inner contour (Figure 3)
    df_images['inner_horizontal_left_radius'] = df_images.apply(lambda x: get_ellipse_radii(x['NormalizedImage'], x['InnerContourObj'], x['Center'], 'LEFT')[0], axis=1)
    df_images['inner_vertical_left_radius'] =  df_images.apply(lambda x: get_ellipse_radii(x['NormalizedImage'], x['InnerContourObj'], x['Center'], 'LEFT')[1], axis=1)
    df_images['inner_horizontal_right_radius'] = df_images.apply(lambda x: get_ellipse_radii(x['NormalizedImage'], x['InnerContourObj'], x['Center'], 'RIGHT')[0], axis=1)
    df_images['inner_vertical_right_radius'] =  df_images.apply(lambda x: get_ellipse_radii(x['NormalizedImage'], x['InnerContourObj'], x['Center'], 'RIGHT')[1], axis=1)
    df_images['inner_vertical_radii_ratio'] = df_images['inner_vertical_left_radius'] / df_images['inner_vertical_right_radius']
    df_images['inner_horizontal_radii_ratio'] = df_images['inner_horizontal_left_radius'] / df_images['inner_horizontal_right_radius']


import numpy as np

def cv2_clipped_zoom(img, zoom_factor=0):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.
    """
    if zoom_factor == 0:
        return img

    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result




