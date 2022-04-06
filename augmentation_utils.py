import enum

import cv2
import numpy as np
import pandas as pd

from utils import get_masks


def get_intensities_vertical(image, outer_contour, inner_contour, center):
    # inner core
    outer_mask, inner_mask = get_masks(image, outer_contour, inner_contour)
    inner_image = cv2.bitwise_and(image, inner_mask)

    left_inner_mask = inner_mask[:, 0:int(center[0])]
    left_image = inner_image[:, 0:int(center[0])]
    mean_inner_left_intensity = np.mean(left_image[np.where(left_inner_mask == 1)])
    left_inner_mask_whole = np.zeros(image.shape)
    left_inner_mask_whole[:, 0:int(center[0])] = left_inner_mask

    right_inner_mask = inner_mask[:, int(center[0]):]
    right_image = inner_image[:, int(center[0]):]
    mean_inner_right_intensity = np.mean(right_image[np.where(right_inner_mask == 1)])
    right_inner_mask_whole  = np.zeros(image.shape)
    right_inner_mask_whole[:, int(center[0]):] = right_inner_mask

    # outer periphery
    outer_only_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
    outer_image = cv2.bitwise_and(image, outer_only_mask)

    left_outer_mask = outer_only_mask[:, 0:int(center[0])]
    left_outer_image = outer_image[:, 0:int(center[0])]
    mean_outer_left_intensity = np.mean(left_outer_image[np.where(left_outer_mask == 1)])
    left_outer_mask_whole = np.zeros(image.shape)
    left_outer_mask_whole[:, 0:int(center[0])] = left_outer_mask

    right_outer_mask = outer_only_mask[:, int(center[0]):]
    right_outer_image = outer_image[:, int(center[0]):]
    mean_outer_right_intensity = np.mean(right_outer_image[np.where(right_outer_mask == 1)])
    right_outer_mask_whole =  np.zeros(image.shape)
    right_outer_mask_whole[:, int(center[0]):] = right_outer_mask

    return mean_outer_left_intensity, \
           mean_outer_right_intensity,\
           mean_inner_left_intensity, \
           mean_inner_right_intensity, \
           left_outer_mask_whole, \
           right_outer_mask_whole,\
           left_inner_mask_whole,\
           right_inner_mask_whole


def get_intensities_horizontal(image, outer_contour, inner_contour, center):
    # inner core
    outer_mask, inner_mask = get_masks(image, outer_contour, inner_contour)
    inner_image = cv2.bitwise_and(image, inner_mask)

    top_inner_mask = inner_mask[0:int(center[1]), :]
    left_image = inner_image[0:int(center[1]), :]
    mean_inner_left_intensity = np.mean(left_image[np.where(top_inner_mask == 1)])
    top_inner_mask_whole = np.zeros(image.shape)
    top_inner_mask_whole[0:int(center[1]), :] = top_inner_mask

    bottom_inner_mask = inner_mask[int(center[1]):, :]
    right_image = inner_image[int(center[1]):, :]
    mean_inner_right_intensity = np.mean(right_image[np.where(bottom_inner_mask == 1)])
    bottom_inner_mask_whole = np.zeros(image.shape)
    bottom_inner_mask_whole[int(center[1]):, :] = bottom_inner_mask

    # outer periphery
    outer_only_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
    outer_image = cv2.bitwise_and(image, outer_only_mask)

    top_outer_mask = outer_only_mask[0:int(center[1]), :]
    left_outer_image = outer_image[0:int(center[1]), :]
    mean_outer_left_intensity = np.mean(left_outer_image[np.where(top_outer_mask == 1)])
    top_outer_mask_whole = np.zeros(image.shape)
    top_outer_mask_whole[0:int(center[1]), :] = top_outer_mask

    bottom_outer_mask = outer_only_mask[int(center[1]):, :]
    right_outer_image = outer_image[int(center[1]):, :]
    mean_outer_right_intensity = np.mean(right_outer_image[np.where(bottom_outer_mask == 1)])
    bottom_outer_mask_whole = np.zeros(image.shape)
    bottom_outer_mask_whole[int(center[1]):, :] = bottom_outer_mask

    return mean_outer_left_intensity,\
           mean_outer_right_intensity,\
           mean_inner_left_intensity, \
           mean_inner_right_intensity, \
           top_outer_mask_whole,\
           bottom_outer_mask_whole,\
           top_inner_mask_whole,\
           bottom_inner_mask_whole


def get_intensities_cross(image, outer_contour, inner_contour, center):
    _, _, _, _, left_outer_mask, right_outer_mask, left_inner_mask, right_inner_mask =\
        get_intensities_vertical(image, outer_contour, inner_contour, center)

    _, _, _, _, top_outer_mask, bottom_outer_mask, top_inner_mask, bottom_inner_mask = \
        get_intensities_horizontal(image, outer_contour, inner_contour, center)

    # Outer quadrants:
    mask_top_left_outer = cv2.bitwise_and(left_outer_mask, top_outer_mask)
    mask_top_right_outer = cv2.bitwise_and(right_outer_mask, top_outer_mask)
    mask_bottom_left_outer = cv2.bitwise_and(left_outer_mask, bottom_outer_mask)
    mask_bottom_right_outer = cv2.bitwise_and(right_outer_mask, bottom_outer_mask)

    mean_outer_cross1_intensity = np.mean(image[np.where((mask_top_left_outer == 1) | (mask_bottom_right_outer == 1))])
    mean_outer_cross2_intensity = np.mean(image[np.where((mask_top_right_outer == 1) | (mask_bottom_left_outer == 1))])

    # Inner quadrants:
    mask_top_left_inner = cv2.bitwise_and(left_inner_mask, top_inner_mask)
    mask_top_right_inner = cv2.bitwise_and(right_inner_mask, top_inner_mask)
    mask_bottom_left_inner = cv2.bitwise_and(left_inner_mask, bottom_inner_mask)
    mask_bottom_right_inner = cv2.bitwise_and(right_inner_mask, bottom_inner_mask)

    mean_inner_cross1_intensity = np.mean(image[np.where((mask_top_left_inner == 1) | (mask_bottom_right_inner == 1))])
    mean_inner_cross2_intensity = np.mean(image[np.where((mask_top_right_inner == 1) | (mask_bottom_left_inner == 1))])

    return mean_outer_cross1_intensity,\
           mean_outer_cross2_intensity,\
           mean_inner_cross1_intensity,\
           mean_inner_cross2_intensity


class Orientation(enum.Enum):
    VERTICAL = 1
    HORIZONTAL = 2
    CROSS = 3


def get_intensities_orientation(image, outer_contour, inner_contour, center, orientation):
    if orientation == Orientation.VERTICAL:
        mean_outer_left_intensity, mean_outer_right_intensity, mean_inner_left_intensity, mean_inner_right_intensity, \
        _, _, _, _ =  get_intensities_vertical(image, outer_contour, inner_contour, center)
    elif orientation == Orientation.HORIZONTAL:
        mean_outer_left_intensity, mean_outer_right_intensity, mean_inner_left_intensity, mean_inner_right_intensity, \
        _, _, _, _ = get_intensities_horizontal(image, outer_contour, inner_contour, center)
    elif orientation == Orientation.CROSS:
        mean_outer_left_intensity, mean_outer_right_intensity, mean_inner_left_intensity, mean_inner_right_intensity\
            = get_intensities_cross(image, outer_contour, inner_contour, center)
    else:
        assert False

    return mean_outer_left_intensity, mean_outer_right_intensity, mean_inner_left_intensity, mean_inner_right_intensity


def create_control_augmentations(df_day3):
    control_distance = 3.0
    # Add a column of orientation
    df_day3['orientation'] = Orientation.VERTICAL

    df_horizontal = df_day3[df_day3['DistanceFromCHX'] == control_distance].copy()
    df_horizontal['orientation'] = Orientation.HORIZONTAL

    df_cross = df_day3[df_day3['DistanceFromCHX'] == control_distance].copy()
    df_cross['orientation'] = Orientation.CROSS

    df_augmented = pd.concat([df_day3, df_horizontal, df_cross], axis=0)
    return df_augmented
