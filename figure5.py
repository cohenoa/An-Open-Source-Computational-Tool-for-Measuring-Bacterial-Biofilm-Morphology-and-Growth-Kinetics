import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from scipy.stats import linregress
# import statannot
from sklearn import linear_model

from figure4 import get_masks
from macros import *

EDGE_WIDTH = 20
import cv2
import numpy as np


def get_pixel_distances_vs_intensities(row):
  image = row['RawImage']
  distances_map = row['distance_from_CHX_map']
  outer_contour = row['OuterContourObj']
  inner_contour = row['InnerContourObj']
  center = row['Center']
  micrometers_in_pixel = row['MicrometersInPixels']

  outer_mask, inner_mask = get_masks(image, outer_contour, inner_contour)
  outer_ring_only_mask = cv2.bitwise_and(outer_mask,  cv2.bitwise_not(inner_mask))

  # taking only EDGE_WIDTH pixels for edge pixels
  contour_only_mask = cv2.drawContours(np.zeros(image.shape), [row['OuterContourObj']], -1, 1, EDGE_WIDTH)
  leading_edge_mask = cv2.bitwise_and(contour_only_mask, outer_ring_only_mask)
  leading_edge_mask[:, 0:int(center[0])] = 0

  intensities = image[leading_edge_mask == 1].flatten()
  distances = distances_map[leading_edge_mask == 1].flatten()

  # convert from pixel distances to micrometer
  distances = distances * micrometers_in_pixel

  linear_regression = linear_model.LinearRegression()
  linear_regression.fit(distances.reshape(-1, 1), intensities)

  return distances, intensities, linear_regression.coef_


def create_figure5(df_day3, output_dir):
  image_1cm_filename = 'Plate 2 CHX 1 cm day 3 GFP_RGB_GFP_22_38.tif'
  image_2cm_filename = 'Plate 10 CHX 2 cm day 3 GFP_RGB_GFP_22_38.tif'

  distances1, intensities1, slope1 = get_pixel_distances_vs_intensities(df_day3.loc[image_2cm_filename])
  distances2, intensities2, slope2 = get_pixel_distances_vs_intensities(df_day3.loc[image_1cm_filename])

  regr1 = linear_model.LinearRegression()
  regr1.fit(distances1.reshape(-1, 1), intensities1)
  intensities1_predict = regr1.predict(distances1.reshape(-1, 1))

  regr2 = linear_model.LinearRegression()
  regr2.fit(distances2.reshape(-1, 1), intensities2)
  intensities2_predict = regr2.predict(distances2.reshape(-1, 1))

  fig4b = plt.figure(constrained_layout=True, figsize=(7, 7))
  spec4b = gridspec.GridSpec(ncols=2, nrows=3, figure=fig4b)

  # Top left image
  fig4b.add_subplot(spec4b[0, 0])
  image_1cm = df_day3.loc[image_1cm_filename]['RawImage']
  center_1cm = df_day3.loc[image_1cm_filename]['Center']
  contour_only_mask = cv2.drawContours(np.zeros((image_1cm.shape[0],image_1cm.shape[1], 3)), \
                                       [df_day3.loc[image_1cm_filename]['OuterContourObj']], -1, (255, 0, 0), EDGE_WIDTH)
  contour_only_mask[:, 0:int(center_1cm[0]), :] = 0

  plt.imshow(image_1cm, alpha=1)
  plt.colorbar()
  plt.imshow(contour_only_mask, alpha=0.3)
  plt.axis('off')

  # Top right image
  fig4b.add_subplot(spec4b[0, 1])
  image_2cm = df_day3.loc[image_2cm_filename]['RawImage']

  contour_only_mask = cv2.drawContours(np.zeros((image_2cm.shape[0], image_2cm.shape[1], 3)), \
                                       [df_day3.loc[image_2cm_filename]['OuterContourObj']], -1, (255, 0, 0), 15)
  contour_only_mask[:, 0:int(center_1cm[0]), :] = 0
  plt.imshow(image_2cm, alpha=1)
  plt.colorbar()
  plt.imshow(contour_only_mask, alpha=0.3)
  plt.axis('off')

  # Bottom panel
  fig4b.add_subplot(spec4b[1:3, 0:2])
  plt.plot(distances1, intensities1, '.', alpha=0.5)
  plt.plot(distances1, intensities1_predict, color="darkblue", linewidth=3)
  # print(linregress(distances1, intensities1_predict))
  plt.plot(distances2, intensities2, 'r.', alpha=0.5)
  plt.plot(distances2, intensities2_predict, color="darkred", linewidth=3)
  # print(linregress(distances2, intensities2_predict))
  plt.ylabel('Pixel Intensity', fontsize=AXIS_FONT_SIZE)
  plt.xlabel('Distance from CHX ($\mu m$)', fontsize=AXIS_FONT_SIZE)
  plt.tick_params(labelsize=AXIS_TICK_SIZE)
  plt.savefig(os.path.join(output_dir, 'Figure_5.png'))


def create_scatter(df_day3):
  # add the columns distances, intensities for each row:
  df_day3['slope'] = df_day3.apply(lambda x: get_pixel_distances_vs_intensities(x)[2][0], axis=1)
  df_day3['distances'] = df_day3.apply(lambda x: get_pixel_distances_vs_intensities(x)[0], axis=1)
  df_day3['intensities'] = df_day3.apply(lambda x: get_pixel_distances_vs_intensities(x)[1], axis=1)

  slopes_1cm = df_day3[df_day3['DistanceFromCHX'] == 1.0]['slope']
  slopes_15cm = df_day3[df_day3['DistanceFromCHX'] == 1.5]['slope']
  slopes_2cm = df_day3[df_day3['DistanceFromCHX'] == 2.0]['slope']
  slopes_control = df_day3[df_day3['DistanceFromCHX'] == 3.0]['slope']

  print('ttest: 1 cm vs. control {:.3}'.format(scipy.stats.ttest_ind(slopes_1cm, slopes_control).pvalue))
  print('ttest: 1.5 cm vs. control {:.3}'.format(scipy.stats.ttest_ind(slopes_15cm, slopes_control).pvalue))
  print('ttest: 2 cm vs. control {:.3}'.format(scipy.stats.ttest_ind(slopes_2cm, slopes_control).pvalue))

  df_day3 = df_day3[(df_day3['DistanceFromCHX'] == 1.0) | (df_day3['DistanceFromCHX'] == 3.0)]
  sns.set_palette(['darkred', 'darkblue'])
  ax = sns.boxplot(x='DistanceFromCHX', y='slope', data=df_day3, linewidth=1.5, width=0.6)
  ax.set_xticklabels(['1.0', 'Control'])
  ax.set_ylabel("Slope", fontsize=AXIS_FONT_SIZE)
  ax.set_xlabel("Distance from CHX (cm)", fontsize=AXIS_FONT_SIZE)
  plt.show()
