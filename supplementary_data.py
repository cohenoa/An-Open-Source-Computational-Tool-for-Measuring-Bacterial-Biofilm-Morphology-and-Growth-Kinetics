import pandas as pd
import matplotlib.pyplot as plt
import os
import img2pdf
from tqdm import tqdm
import cv2
import numpy as np

from utils import cv2_clipped_zoom


def plot_supp_for_plate(df_plate):
    plate_num = df_plate['Plate'].unique()[0]
    distance_chx = df_plate['DistanceFromCHX'].unique()[0]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, day in enumerate([1, 2, 3]):
        temp = df_plate[df_plate['Day'] == day]

        fig.suptitle('Plate {}, Distance from CHX: {} cm'.format(plate_num, distance_chx))
        image_name = temp.iloc[0].name

        image = df_images.loc[image_name]['NormalizedImage']
        image = np.dstack((image, image, image))
        outer_contour_obj = df_images.loc[image_name]['OuterContourObj']
        image = cv2.drawContours(image, [outer_contour_obj], -1, 1, 3)
        image = cv2.ellipse(image, cv2.fitEllipse(outer_contour_obj), (0, 255, 255), 2)
        plt.imshow(image)

        if day != 1:
            inner_contour_obj = df_images.loc[image_name]['InnerContourObj']
            image = cv2.drawContours(image, [inner_contour_obj], -1, 1, 3)
            image = cv2.ellipse(image, cv2.fitEllipse(inner_contour_obj), (0, 255, 255), 2)

        conversion_rate = df_images.loc[image_name]['MicrometersInPixels']
        scale = (conversion_rate / 22.38)
        print('[{}] conversion rate: {}, scale: {}, shape: [{},{}]'.format(plate_num, conversion_rate, scale, image.shape[0], image.shape[1]))
        image = cv2_clipped_zoom(image, scale)

        axs[i].imshow(image, cmap='gray')
        axs[i].set_title('Day {}'.format(day))
        axs[i].axis('off')


if __name__ == '__main__':
    df_file = 'df_images.pickle'
    use_existing_df = True
    df_images = pd.read_pickle(df_file)

    supp_dir = './SuppImages/'
    if not os.path.exists(supp_dir):
        os.makedirs(supp_dir)

    print('---------- Generating all supp. images -----------')
    all_files = []
    for plate_num in set(df_images['Plate'].values):
        df_plate = df_images[df_images['Plate'] == plate_num].copy()
        distance_chx = df_plate['DistanceFromCHX'].unique()[0]
        plot_supp_for_plate(df_plate)
        out_image_file = os.path.join(supp_dir, 'Plate_{}_CHX_{}.png'.format(plate_num, distance_chx))
        all_files.append(out_image_file)
        plt.savefig(out_image_file)
    print('---------- Done generating all supp. images -----------')

    print('---------- Merging all supp. images to a single PDF file-----------')
    out_pdf_file = os.path.join(supp_dir, 'SuppImages.pdf')
    if os.path.exists(out_pdf_file):
        os.remove(out_pdf_file)

    with open(out_pdf_file, 'wb') as fout:
        files = [os.path.join(supp_dir, f) for f in os.listdir(supp_dir) if f.endswith('.png')]  # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x))
        fout.write(img2pdf.convert(files))
