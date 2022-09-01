import argparse
import os
import pandas as pd
from pathlib import Path

from macros import IMAGES_SUBFOLDER, CENTERS_SUBFOLDER


def is_valid_inputdir(path_str: str):
    """
    Check whether the given path_str is valid: exists and contains the required files\directories.
    Args:
        path_str: Query input directory.

    Returns: True iff the path is valid.

    """

    if not os.path.exists(path_str):
        print('ERROR: Input folder must exist. You entered: [{}]'.format(path_str))
        return False

    # Check that the input folder contains the required sub folders: Centers, Images
    if not os.path.exists(os.path.join(path_str, IMAGES_SUBFOLDER)):
        print('ERROR: Input folder must contain a sub-folder named "{}".'
              ' You entered: "{}"'.format(IMAGES_SUBFOLDER, path_str))
        return False

    if not os.path.exists(os.path.join(path_str, CENTERS_SUBFOLDER)):
        print('ERROR: Input folder must contain a sub-folder named "{}". '
              'You entered: "{}"'.format(CENTERS_SUBFOLDER, path_str))
        return False

    return True


def handle_input_arguments(args):
    # asking for two input arguments:
    # (a) Input folder, containing the images (e.g., tif fileS) and matching centers (csv files)
    # (b) Output folder for generating the images presented in the manuscript.
    parser = argparse.ArgumentParser(description='Running biofilm image processing analysis.')
    parser.add_argument('-i', '--input_folder', required=True, help='Path of input folder.')

    parser.add_argument('-o', '--output_folder',  required=True, help='Path of output folder.')
    results = parser.parse_args(args)
    input_dir = results.input_folder
    output_dir = results.output_folder

    if not is_valid_inputdir(input_dir):
        exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return input_dir, output_dir


def organize_paths(base_path):
    images_data_folder = os.path.join(base_path, IMAGES_SUBFOLDER)
    centers_data_folder = os.path.join(base_path, CENTERS_SUBFOLDER)
    df_params = pd.read_csv(os.path.join(base_path, 'thresh_params.csv'), index_col=0)
    return images_data_folder, centers_data_folder, df_params

