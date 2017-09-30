# compute_detector.py ---
#
# Filename: compute_detector.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Thu Feb 18 19:48:09 2016 (+0100)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), EPFL Computer Vision Lab.

# Code:


from __future__ import print_function

import os
import sys
from .extractors.detector import compute_keypoints


DEFAULT_MODEL_DIR = '' # TODO: put model dir here


# ------------------------------------------
# Main routine
if __name__ == '__main__':

    total_time = 0

    # ------------------------------------------------------------------------
    # Read arguments

    if len(sys.argv) < 6 or len(sys.argv) > 9:
        raise RuntimeError('USAGE: python compute_detector.py '
                           '<config_file> '
                           '<image_file> <output_file> '
                           '<bSavePng> <bUseTheano> '
                           '<bPrintTime/optional> '
                           '<model_dir/optional> '
                           '<num_keypoint/optional> ')
    config_file = sys.argv[1]
    input_image_name = sys.argv[2]
    output_file = sys.argv[3]

    compute_keypoints(config_file, input_image_name, output_file, b_save_png=True)

#
# compute_detector.py ends here
