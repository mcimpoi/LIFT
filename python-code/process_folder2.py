from glob import glob
import sys
from os.path import join, expanduser
from extractors.detector import compute_keypoints
from extractors.orientation import compute_orientations
from extractors.descriptor import compute_descriptors

PROJECT_DIR = expanduser('~/Workspace/research/forks/LIFT')
CONFIG_FILE = join(PROJECT_DIR, 'models', 'configs', 'picc-finetune-nopair.config')

INPUT_DIR = expanduser('~/Downloads/query/iphone7/')
OUTPUT_DIR = expanduser('~/Downloads/query/iphone7/results/')

INPUT_DIR = expanduser('/local/data/mircea/query/iphone7/')
OUTPUT_DIR = expanduser('/local/data/mircea/query/iphone7/results/')


# INPUT_DIR = '/home/mircea/Workspace/research/at_geometricverification/data'
# OUTPUT_DIR = '/home/mircea/Workspace/research/at_geometricverification/data/descr'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        INPUT_DIR = sys.argv[1]
    else:
        INPUT_DIR = '/local/data/mircea/cutouts/DUC1/'
    image_files = sorted([f for f in glob(INPUT_DIR + '/*/*.jpg')])
    cnt = 0
    for img in image_files:
        kp_file_name = img + '_lift_kp.txt'
        ori_file_name = img + '_lift_ori.txt'
        desc_file = img + '_lift_desc.mat'
        compute_keypoints(CONFIG_FILE, img, kp_file_name, b_save_png=False,
                          num_keypoint=3000)
        compute_orientations(CONFIG_FILE, img, kp_file_name, ori_file_name)
        compute_descriptors(CONFIG_FILE, img, ori_file_name, desc_file)
        cnt += 1
