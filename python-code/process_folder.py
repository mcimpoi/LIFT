from os import listdir
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


#INPUT_DIR = '/home/mircea/Workspace/research/at_geometricverification/data'
#OUTPUT_DIR = '/home/mircea/Workspace/research/at_geometricverification/data/descr'

if __name__ == '__main__':
    image_files = sorted([f for f in listdir(INPUT_DIR) if f.endswith('.JPG')])
    cnt = 0
    for img in image_files:
        print(img)
        image_file_name = join(INPUT_DIR, img)
        kp_file_name = join(OUTPUT_DIR, img + '_kp.txt')
        ori_file_name = join(OUTPUT_DIR, img + '_ori.txt')
        desc_file = join(OUTPUT_DIR, img + '_desc.mat')
        compute_keypoints(CONFIG_FILE, image_file_name, kp_file_name, b_save_png=False,
                          num_keypoint=3000)
        compute_orientations(CONFIG_FILE, image_file_name, kp_file_name, ori_file_name)
        compute_descriptors(CONFIG_FILE, image_file_name, ori_file_name, desc_file)
        cnt += 1
