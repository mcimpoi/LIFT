from os import listdir
from os.path import join, expanduser
from extractors.detector import compute_keypoints

PROJECT_DIR = expanduser('~/Workspace/research/forks/LIFT')
CONFIG_FILE = join(PROJECT_DIR, 'models', 'configs', 'picc-finetune-nopair.config')

INPUT_DIR = expanduser('~/Downloads/query/iphone7/')
OUTPUT_DIR = expanduser('~/Downloads/query/iphone7/results/')

if __name__ == '__main__':
    image_files = sorted([f for f in listdir(INPUT_DIR) if f.endswith('.JPG')])
    cnt = 0
    for img in image_files:
        print(img)
        compute_keypoints(CONFIG_FILE, join(INPUT_DIR, img), join(OUTPUT_DIR, img + '_kp.txt'), b_save_png=False)
        cnt += 1
        if cnt > 3:
            break