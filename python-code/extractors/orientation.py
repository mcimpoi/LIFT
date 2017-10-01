import os
import time

import numpy as np

from six.moves import xrange
from Utils.custom_types import paramGroup, paramStruct, pathConfig
from Utils.dataset_tools import test as data_module
from Utils.dump_tools import loadh5
from Utils.kp_tools import IDX_ANGLE, saveKpListToTxt, update_affine
from Utils.solvers import Test

DEFAULT_MODEL_DIR = os.path.expanduser('~/Workspace/research/forks/LIFT/models/picc-best/')


def compute_orientations(config_file, image_file_name, kp_file_name, output_file):
    bDumpPatch = False
    bPrintTime = False

    # ------------------------------------------------------------------------
    # Setup and load parameters
    param = paramStruct()
    param.loadParam(config_file, verbose=True)
    pathconf = pathConfig()
    pathconf.setupTrain(param, 0)

    # Use model dir if given
    model_dir = DEFAULT_MODEL_DIR
    if model_dir is not None:
        pathconf.result = model_dir

    # -------------------------------------------------------------------------
    # Modify the network so that we bypass the keypoint part and the
    # descriptor part.
    param.model.sDetector = 'bypass'
    # This ensures that you don't create unecessary scale space
    param.model.fScaleList = np.array([1.0])
    param.patch.fMaxScale = np.max(param.model.fScaleList)
    # this ensures that you don't over eliminate features at boundaries
    param.model.nPatchSize = int(np.round(param.model.nDescInputSize) *
                                 np.sqrt(2))
    param.patch.fRatioScale = (float(param.patch.nPatchSize) /
                               float(param.model.nDescInputSize)) * 6.0
    param.model.sDescriptor = 'bypass'

    # add the mean and std of the learned model to the param
    mean_std_file = pathconf.result + 'mean_std.h5'
    mean_std_dict = loadh5(mean_std_file)
    param.online = paramGroup()
    setattr(param.online, 'mean_x', mean_std_dict['mean_x'])
    setattr(param.online, 'std_x', mean_std_dict['std_x'])

    # -------------------------------------------------------------------------
    # Load data in the test format
    test_data_in = data_module.data_obj(param, image_file_name, kp_file_name)

    # -------------------------------------------------------------------------
    # Test using the test function
    start_time = time.clock()
    _, oris, compile_time = Test(
        pathconf, param, test_data_in, test_mode="ori")
    end_time = time.clock()
    compute_time = (end_time - start_time) * 1000.0 - compile_time
    print("Time taken to compile is {} ms".format(
        compile_time
    ))
    print("Time taken to compute is {} ms".format(
        compute_time
    ))
    if bPrintTime:
        # Also print to a file by appending
        with open("../timing-code/timing.txt", "a") as timing_file:
            print("------ Orientation Timing ------\n"
                  "Computation time for {} keypoints is {} ms\n".format(
                    test_data_in.x.shape[0], compute_time), file=timing_file)

    # update keypoints and save as new
    kps = test_data_in.coords
    for idxkp in xrange(len(kps)):
        kps[idxkp][IDX_ANGLE] = oris[idxkp] * 180.0 / np.pi % 360.0
        kps[idxkp] = update_affine(kps[idxkp])

    # save as new keypoints
    saveKpListToTxt(kps, kp_file_name, output_file)
