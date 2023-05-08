import sys
import argparse
import numpy as np
import topy
from config import random_config,cantilever_beam_config,continuous_beam_config,simply_supported_beam_config,random_config_64_128,cantilever_beam_config_64_128,continuous_beam_config_64_128,simply_supported_beam_config_64_128
import os
import time

def optimize(t):
    t0 = time.time()
    d = []
    for i in range(t.numiter):
        # Main operations
        t.fea()
        t.sens_analysis()
        t.filter_sens_sigmund()
        t.update_desvars_oc()
        d.append(t.desvars)
    t1 = time.time()
    print(t1 - t0)
    return np.array(d)

def main(dir_to_save, num_samples):
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)
        print('Directory "{}" created'.format(dir_to_save))

    print('Generating the dataset...')
    samples_done = 0
    while samples_done < num_samples:
        try:
            topology = topy.Topology(config=random_config())
            topology.set_top_params()
            sample = optimize(topology)
            path = os.path.join(dir_to_save, str(samples_done))
            np.savez_compressed(path, sample)
            samples_done += 1
        except BaseException:  # For the incorrect constraints
            pass
        print('\rDone: {}/{}'.format(samples_done, num_samples))
        sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='test')
    parser.add_argument('--num', type=int, default=10)
    args = parser.parse_args()
    main(args.dir, args.num)