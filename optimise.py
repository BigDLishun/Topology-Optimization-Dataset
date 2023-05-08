from testsampler import random_config,cantilever_beam_config,simply_supported_beam_config,continuous_beam_config,random_config_64_128,cantilever_beam_config_64_128,continuous_beam_config_64_128,simply_supported_beam_config_64_128
import topy
import time
def optimise(config):
    # Set up ToPy:
    t0=time.time()
    t = topy.Topology(config)
   # t.load_tpd_file(config)
    t.set_top_params()
    topy.optimise(t)
    t1=time.time()
    print (t1-t0)


if __name__ == '__main__':
    config=cantilever_beam_config()
    optimise(config)

