from datetime import datetime
import numpy as np
import const
import utils
from computation import Computation

np.set_printoptions(linewidth=np.inf)
rng = np.random.default_rng(const.RND_SEED)


if __name__ == "__main__":
    interf = 0.005
    forget = const.FORGETTING
    t_mem = 1.0
    tps_order = 2
    tps_method = "CT"
    fn = ""

    # --------------- INPUT ---------------
    results = dict()
    sequences = utils.read_sequences(rng, fn)
    # read percepts using parser function
    start_time = datetime.now()

    # init module for computation
    cm = Computation(rng, order=tps_order, weight=const.WEIGHT, interference=interf, mem_thres=t_mem, unit_len=const.ULENS, method=tps_method)

    # --------------- COMPUTE ---------------
    # compute next percept
    cm.compute(sequences)

