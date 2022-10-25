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

    for iteration, s in enumerate(sequences):
        fis = True
        while len(s) > 0:
            # --------------- COMPUTE ---------------
            # compute next percept
            p, units = cm.compute(s, first_in_seq=fis)
            fis = False
            # update s
            s = s[len(p.strip().split(" ")):]
        cm.compute_last()
        # --------------- GENERATE ---------------
        if iteration % 5 == 1:
            cm.tps_units.normalize()
            results[iteration] = dict()
            results[iteration]["generated"] = cm.tps_units.generate_new_seqs(rng, min_len=100)
            im = dict(sorted([(x, y["weight"]) for x, y in cm.pars.mem.items()],
                             key=lambda it: it[1], reverse=True))
            results[iteration]["mem"] = im
