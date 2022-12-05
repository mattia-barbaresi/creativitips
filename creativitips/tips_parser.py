import json
import os
import numpy as np
from creativitips import utils
from creativitips import const
from creativitips.CTPs.computation import Computation


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    # maintaining INTERFERENCES/FORGETS separation by a factor of 10
    t_mem = 1.0
    interf = 0.00001
    tps_order = 1
    # method = [met_pars]
    # met: AVG, FTP, FTPAVG, CT, BRENT..
    # pars: W = with, N=No, F=forgetting, I=interference
    methods = ["BRENT_NFWI"]
    dir_in = 'data/CHILDES_converted/'

    for tps_met in methods:
        root_dir = "data/CHILDES_tipssss_" + tps_met + "/"
        # init
        rng = np.random.default_rng(const.RND_SEED)
        os.makedirs(root_dir, exist_ok=True)
        with open(root_dir + "params.json", "w") as of:
            json.dump({
                "method": tps_met,
                "rnd": const.RND_SEED,
                "mem thresh": t_mem,
                "interference": interf,
                "weight": const.WEIGHT,
                "lens": const.ULENS,
                "tps_order": tps_order,
                "parser_decay_rate": const.STM_DECAY_RATE,
                "tps_decay_rate": const.LTM_DECAY_RATE,
            }, of)
        # read files
        for subdir, dirs, files in os.walk(dir_in):
            for fn in files:
                if '.capp' in fn:
                    print("processing {} series ...".format(subdir))
                    fi_dir = subdir.replace(dir_in, root_dir)
                    os.makedirs(fi_dir, exist_ok=True)
                    # clean sequences
                    sequences = []
                    with open(subdir + "/" + fn, "r", encoding='utf-8') as fp:
                        for line in fp.readlines():
                            if '*AGEIS:' in line:
                                continue
                            utter = line.strip().strip("!?.").split()
                            # if list has less than 3 elements, it is empty b/c
                            # auto-cleaning removed a non-speech sound, etc.
                            if len(utter) < 2:
                                continue
                            sequences.append(utter[1:])

                    # init module for computation
                    cm = Computation(rng, order=tps_order, interference=interf, unit_len=const.ULENS, method=tps_met)
                    # compute series
                    cm.compute(sequences)
                    # save shallow parsing results
                    with open(fi_dir + "/" + fn.split('.capp')[0] + '.shpartips', "w", encoding='utf-8') as fp:
                        for ln in cm.shallow_parsing:
                            fp.write(ln + "\n")
