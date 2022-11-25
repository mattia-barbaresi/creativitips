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
    tps_order = 2
    # method = [met_pars]
    # met: AVG, FTP, MI, CT or BRENT
    # pars: W = with, N=No, F=forgetting, I=interference
    methods = ["AVG_WFWI","FTP_WFWI"]

    root_dir = "data/CHILDES_tips/"
    dir_in = 'data/CHILDES_converted/'
    for tps_met in methods:
        # init
        rng = np.random.default_rng(const.RND_SEED)
        os.makedirs(root_dir, exist_ok=True)
        # read files
        for subdir, dirs, files in os.walk(dir_in):
            for fn in files:
                if '.capp' in fn:
                    print("processing {} series ...".format(fn))
                    fi_dir = subdir.replace(dir_in, root_dir)
                    os.makedirs(fi_dir, exist_ok=True)

                    # clean sequences
                    sequences = []
                    with open(subdir + "/" + fn, "r") as fp:
                        for line in fp.readlines():
                            if '*AGEIS:' in line:
                                continue
                            utter = line.strip().split()
                            # if list has less than 3 elements, it is empty b/c
                            # auto-cleaning removed a non-speech sound, etc.
                            if len(utter) < 3:
                                continue
                            sequences.append(utter[1:])

                    # init module for computation
                    cm = Computation(rng, order=tps_order, interference=interf, unit_len=const.ULENS, method=tps_met)
                    # compute series
                    cm.compute(sequences)
                    # save shallow parsing results
                    # with open(fi_dir + ".shparctps", "w") as fp:
                    with open(fi_dir + "/" + fn.split('.capp')[0] + '.shpartips', "w") as fp:
                        for ln in cm.shallow_parsing:
                            fp.write(ln + "\n")
