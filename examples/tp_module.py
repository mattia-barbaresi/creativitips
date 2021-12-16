import numpy as np
from examples import parser


class TPmodule:

    def __init__(self, order=1):
        self.mem = dict()
        self.order = order

    def encode(self, percept, wg=1.0):
        """
        Encode TPs of the given percept.
        :param wg assigned weight for memory entries
        :param percept could be a (i) string, or (ii) an (ordered, sequential) array of strings.
        In case (ii) TPs are counted between elements of the array (units), instead of between symbols
        """

        if self.order > 0:
            for _i in range(self.order, len(percept)):
                h = "".join(percept[_i - self.order:_i])
                o = "".join(percept[_i:_i + 1])
                if h in self.mem:
                    if o in self.mem[h]:
                        self.mem[h][o] += wg
                    else:
                        self.mem[h][o] = wg
                else:
                    self.mem[h] = {o: wg}
        else:
            print("(TPmodule):Order must be grater than 1.")

    def get_units(self, percept):
        """
        Returns segmented percept using stored TPs
        :param percept could be a string, or an (ordered, sequential) array of strings.
        In latter case TPs are counted between elements of the array(units), instead of between symbols
        """
        mems = sorted([(x,y) for x, y in self.mem.items() if x in percept], key=lambda item: item[1], reverse=True)
        if self.order > 0:
            for im in mems:
                s = percept.split(im)
                # self.mem[im][o]
                for _i in range(self.order, len(percept)):
                    h = "".join(percept[_i - self.order:_i])
                    o = "".join(percept[_i:_i + 1])
                    # if h in self.mem:
        else:
            print("(TPmodule):Order must be grater than 1.")


def read_percepts(seqs):
    res = ["babupada"]  # for testing
    for s in seqs:
        while len(s[::2]) > 1:
            # bi-gram at least
            ln = np.random.randint(low=1, high=4) * 2
            res.append(s[:ln])
            s = s[ln:]
        if len(s) > 0:
            res.append(s)
    return res


if __name__ == "__main__":
    np.random.seed(55)
    w = 1.0
    f = 0.05
    i = 0.005
    tps1 = TPmodule(1)  # memory for TPs
    pmem = dict()  # memory for parser segments

    # input
    with open("../data/input.txt", "r") as fp:
        sequences = [line.rstrip() for line in fp]

    # read percepts using parser function
    for p in parser.read_percepts(sequences):
        print("-----------------------------------")
        print("percept: ", p)
        tps1.encode(p,wg=w)
        units = []
        if len(p) <= 2:
            # p is a unit, a primitive
            if p in pmem:
                pmem[p] += 0.5
            else:
                pmem[p] = 1
        else:
            # add p and its components (bi-grams)
            units = parser.get_units(pmem, p)
            parser.add_weight(pmem, p, comps=units, weight=w)
        # apply forgetting and interference
        parser.forget_interf(pmem, p, comps=units, forget=f, interfer=i)

        print("mem:", list(sorted(pmem.items(), key=lambda item: item[1], reverse=True)))

    ord_mem = dict(sorted([(x, y) for x, y in pmem.items() if len(x) > 2], key=lambda item: item[1], reverse=True))
