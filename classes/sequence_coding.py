# (from https://www.sciencedirect.com/science/article/pii/S089662731500776X)

from graphs import GraphModule
from pparser import ParserModule
from tps import TPSModule


class Knowledge:
    """Class for sequential learning 5 types of knowledge:
        - tps / timing
        - chunking
        - ordinal
        - algebraic
        - tree structure
        Variable, each mechanism is employed when required, due to attentional and input specs."""

    def __init__(self, rng, order=2, weight=1.0, interference=0.001, forgetting=0.05,
                 memory_thres=1.0, unit_len=None, method="BRENT"):

        # tps / timing:
        #
        self.tps = TPSModule()
        self.chunk = ParserModule()
        # self.ordinal = Ordinal()
        # self.algebraic = TPSModule()
        self.tree = GraphModule()

    def encode(self, percept):
        self.tps.encode(percept)
        self.chunk.encode(percept, units)
        # self.algebraic.encode(percept)
        # self.ordinal.encode(units)
        self.tree.encode(units)
