# (from https://www.sciencedirect.com/science/article/pii/S089662731500776X)

from graphs import TPsGraph
from pparser import Parser
from tps import TPS


class Knowledge:
    """Class for sequential learning 5 types of knowledge:
        - tps / timing
        - chunking
        - ordinal
        - algebraic (?)
        - tree structure
        Variable, each mechanism is employed when required, due to attentional and input specs.

        The idea is to model all these types within a graph structure learned incrementally in un unsupervised manner.
        The working memory consists of activation of some nodes in the graph at each time step (also with decay).
        The symbolic (with explicit connectionist traits) approach is the right choice towards opens science for
        cognitive research (psycho and neuro). (like Friston's FEP, by the way)"""

    def __init__(self, rng, order=2, weight=1.0, interference=0.001, forgetting=0.05,
                 memory_thres=1.0, unit_len=None, method="BRENT"):

        self.rng = rng
        self.tps = TPS()
        self.chunk = Parser()
        # self.ordinal = Ordinal()
        # self.algebraic = TPS()
        self.tree = TPsGraph()

    def encode(self, percept):
        pass

    def generate(self):
        pass

