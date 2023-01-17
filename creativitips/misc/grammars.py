# from: https://www.nltk.org/howto/grammar.html

import random as rnd
import random
from nltk import CFG, PCFG, ChartParser
from nltk.parse.generate import generate as nltk_generate


def generate_from_grammar():
    grammar1 = CFG.fromstring('''
        S -> NP VP
        PP -> P NP
        VP -> V NP | VP PP
        NP -> Det N | Det N PP | 'I'
        V -> 'shot' | 'killed' | 'wounded'
        Det -> 'an' | 'my'
        N -> 'elephant' | 'pajamas' | 'cat' | 'dog'
        P -> 'in' | 'outside'
        ''')
    grammar2 = CFG.fromstring('''
        ROOT -> S
        S -> A B C D | E F G
        A -> 'aa' | 'bb' | 'cc'
        B -> 'aaa' | 'bbb' | 'ccc'
        C -> 'tuple' | 'taple' | 'tiple'
        D -> 'zarro' | 'zirro' | 'zorro'
        E -> 'rudolf' | 'radolf' | 'ridolf'
        F -> 'birman' | 'borman' | 'barman'
        G -> 'tri' | 'vas' | 'kuk'
        ''')
    # probabilistic one
    grammar3 = PCFG.fromstring("""
        S -> NP VP [1.0]
        NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
        Det -> 'the' [0.8] | 'my' [0.2]
        N -> 'man' [0.5] | 'telescope' [0.5]
        VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
        V -> 'ate' [0.35] | 'saw' [0.65]
        PP -> P NP [1.0]
        P -> 'with' [0.61] | 'under' [0.39]
        """)

    parser = ChartParser(grammar3)
    gr = parser.grammar()
    for x in nltk_generate(gr):
        # print(' '.join(produce(gr, gr.start())))
        print(' '.join(x))


def hello():
    """Simulated symbolic sequences for arm mimicking 'Hello!' """
    res = []
    AR = ["1", "2", "3", "4", "5"]
    H = ["A", "B", "C"]
    AL = ["1", "2", "3", "4", "5"]

    # symb = [AL-H-AR]

    # hello right arm : 1A5 1A4 1A3 1A2 1A1 1A2 1A1 1A2 1A3 1A4 1A5
    # hello right arm : 1B5 2B4 3A3 1A2 1A1 1A2 1A1 1A2 1A3 1A4 1A5

    # generates right hello

    for x in range(1, 100):
        res.append(rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "1 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "2 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "3 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "4 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "5 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "4 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "5 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "4 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "3 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "2 " +
                   rnd.sample(AL, 1)[0] + rnd.sample(H, 1)[0] + "1")
    random.shuffle(res)
    with open("../data/hello.txt", "w") as fp:
        for _l in res:
            fp.write(_l + "\n")


def thompson_newport2007():
    """
    (thomson&Newport,2007)
    http://socsci.uci.edu/~lpearl/courses/readings/ThompsonNewport2007_StatLearningSyn.pdf

    ABCDEF, plus ABCD, ABEF, and CDEF
    Baseline language.
    There is only one sentence type: ABCDEF. Each letter, A through F, # represents a form class analogous to the lexical
    classes noun and verb. Three words are assigned to each form class.
    Optional phrases.
    The optional phrases language uses the baseline language as a starting point, but in this language the six form
    classes are grouped in pairs (hereafter, phrases) as follows: AB, CD, and EF. A grammatical sentence may be created
    by removing one of the three phrases and making it “optional.” This results in a total of four distinct sentence
    types: ABCDEF, plus ABCD, ABEF, and CDEF. The optional phrases language can be represented by phrase structure rules:
    S→(P1)+(P2)+(P3); P1→A+B; P2→C+D; P3→E+F,
    with the stipulation that every sentence must have at least two phrases.
    """
    res = []

    A = ["mer", "kof", "daz"]
    B = ["hox", "neb", "lev"]
    C = ["tid", "rel", "jes"]
    D = ["lum", "sot", "zor"]
    E = ["rud", "fal", "taf"]
    F = ["ker", "nav", "sib"]

    # generates input and input2 complete
    # G = ["tril","kijo","fido"]
    # H = ["haiku","zidyl","virxu"]
    # res.append(a + g + h + e)

    for a in A:
        for b in B:
            for c in C:
                for d in D:
                    res.append(a + b + c + d)
                    for e in E:
                        for f in F:
                            res.append(a + b + c + d + e + f)

            for e in E:
                for f in F:
                    res.append(a + b + e + f)

    for c in C:
        for d in D:
            for e in E:
                for f in F:
                    res.append(c + d + e + f)

    random.shuffle(res)

    with open("../data/thompson_newport_nebrelsot.txt", "w") as fp:
        for _l in res:
            if "nebrelsot" not in _l:
                fp.write(_l + "\n")


def Gomez2002():
    """
    'Variability and Detection of Invariant Structure'
    (Gomez, 2002)
    """
    pass


def Onnis2003():
    """
    Reduction of Uncertainty in Human Sequential Learning: Evidence from Artificial Grammar Learning
    (Onnis et al., 2003)

    Variability was manipulated in 5 conditions, by drawing X from a pool of either 1, 2, 6, 12, or 24 elements.
    The elements a, b, and c were instantiated as pel, vot, and dak; d, e, and f, were instantiated as rud, jic, tood.
    The 24 middle items were wadim, kicey, puser, fengle, coomo, loga, gople, taspu, hiftam, deecha, vamey,
    skiger, benez, gensim, feenam, laeljeen, chla, roosa, plizet, balip, malsig, suleb, nilbo, and wiffle.
    Following the design by Gómez (2002) the group of 12 middle elements were drawn from the first 12 words in the list,
    the set of 6 were drawn from the first 6, the set of 2 from the first 2 and the set of 1 from the first word.
    Three strings in each language were common to all five groups, and they were used as test stimuli."""

    a = "pet"
    b = "vot"
    c = "dak"
    d = "rud"
    e = "jic"
    f = "tood"
    X = ["wadim", "kicey", "puser", "fengle", "coomo", "loga", "gople", "taspu", "hiftam",
         "deecha", "vamey", "skiger", "benez", "gensim", "feenam", "laeljeen", "chla", "roosa",
         "plizet", "balip", "malsig", "suleb", "nilbo", "wiffle"]

    # the total is 432 strings for all
    # 3x24 -> 72 x 6
    # 3x12 -> 36 x 12
    # 3x6 -> 18 x 24
    # 3x2 -> 6 x 72
    # so 432/3 = 144/n = n° duplicates
    for n in [1]:
        L1 = []
        L2 = []
        for x in X[:n]:
            for _i in range(int(144 / n)):
                #  Strings in L1 had the form aXd, bXe, and cXf.
                L1.append(a + x + d)
                L1.append(b + x + e)
                L1.append(c + x + f)
                # L2 strings had the form aXe, bXf, cXd.
                L2.append(a + x + e)
                L2.append(b + x + f)
                L2.append(c + x + d)

        random.shuffle(L1)
        random.shuffle(L2)

        with open("../data/Onnis2003_L1_{}.txt".format(n), "w") as fp:
            for _l in L1:
                fp.write(_l + "\n")
        with open("../data/Onnis2003_L2_{}.txt".format(n), "w") as fp:
            for _l in L2:
                fp.write(_l + "\n")


if __name__ == "__main__":
    # generate_from_grammar()
    # hello()
    thompson_newport2007()
