# from: https://www.nltk.org/howto/grammar.html
import random

from nltk import CFG, PCFG, ChartParser
from nltk.parse.generate import generate
from random import choice


def produce(gram, symbol):
    words = []
    productions = gram.productions(lhs=symbol)
    production = choice(productions)
    for sym in production.rhs():
        if isinstance(sym, str):
            words.append(sym)
        else:
            words.extend(produce(gram, sym))
    return words


# grammar = CFG.fromstring('''
#     S -> NP VP
#     PP -> P NP
#     VP -> V NP | VP PP
#     NP -> Det N | Det N PP | 'I'
#     V -> 'shot' | 'killed' | 'wounded'
#     Det -> 'an' | 'my'
#     N -> 'elephant' | 'pajamas' | 'cat' | 'dog'
#     P -> 'in' | 'outside'
#     ''')

grammar = CFG.fromstring('''
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
# toy_pcfg1 = PCFG.fromstring("""
#     S -> NP VP [1.0]
#     NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
#     Det -> 'the' [0.8] | 'my' [0.2]
#     N -> 'man' [0.5] | 'telescope' [0.5]
#     VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
#     V -> 'ate' [0.35] | 'saw' [0.65]
#     PP -> P NP [1.0]
#     P -> 'with' [0.61] | 'under' [0.39]
#     """)

# parser = ChartParser(grammar)
#
# gr = parser.grammar()
# for x in generate(gr):
#     # print(' '.join(produce(gr, gr.start())))
#     print(' '.join(x))

# generates input and input2 complete
A = ["mer", "kof", "daz"]
B = ["hox", "neb", "lev"]
C = ["tid", "rel", "jes"]
D = ["lum", "sot", "zor"]
E = ["rud", "fal", "taf"]
F = ["ker", "nav", "sib"]

# G = ["tril","kijo","fido"]
# H = ["haiku","zidyl","virxu"]
# res.append(a + g + h + e)

# (thomson&Newport,2007): http://socsci.uci.edu/~lpearl/courses/readings/ThompsonNewport2007_StatLearningSyn.pdf
# ABCDEF, plus ABCD, ABEF, and CDEF
# Baseline language.
# There is only one sentence type: ABCDEF. Each letter, A through F, # represents a form class analogous to the lexical
# classes noun and verb. Three words are assigned to each form class.
# Optional phrases.
# The optional phrases language uses the baseline language as a starting point, but in this language the six form
# classes are grouped in pairs (hereafter, phrases) as follows: AB, CD, and EF. A grammatical sentence may be created
# by removing one of the three phrases and making it “optional.” This results in a total of four distinct sentence
# types: ABCDEF, plus ABCD, ABEF, and CDEF. The optional phrases language can be represented by phrase structure rules:
# S→(P1)+(P2)+(P3);P1→A+B;P2→C+D;P3→E+F,with the stipulation that every sentence must have at least two phrases.

res = []
for a in A:
    for b in B:
        for c in C:
            for d in D:
                res.append(a + b + c + d)
                for e in E:
                    for f in F:
                        res.append(a + b + c + d + e + f)
for a in A:
    for b in B:
        for e in E:
            for f in F:
                res.append(a + b + e + f)

for c in C:
    for d in D:
        for e in E:
            for f in F:
                res.append(c + d + e + f)


random.shuffle(res)
with open("data/thompson_newport.txt", "w") as fp:
    for _l in res:
        fp.write(_l+"\n")
