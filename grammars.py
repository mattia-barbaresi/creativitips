# from: https://www.nltk.org/howto/grammar.html
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

parser = ChartParser(grammar)

gr = parser.grammar()
for x in generate(gr):
    # print(' '.join(produce(gr, gr.start())))
    print(' '.join(x))
