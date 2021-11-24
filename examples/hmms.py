import json
import string
from collections import Counter
from sklearn import preprocessing
import numpy as np
from hmmlearn import hmm
import complexity as cx
import utils

# set random
np.random.seed(42)
np.set_printoptions(linewidth=np.inf)

# LabelEncoder for symbols (all lower letters)
le = preprocessing.LabelEncoder()
le.fit(list(string.ascii_lowercase))
print("classes:", le.classes_)

# load data
sequences_1, lens_1 = utils.read_input("../data/mod1.txt")
sequences_2, lens_2 = utils.read_input("../data/mod2.txt")

lbls = le.transform(sequences_1)
print("transf:", lbls)
print("inverse transfo:", le.inverse_transform(lbls))
sequences_multi = zip(sequences_1, sequences_2)
# print(Counter(e for e in sequences_multi))
hebb_mtx = utils.mtx_from_multi(le.transform(sequences_1), le.transform(sequences_2))
utils.plot_matrix(hebb_mtx, y_labels=le.classes_,save=False, title="Hebb matrix")

# create random sequence
sequences_rand = []
for ll in lens_1:
    sequences_rand.extend(np.random.randint(len(le.classes_),size=ll))
print("sequences : ",sequences_1)
print("sequences rand: ", sequences_rand)

# ------------------------------ TPS ------------------------------
# plot transitions probabilities matrix
with open("../data/input/tf.json", "r") as fp:
    tps = json.load(fp)
selected_tps = tps["1"]

tps_matrix = utils.matrix_from_tps(selected_tps,x_encoding=le,y_encoding=le)
# utils.plot_matrix(tps_matrix, y_labels=le.classes_)
# -----------------------------------------------------------------

# ------------------------------ entropy ------------------------------
# calculate entropy for the number of hidden states
seq_ent = cx.entropy(sequences_1)
seq_rand_ent = cx.entropy(sequences_rand)
print(len(sequences_rand), len(sequences_1))
print("Entropy of sequence: ", seq_ent)
print("Entropy of rand sequence: ", seq_rand_ent)
# -----------------------------------------------------------------

# ------------------------------ model ------------------------------
model1 = hmm.MultinomialHMM(n_components=3).fit(np.vstack(le.transform(sequences_1)), lens_1)
model2 = hmm.MultinomialHMM(n_components=3).fit(np.vstack(le.transform(sequences_2)), lens_2)
# model = hmm.MultinomialHMM(n_components=int(seq_rand_ent)).fit(np.vstack(sequences_rand), lens_1)
# model.emissionprob_ = tps[1]
# model.fit(np.vstack(sequences), lens)
# print(model.score(np.vstack(sequences)))

print("emission probabilities 1: ")
print(np.round(model1.emissionprob_, decimals=2),"\n")
print("emission probabilities 2: ")
print(np.round(model2.emissionprob_, decimals=2),"\n")

print("samples 1: ")
for x in range(0,10):
    X, Y = model1.sample(15)
    # Plot the sampled data
    # converted = [n2s_dict[n] for n in X[:, 0]]
    converted = le.inverse_transform(X[:, 0])
    print("".join(converted), " -->", Y)

print("\nsamples 2: ")
for x in range(0,10):
    X, Y = model2.sample(15)
    # Plot the sampled data
    # converted = [n2s_dict[n] for n in X[:, 0]]
    converted = le.inverse_transform(X[:, 0])
    print("".join(converted), " -->", Y)

