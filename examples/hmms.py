import os
from collections import Counter
from sklearn import preprocessing
import numpy as np
from hmmlearn import hmm
import complexity as cx
import const
import utils
from scipy.special import softmax

# ------------------------------ INIT ------------------------------
# set random
rng = np.random.default_rng(const.RND_SEED)
np.set_printoptions(linewidth=np.inf)

# LabelEncoder for symbols (all lower letters)
le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
le_multi = preprocessing.LabelEncoder()

# ------------------------------ DATA ------------------------------
# sequences_1, lens_1 = utils.read_input("../data/mod1.txt")
# sequences_2, lens_2 = utils.read_input("../data/mod2.txt")
# le.fit(list(string.ascii_lowercase))

# load bicinia
sequences_1, sequences_2, lens_1, lens_2 = utils.load_bicinia("../data/bicinia/")
le1.fit(sequences_1)
le2.fit(sequences_2)
sequences_multi = ["-".join(x) for x in zip(sequences_1, sequences_2)]
le_multi.fit(sequences_multi)
# -----------------------------------------------------------------

print(Counter(e for e in sequences_multi))
print("le1: ", le1.classes_)
print("le2: ", le2.classes_)
hebb_mtx = utils.mtx_from_multi(le1.transform(sequences_1), le2.transform(sequences_2),
                                nd1=len(le1.classes_), nd2=len(le2.classes_))
# utils.plot_matrix(hebb_mtx, x_labels=le.classes_, y_labels=le2.classes_,save=False, title="Hebb matrix", clim=False)
# sm_hebb = utils.softmax(hebb_mtx, axis=1)
sm_hebb = softmax(hebb_mtx, axis=1)
print("sm_hebb")
print(np.round(sm_hebb, decimals=4))
# utils.plot_matrix(sm_hebb, x_labels=le2.classes_,y_labels=le1.classes_, title="sm_hebb matrix", clim=False)

# create random sequence
# sequences_rand = []
# for ll in lens_1:
#     sequences_rand.extend(rng.integers(len(le.classes_),size=ll))
# print("sequences : ",sequences_1)
# print("sequences rand: ", sequences_rand)

# ------------------------------ TPS ------------------------------
# plot transitions probabilities matrix
# with open("../data/input/tf.json", "r") as fp:
#     tps = json.load(fp)
# selected_tps = tps["1"]
#
# tps_matrix = utils.matrix_from_tps(selected_tps,x_encoding=le,y_encoding=le)
# utils.plot_matrix(tps_matrix, y_labels=le.classes_)
# -----------------------------------------------------------------

# ------------------------------ entropy ------------------------------
# calculate entropy for the number of hidden states
seq_ent1 = cx.entropy(sequences_1)
seq_ent2 = cx.entropy(sequences_2)
seq_ent_multi = cx.entropy(le_multi.transform(sequences_multi))
# seq_rand_ent = cx.entropy(sequences_rand)
print("seq_ent1: ", seq_ent1)
print("seq_ent2: ", seq_ent2)
print("seq_ent_multi: ", seq_ent_multi)
# print("seq_rand_ent: ", seq_rand_ent)
# -----------------------------------------------------------------

# ------------------------------ model ------------------------------
# n_state_1 = int(seq_ent1)+1
# n_state_2 = int(seq_ent2)+1
# n_state_multi = int(seq_ent_multi)+1
n_state_1 = 11
n_state_2 = 10
n_state_multi = 15
model1 = hmm.MultinomialHMM(n_components=n_state_1).fit(np.vstack(le1.transform(sequences_1)), lens_1)
model2 = hmm.MultinomialHMM(n_components=n_state_2).fit(np.vstack(le2.transform(sequences_2)), lens_2)
model_multi = hmm.MultinomialHMM(n_components=n_state_multi).fit(np.vstack(le_multi.transform(sequences_multi)), lens_1)
# model = hmm.MultinomialHMM(n_components=int(seq_rand_ent)).fit(np.vstack(sequences_rand), lens_1)

# prints
print("state transitions 1: ")
print(np.round(model1.transmat_, decimals=2), "\n")
# utils.plot_matrix(model1.transmat_, fileName="", title="model1.transmat_")
print("state transitions 2: ")
print(np.round(model2.transmat_, decimals=2), "\n")
# utils.plot_matrix(model2.transmat_, fileName="", title="model2.transmat_")
print("state transitions multi: ")
print(np.round(model_multi.transmat_, decimals=2), "\n")
# utils.plot_matrix(model_multi.transmat_, fileName="", title="model_multi.transmat_")
print("emission probabilities 1: ")
print(np.round(model1.emissionprob_, decimals=2), "\n")
utils.plot_matrix(model1.emissionprob_,x_labels=le1.classes_, fileName="", title="model1.emissionprob_")
print("emission probabilities 2: ")
print(np.round(model2.emissionprob_, decimals=2), "\n")
# use le2 instead of le because this stream has less symbols
# utils.plot_matrix(model2.emissionprob_,x_labels=le2.classes_, save=False, title="model2.emissionprob_")
# print("emission probabilities multi: ")
print(np.round(model_multi.emissionprob_, decimals=2), "\n")
utils.plot_matrix(model_multi.emissionprob_, x_labels=le_multi.classes_, fileName="", title="mmulti.emissionprob_")

# ------------------------------ out ------------------------------
# save models
dir_name = const.EXAMPLES_OUT_DIR + "out_hmms_{}_{}_{}".format(n_state_1,n_state_2,n_state_multi)
data = {
    "m1_state_transitions": model1.transmat_,
    "m1_emission_probabilities": model1.emissionprob_,
    "m2_state_transitions": model2.transmat_,
    "m2_emission_probabilities": model1.emissionprob_,
    "multi_state_transitions": model_multi.transmat_,
    "multi_emission_probabilities": model_multi.emissionprob_
}

# os.makedirs(dir_name, exist_ok=True)
os.makedirs(dir_name+"/model/", exist_ok=True)

for fn, d in data.items():
    with open(dir_name+"/model/"+fn+".txt", "w") as a_file:
        # for row in d:
        np.savetxt(a_file, d, fmt='%.2f')


# with open(dir_name+"/models.json", "w") as of:
#     json.dump(data,of)

# separate generation
for x in range(0, 10):
    with open(dir_name+"/single{}.txt".format(x), "w") as of:
        X1, Y1 = model1.sample(500)
        X2, Y2 = model2.sample(500)
        # Plot the sampled data
        converted1 = le1.inverse_transform(X1[:, 0])
        converted2 = le2.inverse_transform(X2[:, 0])
        of.write(" ".join([str(x) for x in converted1]))
        of.write("\n\n")
        of.write(" ".join([str(x) for x in converted2]))
        # print(" ".join([str(x) for x in converted1]), " -->", Y1)
        print(" ".join([str(x) for x in converted2]), " -->", Y2)
        # print()

    # hebb generation with previous samples X1
    with open(dir_name+"/hebb{}.txt".format(x), "w") as of:
        # Plot the sampled data
        converted = le1.inverse_transform(X1[:, 0])
        hebb_seq = utils.hebb_gen(rng, X1[:, 0],sm_hebb)
        converted2 = le2.inverse_transform(hebb_seq)
        of.write(" ".join([str(x) for x in converted]))
        of.write("\n\n")
        of.write(" ".join([str(x) for x in converted2]))
        # print(" ".join(list(zip(*converted2))[0]))
        # print(" ".join(list(zip(*converted))[1]))
        print()

# multi generation
for x in range(0, 10):
    with open(dir_name+"/multi{}.txt".format(x), "w+") as of:
        X, Y = model_multi.sample(500)
        # Plot the sampled data
        converted = [(x.split("-")) for x in le_multi.inverse_transform(X[:, 0])]
        of.write(" ".join(list(zip(*converted))[0]))
        of.write("\n\n")
        of.write(" ".join(list(zip(*converted))[1]))
        # print(" ".join(list(zip(*converted))[0]))
        # print(" ".join(list(zip(*converted))[1]))
        # print()
