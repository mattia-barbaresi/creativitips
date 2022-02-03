# module for word embeddings

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd


class EmbedModule:
    def __init__(self, mtx):
        self.vh = None
        self.u = None  # Each row of U matrix is a 3-dimensional vector representation of word
        self.s = None
        self.trans = None
        self.mtx = mtx
        self.squared = None

    def compute(self,lex):
        # renorm (Hellinger) distance
        self.squared = np.sqrt(self.mtx)
        # singular value decomposition
        svd = TruncatedSVD(n_components=3)
        self.trans = svd.fit_transform(self.squared)
        self.plot3D(self.trans,lex)
        u, s, vt = randomized_svd(self.squared, n_components=3)
        self.plot3D(u,lex)
        # self.u, self.s, self.vh = np.linalg.svd(self.squared, full_matrices=True)
        print("dfsf")

    @staticmethod
    def plot3D(mtx,le):
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(top=1.1, bottom=-.1)
        ax = fig.add_subplot(111, projection='3d')

        x,y,z = mtx.T
        for _ in range(len(z)):
            ax.scatter(x[_], y[_], z[_], cmap='viridis', linewidth=0.5)
            ax.text(x[_], y[_], z[_], le.inverse_transform([_])[0])
            print(le.inverse_transform([_])[0])
        plt.show()
