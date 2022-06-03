# Examples of Bayesian and HMMs: statistics, learning and inference

---
### An HMM:
 - **Q** = {q<sub>1</sub>,q<sub>2</sub>, .. , q<sub>N</sub>}, states
 - **V** = {v<sub>1</sub>, ..., v<sub>M</sub>}, symbols
 - **A** = {a<sub>ij</sub>}, a<sub>ij</sub> = Pr(q<sub>j</sub> at t+1, q<sub>i</sub> at t), state transition probability distribution
 - **B** = {b<sub>j</sub>(k)}, b<sub>j</sub>(k) = Pr(v<sub>k</sub> at state q<sub>j</sub>), observation symbol probability distribution in state j
 - **π** = {π<sub>i</sub>}, π<sub>i</sub> = Pr(q<sub>i</sub> at t =1), initial state distribution


There are three fundamental problems for HMMs:
- Given the model parameters and observed data, estimate the optimal sequence of hidden states.
- Given the model parameters and observed data, calculate the model likelihood.
- Given just the observed data, estimate the model parameters (Structure Learning).

The first and the second problem can be solved by the dynamic programming algorithms known as the **Viterbi** algorithm and the **Forward-Backward algorithm**, respectively.
The last one can be solved by an iterative *Expectation-Maximization* (EM) algorithm, known as the **Baum-Welch algorithm**.


#### Sources
hmmlearn: https://hmmlearn.readthedocs.io/en/latest/

hmmlearn vs. pomegranate: https://github.com/jmschrei/pomegranate/blob/master/benchmarks/pomegranate_vs_hmmlearn.ipynb

hmm model testing: https://github.com/manitadayon/Auto_HMM

causal Hmms: https://github.com/LilJing/causal_hmm 

>#####References: 
>  - Lawrence R. Rabiner “A tutorial on hidden Markov models and selected applications in speech recognition”, Proceedings of the IEEE 77.2, pp. 257-286, 1989.
>  - Jeff A. Bilmes, “A gentle tutorial of the EM algorithm and its application to parameter estimation for Gaussian mixture and hidden Markov models.”, 1998.
>  - Mark Stamp. “A revealing introduction to hidden Markov models”. Tech. rep. Department of Computer Science, San Jose State University, 2018. url: http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf.

