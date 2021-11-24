# Examples of Bayesian and MCMC statistics and inference with pymc and scipy

---

## HMM

##### An HMM:
 - **Q** = $${q_1,q_2, .. , q_N}$, states
 - **V** = {v_1, ..., v_M}, symbols
 - **A** = {aij}, aij = Pr(qj at t + 11 qi at t), state transition probability distribution
 - **B** = {bj(k)}, bj(k) = Pr(vk at tl q; at t), observation symbol probability distribution in state i
 - **π** = {π_i}, π_i = Pr(q_i at t =1), initial state distribution


There are three fundamental problems for HMMs:
- Given the model parameters and observed data, estimate the optimal sequence of hidden states.
- Given the model parameters and observed data, calculate the model likelihood.
- Given just the observed data, estimate the model parameters (Structure Learning).

The first and the second problem can be solved by the dynamic programming algorithms known as the **Viterbi** algorithm and the **Forward-Backward algorithm**, respectively.
The last one can be solved by an iterative *Expectation-Maximization* (EM) algorithm, known as the **Baum-Welch algorithm**.


#### Sources
hmmlearn: https://hmmlearn.readthedocs.io/en/latest/

pomegranate comparison: https://github.com/jmschrei/pomegranate/blob/master/benchmarks/pomegranate_vs_hmmlearn.ipynb


>#####References: 
>  - Lawrence R. Rabiner “A tutorial on hidden Markov models and selected applications in speech recognition”, Proceedings of the IEEE 77.2, pp. 257-286, 1989.
>  - Jeff A. Bilmes, “A gentle tutorial of the EM algorithm and its application to parameter estimation for Gaussian mixture and hidden Markov models.”, 1998.
>  - Mark Stamp. “A revealing introduction to hidden Markov models”. Tech. rep. Department of Computer Science, San Jose State University, 2018. url: http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf.

