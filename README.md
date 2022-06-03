# Segmentation and Chunking
## using Transitional Probabilities and attentional mechanisms

____________________________
### Dependencies:
- pymc
- pygraphviz
- 
### Some key parameters that alter results the most:

- **fogetting**
- **memory activation**
- **interference**
- the size of data:
    - for a short-term memory, a lot of (different) data create messy representations.

### TODOs:
- TPS:
  - why class form retain "END" but not "START"?
  - deleting normalization of memories 
  - (interference or softmax?): calculating tps with interference (incremental, on-line, not with softmax (normalized)
  - from tps of symbols to tps of units (generalization, abstraction)
  - Replay of experiences?? 


### Issues: 
- segmentation: TPS (forward, backward, MI)??
- chunking: with or without TPS? using MDL?
- abstraction/generalization: (algebraic patterns?)
- creative generation, using **Simonton's formula**: how to make it converge faster?


### Local dist steps:
build a package:
- `pip install .`
- `python setup.py sdist`
- 

Then (inside new project's venv):
`pip install [FULL_PATH_TO_DIST]/creativitips-[VERS].tar.gz`


### What's next?

- abstraction
- state uncertainty
- multi-modal integration