# ISL and Creativity - A production-oriented model
## Segmentation and Chunking using Transitional Probabilities and attentional mechanisms

____________________________
### Dependencies:
- pymc
- pygraphviz

### Some key parameters that alter results the most:

- **fogetting**
- **memory activation**
- **interference**
- the size of data:
    - for a short-term memory, a lot of (different) data create messy representations.


##### TODOs:
- TPS:
  - deleting normalization of memories 
  - (interference or softmax?): calculating tps with interference (incremental, on-line, not with softmax (normalized)



### Issues: 
- segmentation: TPS (forward, backward, MI)??
- chunking: with or without TPS? using MDL?
- abstraction/generalization: (algebraic patterns?)
- creative generation, using **Simonton's formula**: how to make it converge faster?


### What's next?
- Replay of experiences
- Retrieval
- state uncertainty
- Inference and error-driven learning
- Multi-modal integration

--------------------------------
##### Local dist steps:
build a package:
- `pip install .`
- `python setup.py sdist`

Then (inside new project's venv):
`pip install [FULL_PATH_TO_DIST]/creativitips-[VERS].tar.gz`


