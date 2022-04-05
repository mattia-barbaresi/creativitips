Results are stored in dir with the following name format:

    [seg_fun]_[fun_par]_([mem_thresh]_[forgetting]_[interference])_[timestamp]

where

`seg_fun` is the chosen segmentation function: 
- 'RND': random segmentation, **PARSER-like** but using both bigram and trigram
- 'TPS': using tps and break percept when **tps(i) < tps(i+1)**
- 'BRENT': using tps and break when **tps(i-1) > tps(i) < tps(i+1)**;

`fun_par` are:
- for 'RND': length of minimum unit segmentation (e.g., for bigram and trigam is `(2,3)`)
- for 'TPS' and 'BRENT': the order of TPs used in segmentation (e.g., `1`,`2` or `3`);

`mem_thresh`,`forgetting`, and `interference` are the parameter for PARSER-like memory.

These directories contain also a *pars.txt* file storing the used parameters.

----------------------------------------------------------------------------

Finally, the inner dirs are named with the input file used and contain all results:
- *action.json* and *actions.pdf* store the action used for segmenting at each step (using memory, tps or random)
- *generated.json* contains the final generated sequences
- *results.json* contains the intermediate result generated each n=10 sequences and contains:
  - `generated`: generated sequences,
  - `mem`: content of memory at that iter
- *tps_symbols* and *tps_symbols.pdf* contain TPS (of the specified order) between symbols 
- *tps_units* and *tps_units.pdf* contain TPS of order=1 between segmented units
- *words_plot.png* is the final content of memory