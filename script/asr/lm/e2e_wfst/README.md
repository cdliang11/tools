# E2E WFST Tools

## Key Features

* static graph(TLG) and dynamic graph(TL/G)
* LM OOV handling
* Replace in graph building
  * class LM repalce, such as city names
  * FST replace, such as phone number FST
* Bug fix

## How to use?

1. Build decoding graph

``` sh
e2e/make_graph.sh \
  --e2e_unit $MODEL/unit.txt \
  --lm $MODEL/lm.arpa $dir

# Required parameters:
# 1. --e2e_unit: e2e model unit file
# 2. --lm: LM lm file
# Optional, you can add the following parameters:
# 1. --bpe_model: BPE model file, it is required for BPE based English word modeling.
# 2. --slot_config: slot config file
# 3. --fix_dict: dict for handing OOV or fix bug
# 4. --static: true or false, if true static TLG, if false dynamic TL/G.

```

## Implementation Details

1. Disambiguous symbol
  * @1, @2, ..., is used for disambiguous symbol for lexicon(L)
  * #0, #1, #2, is used for disambiguous symbol for LM(G)

2. token.txt contains:
  * <eps>
  * e2e model unit
  * disambig unit @1, @2, ..., which is determined by lexicon

3. words.txt contains:
  * <eps>
  * $1, $2, $3 for class replacement
  * <class> tag in class list
  * word in lexicon
  * #0 for LM backoff disambig, #1, #2, #3 for class disambig.

