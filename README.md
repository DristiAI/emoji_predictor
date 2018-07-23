# emoji_predictor

A model using 183 labelled sentences and using word2vec to be better at generalization

I have skipped the Embedding layer and instead used word2vec dictionary on both train and test time to feed sequences.This helps small dataset models as we can atleast generalise for conditions where the word's though not in training set but has similar semantic meaning can be used effectively by model. The only caveat being that we need to comute embeddings at test time from word2vec for effectively getting vectors for unknown words wrt training set.

