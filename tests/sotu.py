import os
from glob import glob
from .textnets import TextCorpus, Textnets


corpus_files = glob(
        os.path.expanduser('~/nltk_data/corpora/state_union/*.txt'))

c = TextCorpus(corpus_files)
noun_phrases = c.noun_phrases()

tn = Textnets(noun_phrases)
g_groups = tn.graph(node_type='groups')
g_words = tn.graph(node_type='words')
