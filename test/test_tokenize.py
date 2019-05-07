from build_data.preprocess import preprocessor
from build_data.embedding_ops import emb_ops



tk = preprocessor()
map = tk.get_jimple2tokens()

with open('/home/qwe/zfy_lab/pdg2vec/data/tokens', 'w') as f:
    for jimple, tokens_list in map.items():
        tokens = tokens_list[0]
        for t in tokens:
            f.write(t + ' ')
