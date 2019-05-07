from build_data.nl_transform import nl_transformer


trans = nl_transformer()


from_data_path = '/home/qwe/zfy_courses/pdg2vec_data/random_walk_data_edge#one.txt'
to_write_path = '/home/qwe/zfy_courses/pdg2vec_data/pre_tokens_edge.txt'

with open(from_data_path, 'r') as f:
    with open(to_write_path, 'w') as f2:
        text = f.read()
        labels = text.split('#')
        for label in labels:
            nl = trans.label2tokens(label)
            for t in nl:
                f2.write(t + ' ')


"""
following part in zfy_courses/cg2vec/nl_text8_test_model/skipgram_tokens.ipynb
"""