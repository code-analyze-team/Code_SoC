from build_data.nl_transform import nl_transformer
from build_data.embedding_ops import emb_ops
from metrics.variance import metric_variance
import tensorflow as tf
import gc
from sys import getrefcount

def preprocess(data):
    stop_list = [':', '@', '=', '(', ')', '()', '\'\'', '( )', ',', '.','$', '\\', '<', '>', 'z', 'r', 'a']
    filtered = [t for t in data if (t not in stop_list)]
    return filtered

op = emb_ops('/home/qwe/zfy_courses/pdg2vec_data/nl_res_skipgram/embeddings.tsv')
trans = nl_transformer()

"""
make sure metadata.tsv == pretrain tokens
"""
token_id_map = {}
with open('/home/qwe/zfy_courses/pdg2vec_data/nl_res_skipgram/metadata.tsv', 'r') as f:
    count = 0
    for line in f.readlines():
        label = line
        label = label.replace('\n', '')
        id_ = count
        token_id_map.setdefault(label, '')
        token_id_map[label] = id_
        count += 1

tokens = []
with open('/home/qwe/zfy_courses/pdg2vec_data/pre_tokens_edge.txt', 'r') as f:
    text = f.read().split(' ')
    for t in text:
        tokens.append(t)
tokens = preprocess(tokens)
print(len(set(tokens)), count)
assert len(set(tokens)) == count


"""
fetch all labels
"""
labels = []
with open('/home/qwe/zfy_courses/pdg2vec_data/random_walk_data_edge#one.txt', 'r') as f:
    text = f.read().split('#')
    for t in text:
        labels.append(t)
label_set = set(labels)
print(len(label_set))


with open('/home/qwe/zfy_courses/pdg2vec_data/jimple_emb.tsv', 'w') as f:
    with tf.Session() as sess:
        i = 0
        for label in label_set:
            print(i)
            tokens = trans.label2tokens(label)
            tokens = preprocess(tokens)
            tokens_id = [token_id_map.get(t) for t in tokens]
            embs = op.lookup_1d(tokens_id)
            mean = tf.reduce_mean(embs, axis=0)
            mean_value = sess.run(mean)
            # f.write(jimple + '##')
            for v in mean_value:
                f.write(str(v) + ' ')
            f.write('\n')
            gc.collect()
            i += 1


