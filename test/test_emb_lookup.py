from build_data.nl_transform import nl_transformer
from build_data.embedding_ops import emb_ops
from metrics.variance import metric_variance
import tensorflow as tf
import gc
from sys import getrefcount

op = emb_ops('/home/qwe/zfy_courses/w2v_learn/skipgram_tokens/embeddings.tsv')
trans = nl_transformer()
map = trans.label2tokens()

mt = metric_variance()

token_tsvid_map = {}
with open('/home/qwe/zfy_courses/w2v_learn/skipgram_tokens/metadata.tsv', 'r') as f:
    count = 0
    for line in f.readlines():
        label = line
        label = label.replace('\n', '')
        id_ = count
        token_tsvid_map.setdefault(label, '')
        token_tsvid_map[label] = id_
        count += 1

jimples = list(map.keys())
print(len(jimples))
print(count)

# with open('/home/qwe/zfy_lab/pdg2vec/data/jimple_mean_emb.tsv', 'w') as f:
#     with tf.Session() as sess:
#         i = 0
#         for jimple in jimples:
#             print(i)
#             tokens = map.get(jimple)[0]
#             tokens_id = [token_tsvid_map.get(t) for t in tokens]
#             embs = op.lookup_1d(tokens_id)
#             mean = tf.reduce_mean(embs, axis=0)
#             mean_value = sess.run(mean)
#             # f.write(jimple + '##')
#             for v in mean_value:
#                 f.write(str(v) + ' ')
#             f.write('\n')
#             gc.collect()
#             i += 1


