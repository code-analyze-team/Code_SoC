# from build_data.embedding_ops import emb_ops
#
# glove_path = '/home/qwe/zfy_data/SoC/data_new/nl_pre/glove/glove.6B.100d.txt'
#
# op = emb_ops(glove_path)
#
# tokens = []
# with open('/home/qwe/zfy_data/SoC/data_new/nl_pre/metadata.tsv', 'r') as f:
#     for line in f.readlines():
#         line = line.strip('\n')
#         tokens.append(line)
# # print(len(tokens))
#
# emb = op.lookup_glove_embedding_batch(tokens[:5])
# print(emb.shape)

l = ['a', 'a', 'c', 'd' ,'d']
l_ = set(l)
print(l_)