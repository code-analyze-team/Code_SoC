from build_data.nl_transform import nl_transformer

trans = nl_transformer()

jimples = []
with open('/home/qwe/zfy_lab/pdg2vec/data/jimples.txt', 'r') as f:
    for line in f.readlines():
        jimples.append(line)


# jimple = '"JAssignStmt_$i1 = $i0 + 1"'
# r = trans.determine_and_solve(jimple)

for jimple in jimples:
    try:
        trans.determine_and_solve(jimple)
    except:
        print(jimple)
        continue

print(len(jimples))
#
# with open('/home/qwe/zfy_lab/pdg2vec/data/nls' , 'w') as f:
#     for jimple in jimples:
#         nl = trans.determine_and_solve(jimple)
#         f.write(nl)
#         f.write('\n')