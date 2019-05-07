from build_data.nl_transform import solve_returns


# invokes = []
# with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/assignments/assign_dynamic_invoke', 'r') as f:
#     for line in f.readlines():
#         invokes.append(line)
#
# errcount = 0
# with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/assignments/assign_dynamic_invoke_fail', 'w') as f:
#     for invoke in invokes:
#         if solve_assign_dynamic_invoke(invoke) == None:
#             f.write(invoke)
#             errcount += 1
# print('err: ' + str(errcount))

input = 'JReturnStmt_return \"Property \'mapperLocations\' was not specified or no matching resources found\"'
r = solve_returns(input)
print(r)