from build_data.nl_transform import solve_assign_others


invokes = []
with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/assignments/assignments', 'r') as f:
    for line in f.readlines():
        invokes.append(line)

# errcount = 0
# with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/assignments/assign_dynamic_invoke_fail', 'w') as f:
#     for invoke in invokes:
#         if solve_assign_others(invoke) == None:
#             f.write(invoke)
#             errcount += 1
# print('err: ' + str(errcount))

# invo = invokes[228]
# print(invo)
# r = solve_assign_others(invo)
# print(r)

invo = 'JAssignStmt_$r1 = r0.<org.mybatis.spring.batch.MyBatisCursorItemReader: java.util.Iterator cursorIterator>'
solve_assign_others(invo)