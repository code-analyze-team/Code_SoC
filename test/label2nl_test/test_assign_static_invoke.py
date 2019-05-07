from build_data.nl_transform import solve_assign_static_invoke


invokes = []
with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/assignments/assign_static_invoke', 'r') as f:
    for line in f.readlines():
        invokes.append(line)

errcount = 0
with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/assignments/assign_static_invoke_fail', 'w') as f:
    for invoke in invokes:
        if solve_assign_static_invoke(invoke) == None:
            f.write(invoke)
            errcount += 1
print('err: ' + str(errcount))

invo = invokes[27]
r = solve_assign_static_invoke(invo)
print(invo)
print(r)