from build_data.nl_transform import solve_virtual_invoke


invokes = []
with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/virtual_invokes', 'r') as f:
    for line in f.readlines():
        invokes.append(line)

errcount = 0
with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/virtual_invokes_fail', 'w') as f:
    for invoke in invokes:
        if solve_virtual_invoke(invoke) == None:
            f.write(invoke)
            errcount += 1
print('err: ' + str(errcount))

invo = invokes[7]
r = solve_virtual_invoke(invo)
print(invo)
print(r)