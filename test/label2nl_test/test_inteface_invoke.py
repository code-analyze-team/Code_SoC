from build_data.nl_transform import solve_interface_invoke


invokes = []
with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/interface_invokes', 'r') as f:
    for line in f.readlines():
        invokes.append(line)

errcount = 0
with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/interface_invokes_fail', 'w') as f:
    for invoke in invokes:
        if solve_interface_invoke(invoke) == None:
            f.write(invoke)
            errcount += 1
print('err: ' + str(errcount))

invo = invokes[123]
r = solve_interface_invoke(invo)
print(invo)
print(r)