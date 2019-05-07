from build_data.nl_transform import solve_special_invoke


invokes = []
with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/special_invokes', 'r') as f:
    for line in f.readlines():
        invokes.append(line)

errcount = 0
with open('/home/qwe/PycharmProjects/cg2vec/data/jimple_types/special_invokes_fail', 'w') as f:
    for invoke in invokes:
        if solve_special_invoke(invoke) == None:
            f.write(invoke)
            errcount += 1
print('err: ' + str(errcount))

invo = invokes[17]
r = solve_special_invoke(invo)
print(invo)
print(r)