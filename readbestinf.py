import ast
import sys
import numpy as np
with open(sys.argv[1]) as f:
    content = f.readlines()
f.close()
content = [ j.strip() for j in content]
objective = []
succes = []
params = []
for j in content:
    if j[0:2] == 'fu':
        objective.append(float(j.replace('fun: ','')))
    if j[0:2] == 'su':
        succes.append(bool(j.replace('success: ','')))
    if j[0:2] == 'x:':
        params.append((j.replace('x: array(','').replace(')','')))
params = [ast.literal_eval(j) for j in params]
params = np.array(params)
succes = np.array(succes)
objective = np.array(objective)
minob = np.argmin(objective)
if succes[minob]==True:
    np.save('parameters.npy',params[minob])
else:
    print('Something wrong in minimization parameters')

