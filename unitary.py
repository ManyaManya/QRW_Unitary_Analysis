import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#set the number of dimensions
n = 3
N = pow(2, n)

np.set_printoptions(linewidth=np.inf)

psi_proj = (1/n)*np.matrix(np.ones((n, n))) #superposition of coin space

Co = 2*psi_proj -  np.matrix(np.identity(n))
print("Co:\n", n*Co, "\n")#factor out normalization for ease of reading

C = np.matrix(np.kron(Co, np.identity(N)))
print("C:\n", n*C, "\n")

S = np.matrix(np.zeros((n*N, n*N)))
d = 0
for i in range(0, n):
    for x in range(0, N):
        ed = d + pow(2, i)
        S[x^ed, x+d] = 1
        S[x+d, x^ed] = 1
    d += N

print("S:\n", S, "\n")

U = S*C
print("U:\n", n*U, "\n")

zero_proj = np.matrix(np.zeros((N, N)))
zero_proj[0,0] = 1

#the qrw unitary operator
U_ = U - 2*S * np.kron(psi_proj, zero_proj)
print("U_:\n", n*U_, "\n")

#calc the eigenvalues
val, vect = np.linalg.eig(U_)
val_u, vect_u = np.linalg.eig(U)

#numpy calcs extremely close valued eignvals, round them off 
val = np.ndarray.round(val, 8)
val_u = np.ndarray.round(val_u, 8)

#remove duplicate values
val, idx = np.unique(val[val.imag != 0], return_index=True)
vect = vect[:, idx]
val_u, idx_u = np.unique(val_u, return_index=True)
vect_u = vect_u[:, idx_u]

#print eigvals and vects in arc
idx_s = np.argsort(-val)#rev values to sort in descending order
print("\neigval_0+: ", val[idx_s[0]],"\n omega_0+:\n", vect[:, idx_s[0]])
print("\neigval_0-: ", val[idx_s[1]], "\nomega_0-:\n",vect[:, idx_s[1]])

#plot arc of where eignvalues of omega_0+ and omega_0- lie
x = 1 - (2/(3*n))
y = ((1 - x**2)**0.5)
start = np.degrees(np.arctan(-y/x))
end = np.degrees(np.arctan(y/x))
eigval_range = mpl.patches.Arc((0,0), height=2 , width=2, angle=0, theta1=start, theta2=end)
plt.gca().add_patch(eigval_range)

#plot the eign values
plt.plot(val.real, val.imag, 'ro')
plt.plot(val_u.real, val_u.imag, 'bo')
plt.text(val[idx_s[0]].real, val[idx_s[0]].imag, 0)
plt.text(val[idx_s[1]].real, val[idx_s[1]].imag, 1)

plt.show()
