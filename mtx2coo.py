from scipy.io import mmread
import numpy as np
from scipy import sparse
A = mmread('bwm200.mtx')
# Coo = sparse.coo_matrix(A.toarray())
B = A.toarray()
# print(B.shape)
cnt = 0
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        if B[i][j] != 0:
            print('cooRowIndexHostPtr[' + str(cnt) +']=' + str(i) + '; cooColIndexHostPtr[' + str(cnt) + ']=' + str(j) + '; cooValHostPtr[' + str(cnt) + ']=' + str(B[i][j]) + ';')
            cnt += 1
