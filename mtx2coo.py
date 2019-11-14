from scipy.io import mmread
A = mmread('bwm200.mtx')
print(A.toarray())