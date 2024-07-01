import numpy as np
import matplotlib.pyplot as plt

class TI_B:
    def __init__(self, n, k_o, k_s, p, q, c_var):
        self.n = n
        self.k_o = k_o
        self.k_s = k_s
        self.alpha = 1 - (2 * (self.k_o / self.k_s))
        self.alpha_prime = self.k_o / self.k_s
        self.p = p
        self.q = q
        self.c_var = c_var
        self.lamb_val = np.arccosh((1+self.c_var)/(1-self.alpha))

        self.K = {}
        self.K_exp = {}
        self.A = {}

        self.arr_len = self.n - 1
        self.dual_len = (self.arr_len) * 2
        self.dual_len_exp = (self.arr_len) * 2 + 2
        self.a = np.zeros(self.dual_len)
        self.a_exp = np.zeros(self.dual_len_exp)
        self.b = np.zeros(self.arr_len)
        self.b_bar = np.zeros(self.arr_len)
        self.c = np.zeros(1)
        self.c_bar = np.zeros(1)

    def tridiag(self, n):
        arr_len = n-1
        diag_len = arr_len - 1
        B = np.diag(np.ones((diag_len)), k=1) + np.diag(np.ones((diag_len)), k=-1)
        return B        

    def k_create(self):
        arr_len = self.n-1
        A = np.identity(arr_len)
        D = A
        B = self.alpha_prime*self.tridiag(self.n)
        C = B
        E = np.block([[A,B],[B,A]])
        return E

    def y_create(n):
        A = np.ones(n-1)
        B = A * -1
        D = np.concatenate([A, B])
        return D

    def a_create(n, alpha_prime):
        K = k_create()
        y = y_create(n)
        a = np.linalg.solve(K, y)
        return a