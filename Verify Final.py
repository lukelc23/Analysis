import numpy as np
import matplotlib.pyplot as plt
import Ranking_exp_test 

#check final formula

sim = Ranking_exp_test(n=5, k_o=.4, k_s=1, k_d=0, p=4, q=2, c_var=0) 
    # uses rank_mult_exp_analytical
    # uses "previous" form
    # rank = rank_til(j) + c_analytical_form() * alpha_prime * (D_ij_analytical_sol(j, q) - D_ij_analytical_sol(j, p) + delta(j,q) - delta(j,p))
ranks = sim.calc_rank()   
x_vals = np.arange(1, n)
plt.plot(x_vals, ranks, label=f'n = 5')