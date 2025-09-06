import sympy as sp
from sympy import trigsimp, cosh, sinh, tanh, symbols

lamb = symbols('lamb_val')
n = symbols('n')
p = symbols('p')
q = symbols('q')


sp.trigsimp(2 * cosh( ( (n+1/2) - (p/2) ) * lamb  ) * sinh(p * lamb) - 
         2 * cosh( ( (n/2) - (p/2) ) * lamb  ) * sinh(q * lamb))

expression = 1 + (
    (sp.sinh(((n + 1) / 2 - q) * lamb_val) - sp.sinh(((n + 1) / 2 - p) * lamb_val)) 
    / (sp.sinh(((n + 1) / 2) * lamb_val) - sp.sinh(((n - 1) / 2) * lamb_val))
)

# Simplify the expression if needed
simplified_expression = sp.simplify(expression)

print(simplified_expression)

