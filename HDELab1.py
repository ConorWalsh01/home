#!/usr/bin/env python
# coding: utf-8

# # Lab Assignment 1
# 
# 

# ### Conor Walsh, s1949139

# ## Task 1
# 
# Use `SymPy` to solve the differential equation $y' = (y-y^2)\cos(x)$, with $y(0)=0.5$, and plot the solution.

# In[8]:


# Use the standard setup to import the packages we need
import sympy as sym
sym.init_printing()
from IPython.display import display_latex
import sympy.plotting as sym_plot
from sympy import cos
from sympy import *

# Define the differential equation in terms of x and y
x = sym.symbols('x')
y = sym.Function('y')

eq1 = sym.Eq(y(x).diff(x), (y(x) - y(x)**2) * sym.cos(x))
eq1_sol = sym.dsolve(eq1, y(x), ics={y(0):1/2})

print("The equation")
display_latex(eq1)

print("has solution\n")
display_latex(eq1_sol)

print("\nwhich looks like")
sym_plot.plot(eq1_sol.rhs, (x,0,10),xlabel = 'x', ylabel = 'y', line_color = 'black', legend = True)


# ## Task 2
# 
# Use `SciPy`'s `odeint` function to solve the system of equations
# 
# \begin{align*}
# \frac{dx}{dt} &= 0.4x - 0.8xy \\ \frac{dy}{dt}&=xy -1.2y
# \end{align*}
# 
# Produce a plot of the solutions for $0\leq t\leq 15$ with initial conditions $x(0)=0.5$ and $y(0)\in\{0.25, 0.75, 1.25, \ldots, 3.25\}$.
# 
# How many curves do you expect to see plotted?

# ### We should expect to see 7 curves plotted.

# In[17]:


# Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
get_ipython().run_line_magic('matplotlib', 'notebook')

# Begin solution
def dX_dt(X, t):
    x, y = X
    return [0.4*x -0.8*x*y, x*y - 1.2*y]

# Set up the time samples
t = np.linspace(0, 15, 1000)

# Set a loop for the various initial conditions of y
for y0 in np.arange(1, 8)/2 - 0.25:
    # Set within the loop the initial conditon of x
    X0 = [0.5, y0]
    
    # Solve
    Xsol = odeint(dX_dt, X0, t)
    
    # Plot the (x, y) coordinates of the solution
    plt.plot(Xsol[:, 0], Xsol[:, 1])
        
plt.xlabel('x')
plt.ylabel('y')
plt.grid()


# In[ ]:




