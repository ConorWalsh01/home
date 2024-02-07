#!/usr/bin/env python
# coding: utf-8

# # Lab Assignment 3
# 
# ## Name Surname, s1234567
# 
# We consider the system $$\frac{dx}{dt}=x(y-1),\quad \frac{dy}{dt}=4-y^2-x^2.$$

# ## Task 1 (2 marks)
# 
# Use `SymPy` to find the critical points of the system.

# In[1]:


import sympy as sym
sym.init_printing()
from IPython.display import display_latex

x = sym.Function('x')
y = sym.Function('y')
t = sym.symbols('t')
eq1 = sym.Eq(x(t).diff(t),x(t)*(y(t)-1))
eq2 = sym.Eq(y(t).diff(t),4-y(t)**2-x(t)**2)
[eq1, eq2]


# In[2]:


EQS = sym.Matrix([eq1.rhs, eq2.rhs])
CPs = sym.solve(EQS)
CPs


# ## Task 2 (4 marks)
# 
# Give your implementation of the `linearise` function from Lab 3.
# 
# Use this to find linear approximations of the system around the critical points with $x \geq 0$ and $y \geq 0$. Use the output to classify these critical points (use markdown cells and proper reasoning to explain the type of each critical point).

# In[3]:


def lin_matrix(system, vec0):
    X, Y = sym.symbols('X, Y')
    FG = sym.Matrix([system[0].rhs, system[1].rhs]).subs({x(t):X, y(t):Y})
    matJ = FG.jacobian([X, Y])
    return matJ.subs({X:vec0[0], Y:vec0[1]})

def linearise(system, vec0):
    u = sym.Function('u')
    v = sym.Function('v')
    lin_mat = lin_matrix(system, vec0)
    lin_rhs = lin_mat * sym.Matrix([u(t), v(t)])
    linsys = [sym.Eq(u(t).diff(t), lin_rhs[0]),
              sym.Eq(v(t).diff(t), lin_rhs[1])]
    return linsys


# For the critical point $(\sqrt{3},1)$:

# In[4]:


linearise([eq1, eq2], [sym.sqrt(3),1])


# In[5]:


lin1 = lin_matrix([eq1, eq2], [sym.sqrt(3),1])
lin1.eigenvects()


# The eigenvalues are complex with negative real part, so the critical point is a stable focus.

# For the critical point $(0,2)$:

# In[6]:


linearise([eq1, eq2], [0, 2])


# In[7]:


lin2 = lin_matrix([eq1, eq2], [0, 2])
lin2.eigenvects()


# The eigenvalues are real and have unequal signs, so the critical point is a saddle.

# ## Task 3 (4 marks)
# 
# Produce a phase portrait of the system, with trajectories showing the behaviour around all the critical points. A few trajectories are enough to show this behaviour. Use properly-sized arrows to diplay the vector field (the RHS of the ODE). There are some marks allocated to the quality of your figure in this part. Try to keep it illustrative yet not too cluttered.

# In[8]:


# Import packages

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
get_ipython().run_line_magic('matplotlib', 'inline')

# Define vector field
def vField(x,t):
    u = x[0]*(x[1]-1)
    v = 4-x[0]**2-x[1]**2
    return [u,v]

# Plot vector field
X, Y = np.mgrid[-4:4:30j,-4:4:30j]
U, V = vField([X,Y],0)

# define colours for each vector based on their lengths
M = np.hypot(U, V)

fig, ax = plt.subplots(figsize=(10, 7))
ax.quiver(X, Y, U, V, M, scale=1/0.005, pivot = 'mid', cmap = plt.cm.bone)

# Settings for trajectories
# for [0,-2]
ics = [[-1,-2], [1,-2],[-1,-1.8],[1,-1.8]]
durations = [0.6,0.6,1,1]
# for [0,2]
ics.extend([[-0.1,3],[0.1,3],[0.1,1],[-0.1,1]])
durations.extend([2,2,2,2])
# for [sqrt(3),1]
ics.extend([[1,0],[2,2]])
durations.extend([4,4])
# for [-sqrt(3),1]
ics.extend([[-1,0],[-2,2]])
durations.extend([4,4])
# away from CPs
ics.extend([[3,4],[-3,4]])
durations.extend([1,1])
vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))  # colors for each trajectory

# plot trajectories
for i, ic in enumerate(ics):
    t = np.linspace(0, durations[i], 100)
    x = odeint(vField, ic, t)
    ax.plot(x[:,0], x[:,1], color=vcolors[i], label='X0=(%.2f, %.2f)' % (ic[0], ic[1]) )

cps = [[0,-2], [0,2], [np.sqrt(3),1], [-np.sqrt(3),1]]
cp_x = [cp[0] for cp in cps]
cp_y = [cp[1] for cp in cps]
ax.scatter(cp_x, cp_y, color='red', s=40)

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-4,4)
# plt.legend()

plt.show()


# In[ ]:




