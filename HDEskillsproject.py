#!/usr/bin/env python
# coding: utf-8

# # Honours Differential Equations
# ## Project Assignment
# 
# Due: Friday 2nd December 2022, noon
# 
# # Conor Walsh
# # s1949139

# **In the cell below I have listed all of the required packages to complete the tasks:**

# In[1]:


import sympy as sym
sym.init_printing()
from IPython.display import display_latex
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.integrate import odeint
import math
from math import sqrt
from pandas import DataFrame
import sympy.plotting as sym_plot
from sympy import *
from mpl_toolkits.mplot3d import axes3d


# ## Question 1:

# **Question 1a:**

# In[5]:


# Define the equations using appropriate functions and symbols for x, y, a and t:

x = sym.Function('x')
y = sym.Function('y')
t = sym.symbols('t')
a = sym.symbols('a')

# Define the equations for dx/dt and dy/dt as Fxy and Gxy:

Fxy = y(t)
Gxy = -x(t) + a*(y(t) - (y(t)**3)/3)

# Place the equations into a Sympy matrix to which we can apply the "jacobian" function: 

FGmat = sym.Matrix([Fxy,Gxy])

# Compute the Jacobian to find the linearisation of the system:

matJ = FGmat.jacobian([x(t), y(t)])

# Substitute the values of the critical point(0,0) into the Jacobian to find the linearisation around the origin:

matJatCP = matJ.subs(x(t),0).subs(y(t),0)
print("The linearisation of the system about the origin is:")
display_latex(matJatCP)

# Display the eigenvalues and eigenvectors of the system linearised around the origin:
print("The eigenvalues and their corresponding eigenvectors are:")
display_latex(matJatCP.eigenvects())


# **Written discussion for 1a:**
# 
# For our obtained linearisation of the system around the critical point $(0, 0)$, we have the following categories relating to the behaviour of the critical point as the value of $a$ varies:
# 
# For $a < -2$, our eigenvalues $\lambda_1$ and $\lambda_2$ are negative, real and unequal. Hence the critical point will be a Nodal Sink which is asymptotically stable.
# 
# For $a = -2$, our eigenvalues $\lambda_1$ and $\lambda_2$ are equal and negative, and thus have algebraic multiplicity $2$. Hence the critical point will be an Improper Node as there is only one linearly independent eigenvector. Furthermore, the critical point is asymptotically stable.
# 
# For values of $a$ on the interval $-2 < a < 0$, our eigenvalues $\lambda_1$ and $\lambda_2$ are complex. For values of $a$ on this interval, our eigenvalues are complex with non-vanishing negative real parts. Hence the critical point will be a Spiral for these values of $a$. Therefore the critical point will be asymptotically stable if $a$ is negative.
# 
# For values of $a$ on the interval $0 < a < 2$, our eigenvalues $\lambda_1$ and $\lambda_2$ are also complex. For values of $a$ on this interval, our eigenvalues are complex with non-vanishing positive real parts. Hence the critical point will be a Spiral for such $a$. Therefore the critical point will be unstable if $a$ is positive.
# 
# For $a = 0$, our eigenvalues $\lambda_1$ and $\lambda_2$ are equal to $\pm{i}$, representing complex eigenvalues with vanishing real parts. Hence the critical point will be a Center, with stable behaviour.
# 
# For $a = 2$, our eigenvalues $\lambda_1$ and $\lambda_2$ are equal and positive, and thus have algebraic multiplicity $2$. Hence the critical point will be an Improper Node, as there is only one linearly independent eigenvector, and the critical point is unstable.
# 
# For $a > 2$, our eigenvalues $\lambda_1$ and $\lambda_2$ are positive, real and unequal. Hence the critical point will be a Nodal Source and the critical point will be unstable.

# **Question 1b:**

# In[6]:


# For the first phase portrait, set a = 0.1:

a = 0.1

# Define a vector field:

def vField(x,t):
    u = x[1]
    v = -x[0] + a*(x[1] - (x[1]**3)/3)
    return [u,v]

# Plot the vector field:
X, Y = np.mgrid[-6:6:30j, -6:6:30j]
U, V = vField([X,Y],0)

# Define colours for each vector based on their lengths:
M = np.hypot(U, V)

# Create a figure for the plot:

fig, ax = plt.subplots(figsize=(10, 7))
ax.quiver(X, Y, U, V, M, scale=1/0.005, pivot = 'mid', cmap = plt.cm.bone)

# Settings for trajectories:
# for [0,0]

ics = [[0.1,0.1], [2,2]]
durations = [80,80]

# Set the colors for each trajectory:

vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))

# Plot the trajectories:

for i, ic in enumerate(ics):
    t = np.linspace(0, durations[i], 1000)
    x = odeint(vField, ic, t)
    ax.plot(x[:,0], x[:,1], color=vcolors[i], label='X0=(%.f, %.f)' % (ic[0], ic[1]) )

# Create a scatter plot of the trajectories:    

ax.scatter([0],[0], color='red', s=40)

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.title("Phase portrait of the system for a = 0.1:", fontsize = 15)

plt.show()


# In[7]:


# For the second phase portrait, set a = -0.1:

a = -0.1

# Plot the vector field:
X, Y = np.mgrid[-6:6:30j, -6:6:30j]
U, V = vField([X,Y],0)

# Define colours for each vector based on their lengths:
M = np.hypot(U, V)

# Create a figure for the plot:

fig, ax = plt.subplots(figsize=(10, 7))
ax.quiver(X, Y, U, V, M, scale=1/0.005, pivot = 'mid', cmap = plt.cm.bone)

# Settings for trajectories:
# for [0,0]
ics = [[0.1,0.1], [2,2]]
durations = [6, 6]

# Set the colors for each trajectory:

vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))

# Plot the trajectories:

for i, ic in enumerate(ics):
    t = np.linspace(0, durations[i], 100)
    x = odeint(vField, ic, t)
    ax.plot(x[:,0], x[:,1], color=vcolors[i], label='X0=(%.f, %.f)' % (ic[0], ic[1]) )

# Create a scatter plot of the trajectories:    

ax.scatter([0],[0], color='red', s=40)

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-6,6)
plt.ylim(-6,6)
plt.title("Phase portrait of the system for a = -0.1:", fontsize = 15)

plt.show()


# **Written discussion for 1b:**
# 
# In both of the phase portraits shown where $a = 0.1$ and $a = -0.1$ respectively, we can see that for the given initial conditions $(x(0),y(0)) = (0, 0)$ and $(x(0),y(0)) = (2, 2)$, the phase portraits both show the critical point at the origin to be a Spiral.
# 
# This visualisation supports our remarks from 1a) where we discussed the behaviour and stability of the critical point at the origin. 
# 
# In the first phase portrait where $a = 0.1$, we can deduce that the system is unstable at the critical point as the trajectories spiral towards the origin.
# 
# In the second phase portrait where $a = -0.1$, we can deduce that the system is asymptotically stable at the critical point as the trajectories spiral away from the origin.

# ## Question 2

# **Question 2a:**

# In[8]:


# Define the equations using appropriate functions and symbols for x, y, a and t:

x = sym.Function('x')
y = sym.Function('y')
t = sym.symbols('t')
a = sym.symbols('a')
b = sym.symbols('b')
eq1 = sym.Eq(x(t).diff(t), a - x(t) - b*x(t) + y(t)*x(t)**2)
eq2 = sym.Eq(y(t).diff(t), b*x(t) - y(t)*x(t)**2)

display_latex([eq1, eq2])

# Define the functions lin_matrix and linearise:

def lin_matrix(system, vec0):
    '''
    Takes in a system of two equations and the co-ordinates of a point, returning the Jacobian matrix of the system
    evaluated at this point
    --------------------------------------------------------------------------------------------------------------
    inputs:
        system: a system of two equations given as a list of SymPy equations
        vec0: the co-ordinates of a point, given as a list
    outputs:
        matJ.subs({X:vec0[0], Y:vec0[1]}): The Jacobian matrix of the system evaluated for the corresponding
        critical point
    '''
    X, Y = sym.symbols('X, Y')
    FG = sym.Matrix([system[0].rhs, system[1].rhs]).subs({x(t):X, y(t):Y})
    matJ = FG.jacobian([X, Y])
    return matJ.subs({X:vec0[0], Y:vec0[1]})

def linearise(system, vec0):
    '''
    Takes in the same inputs as lin_matrix, returning a system of linear equations in u and v
    --------------------------------------------------------------------------------------------------------------
    inputs:
        system: a system of two equations given as a list of SymPy equations
        vec0: the co-ordinates of a point, given as a list
    outputs:
        linsys: a system of linear equations in u and v
    '''
    u = sym.Function('u')
    v = sym.Function('v')
    lin_mat = lin_matrix(system, vec0)
    lin_rhs = lin_mat * sym.Matrix([u(t), v(t)])
    linsys = [sym.Eq(u(t).diff(t), lin_rhs[0]),
              sym.Eq(v(t).diff(t), lin_rhs[1])]
    return linsys

# Enter the given system of equations and place it into a list:

project_q2a = [sym.Eq(x(t).diff(t),- x(t) - b*x(t) + a + y(t)*x(t)**2),
                  sym.Eq(y(t).diff(t), - y(t)*x(t)**2 + b*x(t))]

# Find the critical point and display it in terms of a and b:

EQS = sym.Matrix([project_q2a[0].rhs, project_q2a[1].rhs])
CPs = sym.solve(EQS, (x(t), y(t)))

# Find the linearisation and show the details, including the critical point, u'(t), v'(t) and the eigenvalues
# and eigenvectors of the system as expressions in terms of a and b:

all_linsys = []
for CP in CPs:
    print("The critical point of the system is:")
    display_latex(CP)
    vec0 = list(CP)
    print("For the critical point"+str(vec0))
    print("We have the following linearised system:")
    linmat = lin_matrix(project_q2a,vec0)
    display_latex(linmat)
    print("Where:")
    linsys = linearise(project_q2a,vec0)
    display_latex(linsys)
    print("The corresponding eigenvalues and eigenvectors are:")
    display_latex(list(linmat.eigenvects()))


# **Question 2b:**

# In[9]:


# Fix the parameter a = 1 for the question:

a = 1

# For the first phase portrait of the system, set b = 0.5:

b = 0.5

# Define a function for the vector field:

def vField(x,t):
    u = a - x[0] - b*x[0] + x[1]*x[0]**2
    v = b*x[0] - x[1]*x[0]**2
    return [u,v]

# Plot the vector field:

X, Y = np.mgrid[0:5:30j, 0:5:30j]
U, V = vField([X,Y],0)

# Define colours for each vector based on their lengths:

M = np.hypot(U, V)

# Create a plot:

fig, ax = plt.subplots(figsize=(10, 7))
ax.quiver(X, Y, U, V, M, scale=1/0.0015, pivot = 'mid', cmap = plt.cm.bone)

# Establish the trajectories and initial conditons for the critical point:

# For the critical point (a, b/a), with our given values of a and b the critical point is (1, 0.5):

ics = [[0,0], [2,3]]
durations = [15, 15, 15, 15]

vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))

# Plot the trajectories:

for i, ic in enumerate(ics):
    t = np.linspace(0, durations[i], 1000)
    x = odeint(vField, ic, t)
    ax.plot(x[:,0], x[:,1], color=vcolors[i], label='X0=(%.2f, %.2f)' % (ic[0], ic[1]) )
    
# Create a scatter plot for the critical point:    

cps = [[a, b/a]]
cp_x = [cp[0] for cp in cps]
cp_y = [cp[1] for cp in cps]
ax.scatter(cp_x, cp_y, color='red', s=40)

# Set labels and limits for the x and y axes:

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,5)
plt.ylim(0,5)

# Set a title for the phase portrait:

plt.title('Phase Portrait For b = 0.5', fontsize = 20)
#plt.legend()

# Show the plot:

plt.show()


# In[10]:


# For the second phase portrait of the system, set b = 3:

b = 3

# Define a function for the vector field:

def vField(x,t):
    u = a - x[0] - b*x[0] + x[1]*x[0]**2
    v = b*x[0] - x[1]*x[0]**2
    return [u,v]

# Plot the vector field:

X, Y = np.mgrid[0:5:30j, 0:5:30j]
U, V = vField([X,Y],0)

# Define colours for each vector based on their lengths:

M = np.hypot(U, V)

# Create a plot:

fig, ax = plt.subplots(figsize=(10, 7))
ax.quiver(X, Y, U, V, M, scale=1/0.0015, pivot = 'mid', cmap = plt.cm.bone)

# Establish the trajectories and initial conditions for the critical point:

# For the critical point (a, b/a), with our given values of a and b the critical point is (1,3):

ics = [[0,0], [2,3]]
durations = [15, 15]

# Set the colour of the trajectories:

vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))

# Plot the trajectories:

for i, ic in enumerate(ics):
    t = np.linspace(0, durations[i], 10000)
    x = odeint(vField, ic, t)
    ax.plot(x[:,0], x[:,1], color=vcolors[i], label='X0=(%.2f, %.2f)' % (ic[0], ic[1]) )
    
# Create a scatter plot for the critical point:    

cps = [[a, b/a]]
cp_x = [cp[0] for cp in cps]
cp_y = [cp[1] for cp in cps]
ax.scatter(cp_x, cp_y, color='red', s=40)

# Set labels and limits for the x and y axes:

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,5)
plt.ylim(0,5)

# Set a title for the phase portrait:

plt.title('Phase Portrait For b = 3', fontsize = 20)
#plt.legend()

# Show the plot:

plt.show()


# **Written discussion for Question 2b:**
# 
# The difference in behaviour between the two plots is found to be related to periodic vs non-periodic behaviour when the value of $b$ varies.
# 
# For the first phase portrait in which $b = 0.5$, the produced phase portrait shows non-perodic behaviour around the corresponding critical point $(1, 0.5)$, whereas in the second phase portrait in which $b = 3$, the produced phase portrait exhibits periodic behaviour around the corresponding critical point $(1, 3)$.

# **Question 2c:**

# In[11]:


# We will plot two new phase portraits with new values of b above and below the critcal value, 
# to illustrate the bifurication effect:

# In this code cell we set b = 1.5:

b = 1.5

# Define a function for the vector field:

def vField(x,t):
    u = a - x[0] - b*x[0] + x[1]*x[0]**2
    v = b*x[0] - x[1]*x[0]**2
    return [u,v]

# Plot the vector field:

X, Y = np.mgrid[0:5:30j, 0:5:30j]
U, V = vField([X,Y],0)

# Define colours for each vector based on their lengths:

M = np.hypot(U, V)

# Create a plot:

fig, ax = plt.subplots(figsize=(10, 7))
ax.quiver(X, Y, U, V, M, scale=1/0.0015, pivot = 'mid', cmap = plt.cm.bone)

# Establish the trajectories and initial conditions for the critical point:

# For the critical point (a, b/a), with our given values of a and b the critical point is (1,1.5):

ics = [[0,0], [2,3]]
durations = [50, 50]

# Set the colour of the trajectories:

vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))

# Plot the trajectories:

for i, ic in enumerate(ics):
    t = np.linspace(0, durations[i], 1000)
    x = odeint(vField, ic, t)
    ax.plot(x[:,0], x[:,1], color=vcolors[i], label='X0=(%.2f, %.2f)' % (ic[0], ic[1]) )
    
# Create a scatter plot for the critical point:    

cps = [[a, b/a]]
cp_x = [cp[0] for cp in cps]
cp_y = [cp[1] for cp in cps]
ax.scatter(cp_x, cp_y, color='red', s=40)

# Set labels and limits for the x and y axes:

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,5)
plt.ylim(0,5)

# Set a title for the phase portrait:

plt.title('Phase Portrait For b = 1.5', fontsize = 20)
#plt.legend()

# Show the plot:

plt.show()


# In[12]:


# In this code cell we set b = 2.5:

b = 2.5

# Define a function for the vector field:

def vField(x,t):
    u = a - x[0] - b*x[0] + x[1]*x[0]**2
    v = b*x[0] - x[1]*x[0]**2
    return [u,v]

# Plot the vector field:

X, Y = np.mgrid[0:5:30j, 0:5:30j]
U, V = vField([X,Y],0)

# Define colours for each vector based on their lengths:

M = np.hypot(U, V)

# Create a plot:

fig, ax = plt.subplots(figsize=(10, 7))
ax.quiver(X, Y, U, V, M, scale=1/0.0015, pivot = 'mid', cmap = plt.cm.bone)

# Establish the trajectories and initial conditons for the critical point:

# For the critical point (a, b/a), with our given values of a and b the critical point is (1,2.5):

ics = [[0,0], [2,3]]
durations = [50, 50]

# Set the colour of the trajectories:

vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))

# Plot the trajectories:

for i, ic in enumerate(ics):
    t = np.linspace(0, durations[i], 1000)
    x = odeint(vField, ic, t)
    ax.plot(x[:,0], x[:,1], color=vcolors[i], label='X0=(%.2f, %.2f)' % (ic[0], ic[1]) )
    
# Create a scatter plot for the critical point:    

cps = [[a, b/a]]
cp_x = [cp[0] for cp in cps]
cp_y = [cp[1] for cp in cps]
ax.scatter(cp_x, cp_y, color='red', s=40)

# Set labels and limits for the x and y axes:

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,5)
plt.ylim(0,5)

# Set a title for the phase portrait:

plt.title('Phase Portrait For b = 2.5', fontsize = 20)
#plt.legend()

# Show the plot:

plt.show()


# **Written Discussion for Question 2c:**
# 
# Upon calculation of the eigenvalues for different values of $b$, we reach the conclusion that the critical value is given as $b = 2$, where the eigenvalues $\lambda_1$, $\lambda_2$ are equal to $\pm$ $2i$ and the critical point $(a, \frac{b}{a})$ is a Center of indeterminate stability. I have produced two new phase portraits above to illustrate the aforementioned bifurcation using $a = 1$ and two equally spaced values of $b$ both above and below the critical value, to indicate that this bifurcation occurs at $b = 2$ and how this relates to the new values of $b$ on the phase portraits.
# 
# For $b = 1.5$, we are able to determine that the eigenvalues $\lambda_1$, $\lambda_2$ = $-0.25$ $\pm$ $0.96824i$, which indicates that the eigenvalues are complex with non-vanishing, negative real parts. Hence the critical point $(1, 1.5)$ is a spiral with unstable focus, as shown on the corresponding phase portrait above.
# 
# For $b = 2.5$, we are able to determine that the eigenvalues $\lambda_1$, $\lambda_2$ = $0.25$ $\pm$ $0.96824i$, which indicates that the eigenvalues are complex with non-vanishing, positive real parts. Hence the critical point $(1, 2.5)$ is a spiral with stable focus, as shown on the corresponding phase portrait above.
# 
# To summarise, the behaviour of the eigenvalues changes significantly for assigned values of $b$ both above and below the critical value of $b = 2$. As a result, these changes will also go on to affect the behaviour of the critical point and its stability in the linearisation for such chosen values of $b$.

# **Question 2d:**

# In[13]:


# Begin solution by setting a = 1 and b = 3 for periodic behaviour:

a = 1

b = 3

# Define a function dX_dt which returns dx/dt and dy/dt which we can solve using scipy.odeint:

def dX_dt(X, t):
    x, y = X
    return [a - x - b*x + y*x**2, b*x - y*x**2]

# Set up the time samples:

t = np.linspace(0, 50, 1000)

# Set the initial condition (x(0), y(0)) = (0,0):

X0 = [0, 0] 

# Solve for x(t) and y(t) as functions of t using odeint:

Xsol = odeint(dX_dt, X0, t)

# Create a figure:

fig, ax = plt.subplots(1, 2, figsize = (11, 5))

# Plot x(t) and y(t) against t as functions of t, onto two different subplots:

ax[0].plot(t, Xsol[:, 0])
ax[1].plot(t, Xsol[:, 1])

# Label the axes of each subplot:

ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$x(t)$')

ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$y(t)$')

# Set titles for each subplot:

ax[0].set_title('Plot of x(t) against t')
ax[1].set_title('Plot of y(t) against t')

# Show the plot:

plt.show()


# ## Question 3

# **Question 3a:**

# In[14]:


# Define a function which will carry out the modified Euler method:

def ModifiedEuler(func, times, y0):
    '''
    integrates the system of y' = func(y, t) using forward Euler method
    for the time steps in times and given initial condition y0
    ----------------------------------------------------------
    inputs:
        func: the RHS function in the system of ODE
        times: the points in time (or the span of independent variable in ODE)
        y0: initial condition (make sure the dimension of y0 and func are the same)
    output:
        y: the solution of ODE. 
        Each row in the solution array y corresponds to a value returned in column vector t
        times: the points in time converted to a numpy array:
    '''
    # Converts the points in time, "times" and initial condition y(0) to numpy arrays:
    
    times = np.array(times)
    y0 = np.array(y0)
    
    # Finds the dimension of the ODE:
    
    n = y0.size 
    
    # Gives the number of time steps:
    
    nT = times.size 
    y = np.zeros([nT,n])
    y[0, :] = y0
    # Loop for the timesteps:
    for k in range(nT-1):
        h = (times[k+1] - times[k])
        y[k+1, :] = y[k, :] + h*func(times[k] + 0.5*h, y[k, :] + 0.5*h*func(times[k], y[k, :]))
    return y, times


# **Question 3b:**

# In[15]:


# Define a function which returns the RHS of the ODE as a function f(t, y):

def q3b_dy_dt(t, y):
    return 5*t - 2*np.sqrt(y)

# Define a function timesteps which generates an array of numbers,
# starting at the value "start" and increasing with step size "h" until the value "stop" is reached:

def timesteps(start, stop, h):
    '''
    Creates the "times" required for each of our approximation methods
    ---------------------------------------------------------------------------------------------------------------
    input:
        start: beginning of our interval over the parameter t
        stop: end of the interval over the parameter t
        h: step size
    output:
       np.linspace(start, start+num_steps*h, num_steps+1): a numpy array of numbers beginning from "start" 
       and increasing with step size "h" until "stop" is reached at last
    '''
    num_steps = math.ceil((stop - start)/h)
    return np.linspace(start, start+num_steps*h, num_steps+1)

# Define timestep functions for the modified Euler method described:

def ModifiedEuler_step(func, start, stop, h, ics):
    '''
    Solves the ODE numerically for the modified Euler method, returning the values of the numerical
    solution and the list of time steps
    ---------------------------------------------------------------------------------------------------------------
    input:
        func - a function defining the differential equation
        start - the initial time
        stop - the end time
        h - the step size
        ics - the intial conditions
    output:
        values- the values for each numerical solution
        times- the list of time steps
    '''
    times = timesteps(start, stop, h)
    values, times = ModifiedEuler(func, times, ics)
    return values, times

# Define a function produce_df to create a table of solutions:

def produce_df(method, func, start, stop, h, ics):
    '''
    Takes an approximation method as an input and returns a dataframe of having applied said method with our
    ODE above
    --------------------------------------------------------------------------------------------------------------
    input:
        method- the timestep function for the corresponding approximation method
        func - a function defining the differential equation
        start - the initial time
        stop - the end time
        h - the step size
        ics - the intial conditions
    output:
        Dataframe showing the result of applying the corresponding approximation method
    '''
    values, times = method(func, start, stop, h, ics)
    return DataFrame(data = values, index = np.round(times, 3), columns = ["h=0.05"])

# Use the function to produce a dataframe detailing the values for the modified Euler method, using the
#alternate approach:

df_q3b = produce_df(ModifiedEuler_step, q3b_dy_dt, 0, 1, 0.05, 2)

# Label the column "Modified Euler, h = 0.05":

df_q3b.columns = ["Modified Euler, h=0.05"]

# Display the table, filtering for the values of the solution at t = 0.1, 0.2, 0.3, 0.4:

print("The following table details the approximate values of the solution to the IVP, for our chosen values of t:")

display(df_q3b.filter(items = [0.1, 0.2, 0.3, 0.4], axis = 0))


# **Question 3c:**

# In[16]:


# Define t as a symbol and y as a function:

t = sym.symbols('t')
y = sym.Function('y')

# Solve the ODE in sympy using the hint "best" in sym.dsolve:

exact = sym.Eq(y(t).diff(t), 5*t - 2*sym.sqrt(y(t)))
exact_sol = sym.dsolve(exact, y(t), ics={y(0):2}, hint = "best")

# Display the solution:

print("The exact solution of the IVP is:")

display_latex(exact_sol)


# **Question 3d:**

# In[17]:


# To start, define a new function which returns the first 6 terms from the exact solution of the IVP:

def exact_soln(t):
    '''
    Takes in t, the parameter of the ODE and returns the particular solution to our ODE given our initial condition
    ---------------------------------------------------------------------------------------------------------------
    input:
        t: parameter of the ODE
    output:
        particular solution y(t) for initial condition y(0) = 2
    '''
    return 2 - 2*t*np.sqrt(2) + (7/2)*t**2 - (5/12)*np.sqrt(2)*t**3 - (5/24)*t**4 + (np.sqrt(2)/64)*t**5

# Define a function for the forward Euler method:

def ode_Euler(func, times, y0):
    '''
    integrates the system of y' = func(y, t) using forward Euler method
    for the time steps in times and given initial condition y0
    ----------------------------------------------------------
    inputs:
        func: the RHS function in the system of ODE
        times: the points in time (or the span of independent variable in ODE)
        y0: initial condition (make sure the dimension of y0 and func are the same)
    output:
        y: the solution of ODE. 
        Each row in the solution array y corresponds to a value returned in column vector t
    '''
    
    # Converts the points in time, "times" and initial condition y(0) to numpy arrays:
    
    times = np.array(times)
    y0 = np.array(y0)
    
    # Finds the dimension of the ODE:
    
    n = y0.size
    
    # Gives the number of time steps:
    
    nT = times.size    
    y = np.zeros([nT,n])
    y[0, :] = y0
    
    # Loop for timesteps:
    
    for k in range(nT-1):
        y[k+1, :] = y[k, :] + (times[k+1]-times[k])*func(times[k], y[k, :])
    return y

# Create a timestep function for the Euler method:

def Euler_step(func, start, stop, h, ics):
    '''
    Solves the ODE numerically for the Forward Euler method, returning the values of the numerical
    solution and the list of time steps
    ---------------------------------------------------------------------------------------------------------------
    input:
        func - a function defining the differential equation
        start - the initial time
        stop - the end time
        h - the step size
        ics - the intial conditions
    output:
        values- the values for each numerical solution
        times- the list of time steps
    '''
    times = timesteps(start, stop, h)
    values = ode_Euler(func, times, ics)
    return values, times

# Create a new table which includes the values of the solutions for t = 0.1, 0.2, 0.3, 0.4, 0.5, 1.0:

# This table will also contain the values of the solutions for the modified Euler method, as well as the exact
# values of the solution:

df_q3d = produce_df(Euler_step, q3b_dy_dt, 0, 1, 0.05, 2)

# Create the columns corresponding to each method:

df_q3d.columns = ["Euler, h = 0.05"]

df_q3d["Modified Euler, h = 0.05"] = produce_df(ModifiedEuler_step, q3b_dy_dt, 0, 1, 0.05, 2)

df_q3d["Exact, h = 0.05"] = DataFrame(data = [exact_soln(t) for t in timesteps(0, 1, 0.05)],
                                      index = np.round(timesteps(0, 1, 0.05), 3))

# Display the table showing the values of the solutions for t = 0.1, 0.2, 0.3, 0.4, 0.5, 1.0:
                                                                    
display(df_q3d.filter(items = [0.1, 0.2, 0.3, 0.4, 0.5, 1], axis = 0))

# Display the plot of the solutions for 0 ≤ t ≤ 1:

display(df_q3d.plot())


# Write your written solution here.

# ## Question 4

# Please **clearly** indicate where you answer each sub question by using a markdown cell.

# **Question 4a:**

# In[18]:


# Define a function for the vector field:

def vField(x,t):
    u = -x[1] - x[2]
    v = x[0] + 1/5 * x[1]
    w = 1/5 + (x[0] - 5/2)*x[2]
    return [u,v,w]

# Create a 3D plot:

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection = '3d')

ics = [[0,0,0]]
duration = 500

# Set the colour of the trajectories:

vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))

# Plot the trajectories:

for i, ic in enumerate(ics):
    t = np.linspace(0, duration, 10000)
    x = odeint(vField, ic, t)
    ax.plot(x[:,0], x[:,1], x[:, 2], color=vcolors[i], label='X0=(%.2f, %.2f, %.2f)' % (ic[0], ic[1], ic[2]) )
    

# Set labels and limits for the x and y axes:

plt.xlabel('x(t)')
plt.ylabel('y(t)')

# Set a title for the phase portrait and add a legend:

plt.title('3D Phase Portrait Of The System:', fontsize = 20)
plt.legend()

# Show the plot:

plt.show()


# **Question 4b:**

# In[19]:


# Set new times t_b and new solutions obtained from odeint x_b for plotting:

t_b = np.linspace(0, 100, 10000)
x_b = odeint(vField, [0,0,0], t_b)


# Create a figure:

fig, ax = plt.subplots(1, 2, figsize = (11, 5))

# Plot x(t) and y(t) as functions of t onto two different subplots:

ax[0].plot(t_b, x_b[:, 0])
ax[1].plot(t_b, x_b[:, 1])

# Label the axes of each subplot and set limits for the x-axis:

ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$x(t)$')


ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$y(t)$')


# Set titles for each subplot:

ax[0].set_title('Plot of limit cycle for x(t)')
ax[1].set_title('Plot of limit cycle for y(t)')

# Show the plot:

plt.show()

# To show that the solution is periodic, we plot x(t) and y(t) for suitable values of t:

# Create a figure:

fig, ax = plt.subplots(1, figsize = (11, 5))

# Plot x(t) and y(t) as functions of t onto two different subplots:

ax.plot(x_b[:, 0], x_b[:, 1])
ax.set_title("Plot of solutions for x(t), y(t):", fontsize = 20)

# Label the axes:

ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")

# Show the plot:

plt.show()


# In[20]:


# To show that the solution is periodic, we plot x(t) and y(t) for suitable values of t:

# Create a figure:

fig, ax = plt.subplots(1, figsize = (11, 5))

# Plot x(t) and y(t) as functions of t onto two different subplots, indexing for values of t at which the limit
# cycle displays periodic behaviour:

ax.plot(x_b[-1000:, 0], x_b[-1000:, 1])
ax.set_title("Plot detailing periodic behaviour of x(t), y(t):", fontsize = 20)

# Label the axes:

ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")

# Show the plot:

plt.show()


# **Question 4c:**

# In[21]:


# Define a function for the vector field, this time replacing 5/2 with 3 in the third equation:

def vField(x,t):
    u = -x[1] - x[2]
    v = x[0] + 1/5 * x[1]
    w = 1/5 + (x[0] - 3)*x[2]
    return [u,v,w]

# Create a 3D plot:

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection = '3d')

ics = [[0,0,0]]
duration = 500

# Set the colour of the trajectories:

vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))

# Plot the trajectories:

for i, ic in enumerate(ics):
    t = np.linspace(0, duration, 10000)
    x = odeint(vField, ic, t)
    ax.plot(x[:,0], x[:,1], x[:, 2], color=vcolors[i], label='X0=(%.2f, %.2f, %.2f)' % (ic[0], ic[1], ic[2]) )
    

# Set labels and limits for the x and y axes:

plt.xlabel('x(t)')
plt.ylabel('y(t)')

# Set a title for the phase portrait and add a legend:

plt.title('Updated 3D Phase Portrait Of The System:', fontsize = 20)
plt.legend()

# Show the plot:

plt.show()


# In[22]:


# Now we plot the updated plot for the limit cycle:

# Create a figure:

fig, ax = plt.subplots(1, figsize = (11, 5))

# Plot x(t) and y(t) as functions of t onto two different subplots:

ax.plot(x[:, 0], x[:, 1])
ax.set_title("Updated plot of solutions for x(t), y(t):", fontsize = 20)

# Label the axes:

ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")

# Show the plot:

plt.show()

# Now we plot the updated plot for the limit cycle:

# Create a figure:

fig, ax = plt.subplots(1, figsize = (11, 5))

# Plot x(t) and y(t) as functions of t onto two different subplots for :

ax.plot(x[-1000:, 0], x[-1000:, 1])
ax.set_title("Updated plot detailing periodic behaviour of x(t), y(t):", fontsize = 20)

# Label the axes:

ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")

# Show the plot:

plt.show()


# In[23]:


# Now we seek to plot the updated limit cycles from the new phase portrait:

# Set new times t_c and new solutions obtained from odeint, x_c for plotting:

t_c = np.linspace(0, 100, 10000)
x_c = odeint(vField, [0,0,0], t_c)

# Create a figure:

fig, ax = plt.subplots(1, 2, figsize = (11, 5))

# Plot x(t) and y(t) as functions of t onto two different subplots:

ax[0].plot(t_c, x_c[:, 0])
ax[1].plot(t_c, x_c[:, 1])

# Label the axes of each subplot and set limits for the x-axis:

ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$x(t)$')


ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$y(t)$')


# Set titles for each subplot:

ax[0].set_title('Updated plot of limit cycle for x(t)')
ax[1].set_title('Updated plot of limit cycle for y(t)')

# Show the plot:

plt.show()


# **Discussion for Question 4c:**
# 
# In Question 4c we have produced a new 3D phase portrait but the vale of c has changed from $5/2$ to $3$. As a result, this leads to a difference in behaviour between the 3D phase portraits.
# 
# In the first phase portrait, we have a shorter period, whereas in the second phase portrait we can see that the period is clearly longer, which is supported by my updates plots for the limit cycles of $x(t)$ and $y(t)$, as well as the plot detailing periodic behaviour of $x(t)$ and $y(t)$. This indicates that the limit cycles plotted in $c)$ converge to a greater limit than those plotted in $b)$.
