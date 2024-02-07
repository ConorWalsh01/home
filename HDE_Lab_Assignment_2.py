#!/usr/bin/env python
# coding: utf-8

# # Lab Assignment 2
# 
# 

# ### Conor Walsh, s1949139

# ## Presentation and coding style (3 marks)
# 
# In this assignment, some marks are allocated to your coding style and presentation. Try to make your code more readable using the tips given in your computer lab 2. Make sure your figures have good quality, right size, good range and proper labels.

# ## Task 1 (4 marks)
# 
# In this task we try to use several method from Lab 2 to solve the initial value problem 
# 
# \begin{equation}
# y' = 3t-4y, \quad y(0)=1,
# \end{equation}
# 
# Set the step size to $h = 0.05$ and numerically solve this ODE from $t=0$ to $0.5$ using the following methods:
# 
# - Forward Euler 
# 
# - Adams–Bashforth order 2
# 
# - Adams–Bashforth order 3 (we did not code this method in the computer lab, but you can find the formula on [this wikipedia page](https://en.wikipedia.org/wiki/Linear_multistep_method)). For this method, you need to build the very first two steps using other methods. For the first step, use the Euler scheme. For the second step, use Adams–Bashforth order 2. 
# 
# 
# Plot the three different approximations, and display the values in a table.

# In[1]:


# Import packages and define functions
import math
import numpy as np
import matplotlib.pyplot as plt

# Euler scheme
def ode_Euler(func, times, y0):
    '''
    integrates the system of y' = func(y, t) using forward Euler method
    for the time steps in times and given initial condition y0
    --------------------------------------------------------------------------------------------------------------
    inputs:
        func: the RHS function in the system of ODE
        times: the points in time (or the span of independent variable in ODE)
        y0: initial condition (make sure the dimension of y0 and func are the same)
    output:
        y: the solution of ODE. 
        Each row in the solution array y corresponds to a value returned in column vector t
    '''
    # Converts the points in time, "times" and initial condition y(0) to numpy arrays.
    times = np.array(times)
    y0 = np.array(y0)
    # Finds the dimension of the ODE
    n = y0.size
    # Finds the number of time steps 
    nT = times.size
    # Finds the size of the time points array
    y = np.zeros([nT,n])
    # Sets the initial condition array
    y[0, :] = y0
    # Loop for timesteps
    for k in range(nT-1):
        y[k+1, :] = y[k, :] + (times[k+1]-times[k])*func(y[k, :], times[k])
        
    return y ,times

# Adams-Bashforth 2 (here needing a fixed timestep)
def ode_AB2(func, initialTime, finalTime, nSteps, y0):
    '''
    integrates the system of y' = func(y, t) using Adams-Bashforth 2nd order
    for the time steps in times and given initial condition y0
    --------------------------------------------------------------------------------------------------------------
    inputs:
        func: the RHS function in the system of ODE
        initialTime: the starting point for time integration for which initial condition is given
        finalTime: the end time integration intervals
        nSteps: number of steps between "initialTime" and "finalTime"
        y0: initial condition (make sure the dimension of y0 and func are the same)
    output:
        y: the solution of ODE. 
        Each row in the solution array y corresponds to a value returned in column vector t
        times: a numpy array consisting of all times at which the solution is given
    '''
    # Converts the initial condition y(0) to a numpy array
    y0 = np.array(y0)
    # Finds number of time steps
    n = y0.size 
    dt = (finalTime - initialTime)/nSteps
    times = np.linspace(initialTime, finalTime, nSteps + 1)
    y = np.zeros([nSteps + 1, n])
    y[0,:] = y0
    # First step using Euler
    y[1,:] = y[0,:] + dt*func(y[0, :], times[0])
    # Loop for other steps
    for k in range(1, nSteps):
        y[k+1,:] = y[k,:] + (1.5*func(y[k, :], times[k])-0.5*func(y[k-1, :], times[k-1]))*dt
       
    return y ,times

# Adams-Bashforth 3 (again needing fixed timestep) 
def ode_AB3(func, initialTime, finalTime, nSteps, y0):
    '''
    integrates the system of y' = func(y, t) using Forward Euler and Adams-Bashforth 2nd order
    for the time steps in times and given initial condition y0
    ---------------------------------------------------------------------------------------------------------------
    inputs:
        func: the RHS function in the system of ODE
        initialTime: the starting point for time integration for which initial condition is given
        finalTime: the end time integration intervals
        nSteps: number of steps between "initialTime" and "finalTime"
        y0: initial condition (make sure the dimension of y0 and func are the same)
    output:
        y: the solution of ODE. 
        Each row in the solution array y corresponds to a value returned in column vector t
        times: a numpy array consisting of all times at which the solution is given
    '''
    y0 = np.array(y0)
    n = y0.size 
    dt = (finalTime - initialTime)/nSteps
    times = np.linspace(initialTime, finalTime, nSteps + 1)
    y = np.zeros([nSteps + 1, n])
    y[0,:] = y0
    # First step using Euler
    y[1,:] = y[0,:] + dt*func(y[0, :], times[0])
    # Second step using Adams-Bashforth order 2
    y[2,:] = y[1,:] + (1.5*func(y[1,:], times[1]) - 0.5*func(y[0,:], times[0]))
    # Other steps using Adams-Bashforth for order 3
    
    #for k in range(1, nSteps-1):
        #y[k+2,:] = y[k+1,:] + ((23/12)*func(y[k+1,:], times[k+1]) - (16/12)*func(y[k,:], times[k]) + (5/12)*func(y[k,:], times[k]))*dt
    
    return y, times

# Define a function timesteps which generates an array of numbers,
# starting at the value "start" and increasing with step size "h" until the value "stop" is reached.
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

# Call the function
timesteps(0, 0.5, 0.05)

# Define timestep functions for each approximation method described
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
    values, times = ode_Euler(func, times, ics)
    return values, times

def AB2_step(func, start, stop, h, ics):
    '''
    Solves the ODE numerically for the Adams-Bashforth Order 2 method, returning the values of the numerical
    solution and the list of time steps
    ------------------------------------------------------------------------------------------------------------
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
    nSteps = math.ceil((stop - start)/h)
    values, times = ode_AB2(func, start, stop, nSteps, ics)
    return values, times

def AB3_step(func, start, stop, h, ics):
    '''
    Solves the ODE numerically for the Adams-Bashforth Order 3 method, returning the values of the numerical
    solution and the list of time steps
    --------------------------------------------------------------------------------------------------------------
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
    nSteps = math.ceil((stop - start)/h)
    values, times = ode_AB3(func, start, stop, nSteps, ics)
    return values, times


# In[2]:


#  defining the function in the RHS of the ODE given in the question
def eq1_dy_dt(y, t):
    return 3*t - 4*y


# In[3]:


# printing the solution in a table
from pandas import DataFrame

# Define a function produce_df to create a dataframe
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

# Use the function to produce a dataframe detailing the values for each method, using the alternate approach
df_eq1 = produce_df(Euler_step, eq1_dy_dt, 0, 0.5, 0.05, 1)
# Create the columns corresponding to each method
df_eq1.columns = ["Euler, h=0.05"]
df_eq1["AB2, h=0.05"] = produce_df(AB2_step, eq1_dy_dt, 0, 0.5, 0.05, 1)
df_eq1["AB3, h=0.05"] = produce_df(AB3_step, eq1_dy_dt, 0, 0.5, 0.05, 1)

# Display the table
display(df_eq1)

# Display the plot
df_eq1.plot()


# ## Task 2 (3 marks)
# 
# Use `SymPy` to solve the differential equation $y' = 3t-4y$, with $y(0)=1$, present the analytical solution, and check the exact value of $y(0.5)$.
# 
# Compare the result with the approximations from the three methods in Task 1. You may use a table to show the results of each method at $y(0.5)$. Which method is the most/least accurate? Why?

# In[4]:


# standard setup
import sympy as sym
sym.init_printing()
from IPython.display import display_latex
import sympy.plotting as sym_plot
from sympy import *


#eq1_sol = sym.dsolve(eq1, y(t), ics={y(0):1})

# Define the equation with symbols and functions for y and t
t = sym.symbols('t')
y = sym.Function('y')

# Use sym.eq() to solve the ODE using week 1 methods
eq1 = sym.Eq(y(t).diff(t), 3*t - 4*y(t))
print("The equation")
display_latex(eq1)

# Solve the equation
eq1sol = sym.dsolve(eq1, y(t))
print("has an analytical solution of")
display_latex(eq1sol)
print("or equivalently,")
display_latex(sym.simplify(eq1sol))

# Solve the ODE for the initial condition y(0) = 1 using dsolve
eq1_sol = sym.dsolve(eq1, y(t), ics={y(0):1})
print('The particular solution for y(0) = 1 is:')
display_latex(eq1_sol)

# Find the exact value of y(0.5) using a new function, exact_yval
def exact_yval(t):
    '''
    Takes in the parameter of the ODE t and returns the particular solution to our ODE given our initial condition.
    ---------------------------------------------------------------------------------------------------------------
    input:
         t- parameter of the ODE
    output:
         particular solution y(t) for initial condition y(0) = 1
    '''
    return (3*t)/4 - 3/16 + (19*np.exp(-4*t))/16

# Use the function exact_yval to calculate the exact value of y(0.5)
y_exact = exact_yval(0.5)
# Print the exact value
print(f'The exact value of y(0.5) is equal to',y_exact)

# Create a new table which now includes the exact values of y over the interval of t
# Using same steps as before

df_eq1 = produce_df(Euler_step, eq1_dy_dt, 0, 0.5, 0.05, 1)
# Create the columns corresponding to each method including the exact values
df_eq1.columns = ["Euler, h=0.05"]
df_eq1["AB2, h=0.05"] = produce_df(AB2_step, eq1_dy_dt, 0, 0.5, 0.05, 1)
df_eq1["AB3, h=0.05"] = produce_df(AB3_step, eq1_dy_dt, 0, 0.5, 0.05, 1)
df_eq1["Exact"] = DataFrame(data = [exact_yval(t) for t in timesteps(0,0.5,0.05)],
                            index = np.round(timesteps(0,0.5,0.05),3))

# Display the updated table
display(df_eq1)

# Display the updated plot
df_eq1.plot()


# The method which is least accurate is the Forward Euler method. 
# 
# For the functions that I have actually been able to plot correctly, the Forward Euler method, the Adams-Bashforth Order 2 method and the exact values for y(t), the Adams-Bashforth Order 2 method is most accurate. This is because it is a recursive function which takes another approximation method as the initial step (In this case the Forward Euler method), so by approximating to a higher order can allow us to determine a more precise solution than approximation methods of lower order as shown with the Forward Euler method. We can also look at the table to see that the margin of error is lower for AB2 than Forward Euler from t = 0.1 onwards as t approaches 0.5, which reaffirms the idea that higher order approximations are more accurate in general.
# 
# The Adams-Bashford Order 3 method should be of the highest accuracy as per my reasoning for higher order approximation methods, however I made an error somewhere in the definition which I was unable to fix and as a result could not plot the values properly.
