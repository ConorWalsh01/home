#!/usr/bin/env python
# coding: utf-8

# # Lab Assignment 4
# 
# ## Conor Walsh, s1949139
# 
# 

# ## Task 1 (5 marks)
# 
# Give your implementation of the `plot_approx` and `approx_fourier` functions from Lab 4.
# 
# Use them to produce a plot of a Fourier series approximation of the function defined by
# 
# $$
# f(x)=\left\{\begin{array}{ll}
# -\frac{1}{2} x & -2 \leq x<0 \\
# 2 x-\frac{1}{2} x^2 & 0 \leq x<2
# \end{array} \quad f(x+4)=f(x)\right.
# $$
# 
# using the first 10 terms of the Fourier series.
# 
# Also include a piecewise plot of $f(x)$ for a single interval of periodicity.

# In[44]:


# Install the packages:

import sympy as sym
import sympy.plotting as sym_plot
sym.init_printing()
from IPython.display import display_latex

# Define the symbols x and n:

x = sym.symbols('x')
n = sym.symbols('n')

# Define the function f(x) as a sympy expression:

f = sym.Piecewise((2*x-x**2/2, x>=0), (-x/2, x<0))

# Define the function approx_fourier:

def approx_fourier(f, L, num_terms):
    
    '''
    Computes the Fourier series with terms for the function f with period 2L that matches f on the interval [-L, L]
        -----------------------------------------------------------------------------------------------------------
    inputs:
        f: a sympy expression of the function f(x)
        L: The limit of the function f(x) with period 2L that matches f on [-L, L]
        num_terms: the number of terms of the Fourier series
    output:
        f_approx.doit(): the approximated Fourier series of f(x) for our chosen number of terms
    '''

    a0 = sym.Rational(1,L)*sym.integrate(f, (x, -L, L))
    an = sym.Rational(1,L)*sym.integrate(f*sym.cos(n*sym.pi*x/L), (x, -L, L))
    bn = sym.Rational(1,L)*sym.integrate(f*sym.sin(n*sym.pi*x/L), (x, -L, L))
    f_approx = a0/2 + sym.Sum(an*sym.cos(n*sym.pi*x/L)+bn*sym.sin(n*sym.pi*x/L), (n,1,num_terms))
    return f_approx.doit()

# Define the function plot_approx:

def plot_approx(f, L, num_terms):
    
    '''
    Produces a plot of f and its Fourier series using the function approx_fourier:
    -----------------------------------------------------------------------------------------------------------
    inputs:
        same as above for the function approx_fourier():
    output:
        f_plot: a plot of the function f and its corresponding approximated Fourier series:
    '''
    
    f_approx = approx_fourier(f, L, num_terms)
    f_plot = sym_plot.plot((f_approx,(x,-2*L,2*L)), (f,(x,-L,L)), show  = False)
    f_plot[0].line_color = "blue"
    f_plot[1].line_color = "red"
    return f_plot

# Show the plot for the first 10 terms of the Fourier series of f(x):

plot_approx(f, 2, 10).show()

# Show the first 10 terms of the Fourier series of f(x):

display_latex(approx_fourier(f, 2, 10))


# In[45]:


# Piecewise plot of f(x) for a single interval of periodicity:
sym_plot.plot(f, (x,-2,2), title = 'Piecewise plot of f(x) over single interval of periodicity [-2,2]')


# ## Task 2 (5 marks)
# 
# Solve Exercise $3.1$ from Lab 4 , but with the initial condition
# 
# $$
# u(x, 0)=f(x)= \begin{cases}1 & L / 2-1<x<L / 2+1 \\ 0 & \text { otherwise. }\end{cases}
# $$
# 
# Note that to ensure the code runs in reasonable time, you should use $L=10$ and run the animation for $0 \leq t \leq 20$, with only 2 frames per second. You should use at least 200 terms of the series solution in order to obtain a good approximation.
# Describe the behaviour of the solution.

# In[46]:


# Import the packages:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set up L = 10, a = 1 and the initial conditions for the piecewise function f(x):

L = 10
a = 1
f = sym.Piecewise((0, (x<=L/2 - 1)), (1, (x<=L/2 + 1)), (0, True))

# Compute the coefficients c_n:

cn = sym.Rational(2,L)*sym.integrate(f*sym.sin(n*sym.pi*x/L), (x, 0, L))

# Define the symbol t:

t = sym.symbols('t')

# Approximate the overall solution for u(x,t) by taking the first 200 terms of the sum:

u_symbolic = sym.Sum(cn.simplify()*sym.sin(n*sym.pi*x/L)*sym.cos(n*sym.pi*a*t/L), (n,1,200))

# Set the number of frames per second to 2:

fps = 2 

# Define a subplot:

fig, ax = plt.subplots()

x_vals = np.linspace(0,L,2000)

# Transform the sympy expression to a lambda function to calculate numerical values:

u = sym.lambdify([x, t], u_symbolic, modules='numpy')

# Set up the initial frame:
line, = ax.plot(x_vals, u(x_vals,0), 'k-')
plt.plot(x_vals,u(x_vals,0),'r:')

# Add labels to the x and y axes and give the animation a title:

plt.xlabel('x')
plt.ylabel('u')
plt.ylim(-1.5,1.5)
plt.title('Animation of our truncated Fourier series as it converges to f(x)', fontsize = 10.5)
plt.close()

# Add an annotation showing the time (this will be updated in each frame):
txt = ax.text(0, 0.9, 't=0')

# Create a function to represent the initial function itself:

def init():
    line.set_ydata(u(x_vals,0))
    return line,

# Create a function for animating the plot:

def animate(i):
    
    # Update the data:
    
    line.set_ydata(u(x_vals,i/fps))
    
    # Update the annotation:
    
    txt.set_text('t='+str(i/fps))
    return line, txt

# Run the animation of the plot for 0≤t≤20: 

ani = animation.FuncAnimation(fig, animate, np.arange(0, fps*20.5), init_func=init,
                              interval=500, blit=True, repeat=False)

# Show the animation of the plot of the Fourier series converging to f(x):

HTML(ani.to_jshtml())


# Our periodic function $f(x)$ is discontinuous. When we approximated by Fourier series, we see that the truncated Fourier series oscillates rapidly near jumps in the function at $x$ = $4$ and $x$ = $6$. This behaviour of the solution closely resembles the Gibbs Phenomenon, which can be demonstrated explicitly by computing the amplitude of the largest of these oscillations for our given function $f(x)$.

# ## Submission instructions
# 
# After producing the animation (as in the lab), you should also use the following line of code to produce an mp4 file of your animation. The file should then appear alongside the .ipynb file in your Jupyter file list. Note, if you are not using noteable, you will need to install ffmpeg. You can see this page for instructions https://ffmpeg.org/.

# In[47]:


ani.save('hdeq_lab4_task2.mp4', writer='ffmpeg', fps=20)


# Alternatively, you can use the code 

# In[48]:


ani.save('hdeq_lab4_task2.gif', writer='Pillow ', fps=20)


# To save the simulation as a gif. Upload four files on gradsecope the .pdf, the .ipynb, the .py and the .mp4 (or .gif).

# In[ ]:




