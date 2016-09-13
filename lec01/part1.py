#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
import pylab


import csv
reader = csv.reader(open("ex1data1.txt"), delimiter=",")

x = list()
y = list()
for xi, yi in reader:
  x.append(float(xi))
  y.append(float(yi))

#print("x = ", x[:10])
#print("y = ", y[:10])


def regdots(x, y):
  fig = pyplot.figure(figsize=(16*.6,9*.6))
  ax = fig.add_subplot(111)
  fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
  ax.scatter(x, y, c='r', s=80, label="Data")
    
  ax.set_xlabel("Population")
  ax.set_ylabel("Profit")
  ax.margins(.05,.05)
  pyplot.ylim(min(y)-1, max(y)+1)
  pyplot.xlim(min(x)-1, max(x)+1)
  #ax = fig.axes[0]
  #handles, labels = ax.get_legend_handles_labels()
  #fig.legend(handles, labels, fontsize='15', loc='lower right')
  return fig

def legend(fig):
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize='15', loc='lower right')

def regline(fig, fun, theta, x):
  ax = fig.axes[0]
  x0, x1 = min(x), max(x)
  X = [x0, x1]
  Y = [fun(theta, x) for x in X]
  ax.plot(X, Y, linewidth='2',
            label=(r'$y=%.2f+%.2f x$' % (theta[0], theta[1])))


def h(theta, x):
  return theta[0] + theta[1]*x

def J(h, theta, x, y):
  m = len(y)
  return (1.0/(2*m) * sum((h(theta, x[i]) - y[i])**2
                            for i in range(m)))

def errorSurface(x, y):
  pX = np.arange(-10,10.1,0.1)
  pY = np.arange(-1,4.1,0.1)
  pX, pY = np.meshgrid(pX, pY)
    
  pZ = np.matrix([[J(h,[t0, t1], x, y)
                   for t0, t1 in zip(xRow, yRow)]
                    for xRow, yRow in zip(pX, pY)])
      
  fig = pyplot.figure(figsize=(16*.6,9*.6))
  ax = fig.add_subplot(111, projection='3d')
  pyplot.subplots_adjust(left=0.0, right=1, bottom=0.0, top=1)
  ax.plot_surface(pX ,pY, pZ, rstride=2, cstride=8, linewidth=0.5,
                                    alpha=0.5, cmap='jet', zorder=0,
                                    label=r"$J(\theta)$")
  ax.view_init(elev=30., azim=-160)
                    
  ax.set_xlim3d(-10, 10);
  ax.set_ylim3d(-1, 4);
  ax.set_zlim3d(-100, 800);
                    
  N = range(0,800,20)
  pyplot.contour(pX,pY,pZ, N,zdir='z',offset=-100, cmap='coolwarm', alpha=1)
                    
  ax.plot([-3.89578088] * 3,
          [ 1.19303364] * 3,
          [-100, 4.47697137598, 700],
          color='red', alpha=1, linewidth=1.3, zorder=100, linestyle='dashed', label="Minimum: 4.4770")
                    
  ax.set_zlabel(r"$J(\theta)$", fontsize="15")
  ax.set_xlabel(r"$\theta_0$", fontsize="15")
  ax.set_ylabel(r"$\theta_1$", fontsize="15")
  ax.margins(0,0,0)
  fig.tight_layout()
  return fig, ax;


#def GD2(h, fJ, theta, x, y, alpha=0.1, eps=10**-3, steps=None):
#    errorCurr = fJ(h, theta, x, y)
#    errors = [[errorCurr, theta]] # Logujemy poziom błędu
#
#    m = len(y)
#    while True:
#        thetaPrime = [0, 0]
#        thetaPrime[0] = theta[0] - alpha/float(m) * sum(h(theta, x[i]) - y[i] 
#                                                 for i in range(m)) 
#        thetaPrime[1] = theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i] 
#                                                 for i in range(m))
#        theta = thetaPrime
#
#        print("theta[1] part2: {}".format(sum((h(theta, x[i]) - y[i]) * x[i] for i in range(m))))
#
#        if steps != None and len(errors) >= steps:
#            break
#        
#        errorCurr, errorPrev = fJ(h, theta, x, y), errorCurr
#        if abs(errorPrev - errorCurr) <= eps:
#            print("exiting early {} {}".format(errorPrev, errorCurr))
#            break
#        print("recursing {} {} {} {}".format(errorPrev - errorCurr, errorPrev, errorCurr, thetaPrime))
#        errors.append([errorCurr, theta[:]]) 
#        
#    return theta, errors

#thetaBest, errors = GD2(h, J, [0,0], x, y, alpha=0.01, eps=0.0)

def GD(h, fJ, theta, x, y, alpha=0.1, eps=10**-3):
    errorCurr = fJ(h, theta, x, y)
    errors = [[errorCurr, theta]]

    m = len(y)
    while True:
        thetaPrime = [0, 0]
        thetaPrime[0] = theta[0] - alpha/float(m) * sum( h(theta, x[i]) - y[i]         for i in range(m))
        thetaPrime[1] = theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i] for i in range(m))
        
        #print("theta[1] part2: {}".format(sum((h(theta, x[i]) - y[i]) * x[i] for i in range(m))))
        
        theta = thetaPrime
        errorCurr, errorPrev = fJ(h, theta, x, y), errorCurr
        if abs(errorPrev - errorCurr) <= eps:
          #print("exiting early {} {}".format(errorPrev, errorCurr))
            break
        #print("recursing {} {} {} {}".format(errorPrev - errorCurr, errorPrev, errorCurr, thetaPrime))
        errors.append([errorCurr, theta]) # Logujemy poziom błędu
    return theta, errors

#thetaBest, errors = GD(h, J, [0,0], x, y, alpha=0.01, eps=0)

#def LatexMatrix(matrix):
#  ltx = r'\left[\begin{array}'
#  m, n = matrix.shape
#  ltx += '{' + ("r" * n) + '}'
#  for i in range(m):
#    ltx += r" & ".join([('%.4f' % j.item()) for j in matrix[i]]) + r" \\ "
#  ltx += r'\end{array}\right]'
#  return ltx

#fig = regdots(x,y)
#regline(fig, fun, [theta0, theta1], x)
#
#fig, ax = errorSurface(x,y)

#fig = regdots(x, y)
thetaBest, errors = GD(h, J, [0,0], x, y, alpha=0.01, eps=0)
#regline(fig, h, thetaBest, x)
#legend(fig)

#pyplot.show()
print("{}".format(errors[-1]))
#print(errors)
