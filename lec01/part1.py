#!/usr/bin/env python3


import pandas as pd
import numpy as np


import csv
reader = csv.reader(open("ex1data1.txt"), delimiter=",")

x = list()
y = list()
for xi, yi in reader:
  x.append(float(xi))
  y.append(float(yi))

#print("x = ", x[:10])
#print("y = ", y[:10])


def h(theta, x):
  return theta[0] + theta[1]*x

def J(h, theta, x, y):
  m = len(y)
  return (1.0/(2*m) * sum((h(theta, x[i]) - y[i])**2
                            for i in range(m)))



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
