import numpy as np
import pylab
A = np.loadtxt('time.dat', skiprows = 1)
#x = A[15:,1]/ A[15:,0]
#y = A[15:,3]/ A[15:,2]

#pylab.plot(x,y,'*')
pylab.xlabel(r"$\frac{k}{n}$")
pylab.ylabel(r"$\frac{t_{ein}}{t_{dot}} $")
start = 21
for i in range(7,13):
  x = A[start:start + i,1]/ A[start:start +i,0]
  y = A[start:start +i,3]/ A[start:start +i,2]
  pylab.plot(x,y,'-*')
  start = start + i 

pylab.show()



