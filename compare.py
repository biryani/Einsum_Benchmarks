from numpy import einsum, dot
from timeit import default_timer as timer
import timeit
import numpy as np
"""
  Efficiency of einsum in contracting n dimensional tesors vs matrix vector multiplication using I \otimes U

"""

def compare_einsum_dot(n,k, repeat = 10,iterations = 100 ):
  phi = np.random.random_sample([2]*n)
  U =  np.random.random_sample([2]*(2*k))
  M = np.kron(np.eye(2**(n-k)),U.reshape(2**k,2**k))
  x = phi.flatten()
  t1 = timeit.Timer(lambda: dot(M, x)).repeat(repeat, iterations)
  ind = range(n)
  indU = ind[-k:]
  indU = indU + range(n,n+k)
  t2 =  timeit.Timer(lambda:np.einsum(phi, ind, U, indU)).repeat(repeat,iterations)
  return np.mean(t1)/float(iterations), np.mean(t2)/float(iterations)



Time = compare_einsum_dot(10,1,repeat = 3)
print Time        



