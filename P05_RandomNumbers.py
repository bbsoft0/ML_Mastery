# demonstrates the python pseudorandom number generator
from numpy.random import rand
from numpy.random import seed
from numpy.random import randint
import random

# demonstrates the numpy pseudorandom number generator
# seed the generator
seed(7)
print(rand(5))
# seed the generator to get the same sequence
print('Reseeded')
seed(7)
print(rand(5))

seed(770)
print('####################################################')
print('5 numbers between 1 and 45:',  end=" ")
for i in range(5):
    print(randint(1, 45+1), end=" ")
print(' ')
print('1 number between 1 and 20:',  end=" ")
print(randint(1, 20+1))
print('####################################################')
