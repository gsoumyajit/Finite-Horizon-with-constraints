import numpy as np
from numpy.random import choice
from numpy.random import randint
from scipy.special import softmax
from collections import deque
from random import sample

T=10
s=10

save1=randint(s,size=(T,s))
save2=randint(s,size=(T,s))

np.savetxt("mdp/S1",save1)
np.savetxt("mdp/S2",save2)
