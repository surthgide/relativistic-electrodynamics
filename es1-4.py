# =============================================================================
# EXERCISE 1-4
# =============================================================================

# the following exercise uses the convention:
# c = 1

# =============================================================================
# packages import
# =============================================================================

import numpy as np

# =============================================================================
# classes definition
# =============================================================================

class lorVec:
    
    '''Lorentz vector with entries, module, and squared module'''

    def __init__(self, *vec, **check):
        self.vec = rnd4Vec(-10, 10) if check.get('rnd', True) else list(*vec)
        self.mod = dotLor(self.vec, self.vec)
        
        if self.mod == 0:
            self.vecType = 'light'
        elif self.mod > 0:
            self.vecType = 'time'
        else:
            self.vecType = 'space'
    
# ---

class boost:
    
    '''Lorentz boost, with matrix and versor'''
    
    def __init__(self, *vel, **check):
        self.vel = rnd3Vel() if check.get('rnd', True) else list(*vel)
        self.boostMatrix = lorTransf(self.vel)

# =============================================================================
# constants definition
# =============================================================================

# true random generator

rand = np.random.default_rng()

# metric tensor

eta = np.diag([1, -1, -1, -1])

# =============================================================================
# functions definition
# =============================================================================

def dot(m1,m2):
    # dot product
    return np.dot(m1,m2)

# ---

def mod(vec):
    # module of a 3-vector
    return np.sqrt(sum(x ** 2 for x in vec))

# ---

def dotLor(vec1, vec2):
    # squared module of a 4-vector
    return dot(dot(vec1, eta), vec2)

# ---

def g(v):
    # gamma factor for the velocity v
    return 1 / np.sqrt(1 - mod(v) ** 2)

# ---

def rnd3Vel():
    # generate a random 3-velocity (c = 1)
    vel = 20*rand.random(3) - 10
    vel /= mod(vel) / rand.random()
    return vel

# ---

def rnd4Vec(minVal, maxVal):
    # generate a random 4-vector with entries in (minVal, maxVal)
    vec = (maxVal - minVal)*rand.random(4) + minVal
    # vec /= mod(vec) / rand.random()
    return vec

# ---

def lorTransf(v):
    # generate a Lorentz boost matrix with velocity given by v
    B = [ [ g(v), - g(v) * v[0], - g(v) * v[1], - g(v) * v[2] ],
          [ - g(v) * v[0], 1 + (g(v) - 1) * (v[0]  / mod(v)) ** 2, (g(v) - 1) * (v[0] * v[1]  / mod(v) ** 2), (g(v) - 1) * (v[0] * v[2]  / mod(v) ** 2) ],
          [ - g(v) * v[1], (g(v) - 1) * (v[1] * v[0]  / mod(v) ** 2), 1 + (g(v) - 1) * (v[1]  / mod(v)) ** 2, (g(v) - 1) * (v[1] * v[2]  / mod(v) ** 2) ],
          [ - g(v) * v[2], (g(v) - 1) * (v[2] * v[0]  / mod(v) ** 2), (g(v) - 1) * (v[2] * v[1]  / mod(v) ** 2), 1 + (g(v) - 1) * (v[2]  / mod(v)) ** 2 ] ]
    return B

# ---

def printMatrix(matrix):
    for lst in matrix:
        for element in lst:
            print("{:10.6f}".format(element), end="\t")
        print("")

# =============================================================================
# actual code
# =============================================================================

# # clear screen for new output

# os.system('cls' if os.name == 'nt' else 'clear')
# time.sleep(0.1)

print('=============================================================================\nOUTPUT\n=============================================================================')

# generate two random vectors and one fixed

print('VECTORS DECLARATIONS\n')

print('random vector u:\n')

u = lorVec()
print('u = ', u.vec, '\nmodule = ', u.mod, '\ntype = ', u.vecType)

print('\nrandom vector v:\n')

v = lorVec()
print('v = ', v.vec, '\nmodule = ', v.mod, '\ntype = ', v.vecType)

print('\nfixed vector w:\n')

vec = [1,1,1,0]

w = lorVec(vec, rnd=False)
print('w = ', w.vec, '\nmodule = ', w.mod, '\ntype = ', w.vecType)

# inner products between 4-vectors

print('\nINNER PRODUCTS BETWEEN VECTORS\n')

uv = dotLor(u.vec, v.vec)
uw = dotLor(u.vec, w.vec)
vw = dotLor(v.vec, w.vec)

print('u v = ', uv)
print('u w = ', uw)
print('v w = ', vw)

# boost of v in random direction

print('\nRANDOM AND FIXED BOOST DECLARATIONS\n')

print('random boost B:\n')

B = boost()
print('boost velocity = ', B.vel, '\nboost velocity module = ', mod(B.vel))
print('boost matrix =')
printMatrix(B.boostMatrix)

print('\nfixed boost C:\n')

vel = [0, 0, 0.9]

C = boost(vel, rnd=False)
print('boost velocity = ', vel, '\nboost velocity module = ', mod(vel))
print('boost matrix =')
printMatrix(C.boostMatrix)

# boosting vectors

print('\nBOOSTING VECTORS\n')

print('boost via B:\n')

Bu = dot(B.boostMatrix, u.vec)
Bv = dot(B.boostMatrix, v.vec)
Bw = dot(B.boostMatrix, w.vec)

print('B u = ', Bu)
print('B v = ', Bv)
print('B w = ', Bw)

print('\nboost via C:\n')

Cu = dot(C.boostMatrix, u.vec)
Cv = dot(C.boostMatrix, v.vec)
Cw = dot(C.boostMatrix, w.vec)

print('C u = ', Cu)
print('C v = ', Cv)
print('C w = ', Cw)


# inner products between boosted 4-vectors

print('\nINNER PRODUCTS BETWEEN BOOSTED VECTORS\n')

BuBv = dotLor(Bu, Bv)
BuBw = dotLor(Bu, Bw)
BvBw = dotLor(Bv, Bw)

print('B u B v = ', BuBv)
print('B u B w = ', BuBw)
print('B v B w = ', BvBw)

print()

CuCv = dotLor(Cu, Cv)
CuCw = dotLor(Cu, Cw)
CvCw = dotLor(Cv, Cw)

print('C u C v = ', CuCv)
print('C u C w = ', CuCw)
print('C v C w = ', CvCw)

# comparison between inner products before and after boost

print('\nCOMPARISON BETWEEN INNER PRODUCTS BEFORE AND AFTER BOOST\n')

print('B u B v - u v = {:2.3f}'.format(BuBv - uv))
print('B u B w - u w = {:2.3f}'.format(BuBw - uw))
print('B v B w - v w = {:2.3f}'.format(BvBw - vw))

print()

print('C u C v - u v = {:2.3f}'.format(CuCv - uv))
print('C u C w - u w = {:2.3f}'.format(CuCw - uw))
print('C v C w - v w = {:2.3f}'.format(CvCw - vw))

print('=============================================================================\nEND OF OUTPUT\n=============================================================================')
