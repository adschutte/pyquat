import numpy as np
from cadtools import skew

def eigenrot(ang, e):
    """Description:  Calculate a unit quaternion given a rotation angle and eigenaxis.
        Input:  ang - rotation angle [deg]
                e - eigenaxis (a unit 3-vector)
        Output:  out - a unit quaternion"""
    ang    = np.float64(ang) * np.float64(np.pi/180.0)
    scalar = np.cos(ang/2.0)
        
    evec   = np.float64(np.array([e[0],e[1],e[2]]))
    evec   = np.sin(ang/2.0)*evec        
    
    return Quaternion(scalar,evec)

def dcm_to_quat(A):
    """Description:  Calculate a unit quaternion given an inertial to body direction cosine matrix.
        Input:  Inertial to Body direction cosine matrix
        Output:  Unit quaternion in canonical form"""
    try:
        if A.shape != (3,3):
            raise ValueError, "pyquat::dcm_to_quat: error - array dimension must be (3,3)"
    except:
        raise AttributeError, "pyquat::dcm_to_quat: error - input must be a numpy array"
              
    u = np.zeros((4,),dtype=np.float64)
    
    u[0] =  A[0,0] - A[1,1] - A[2,2]
    u[1] = -A[0,0] + A[1,1] - A[2,2]
    u[2] = -A[0,0] - A[1,1] + A[2,2]
    u[3] =  A[0,0] + A[1,1] + A[2,2]
    
    ts = -1
    j = 0
    for i in range(4):
        if u[i] > ts:
            ts = u[i]
            j = i
            
    ts     = 2.0*np.sqrt(1.0+u[j])
    ts_inv = 1.0/ts
    u[j]   = ts/4.0

    if j==0:
        u[1] = (A[0,1]+A[1,0])*ts_inv
        u[2] = (A[2,0]+A[0,2])*ts_inv
        u[3] = (A[1,2]-A[2,1])*ts_inv
    elif j==1:
        u[0] = (A[0,1]+A[1,0])*ts_inv
        u[3] = (A[2,0]-A[0,2])*ts_inv
        u[2] = (A[1,2]+A[2,1])*ts_inv
    elif j==2:
        u[3] = (A[0,1]-A[1,0])*ts_inv
        u[0] = (A[2,0]+A[0,2])*ts_inv
        u[1] = (A[1,2]+A[2,1])*ts_inv
    elif j==3:
        u[2] = (A[0,1]-A[1,0])*ts_inv
        u[1] = (A[2,0]-A[0,2])*ts_inv
        u[0] = (A[1,2]-A[2,1])*ts_inv

    u = u*(3-np.sum(u*u))/2.0
        
    return Quaternion(scalar=u[3],vec=u[0:3])

_types = [int, float, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]

class Quaternion(object):    
    def __init__(self, obj=object, scalar=1.0, vec=np.array([0.0,0.0,0.0]), rand=False):
        if rand:
            x = np.random.randn(4)
            x = x/np.linalg.norm(x)
            self.scalar = x[0]
            self.vec    = x[1:4]

        elif isinstance(obj,self.__class__):
            self.scalar = obj.scalar
            self.vec    = obj.vec
        
        elif isinstance(obj,np.ndarray):
            if obj.shape == (3,3):
                x = dcm_to_quat(obj)
                self.scalar = x.scalar
                self.vec    = x.vec
            else:
                print "pyquat::Quaternion::__init__: warning - expected numpy.ndarray.shape = (3,3); setting to identity quaternion"
                self.scalar = scalar
                self.vec    = vec
        else:
            self.scalar = scalar
            self.vec    = vec
            
    @classmethod
    def __instancecheck__(cls, instance):
        if type(instance) == cls:
            return True
        return False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.vec[0] >= 0.0:
            sign1 = ' + '
        else:
            sign1 = ' - '

        if self.vec[1] >= 0.0:
            sign2 = ' + '
        else:
            sign2 = ' - '
            
        if self.vec[2] >= 0.0:
            sign3 = ' + '
        else:
            sign3 = ' - '

        ns = str(np.float32(self.scalar))
        v0 = str(np.float32(np.abs(self.vec[0])))
        v1 = str(np.float32(np.abs(self.vec[1])))
        v2 = str(np.float32(np.abs(self.vec[2])))

        return ns + sign1 + v0 + 'i' + sign2 + v1 + 'j' + sign3 + v2 + 'k'

    def __add__(self, q):
        if isinstance(q, self.__class__):
            scalar = self.scalar + q.scalar
            vec    = self.vec + q.vec
            return Quaternion(scalar=scalar,vec=vec)
        else:
            raise TypeError, "unsupported operand type(s) for +: {} and {}".format(self.__class__, type(q))

    def __sub__(self, q):
        if isinstance(q, self.__class__):
            scalar = self.scalar - q.scalar
            vec    = self.vec - q.vec
            return Quaternion(scalar=scalar,vec=vec)
        else:
            raise TypeError, "unsupported operand type(s) for -: {} and {}".format(self.__class__, type(q))
            
    def __neg__(self):
        return Quaternion(scalar=-self.scalar,vec=-self.vec)

    def __pos__(self):
        return Quaternion(scalar=+self.scalar,vec=+self.vec)
        
    def __mul__(self, q):
        if isinstance(q, self.__class__):
            scalar = self.scalar*q.scalar - np.dot(self.vec, q.vec)
            vec    = self.scalar*q.vec + q.scalar*self.vec + np.cross(self.vec,q.vec)            
            return Quaternion(scalar=scalar,vec=vec)
        elif type(q) in _types:
            scalar = q*self.scalar
            vec    = q*self.vec
            return Quaternion(scalar=scalar,vec=vec)                
        else:
            raise TypeError, "unsupported operand type(s) for *: '{}' and '{}'".format(self.__class__, type(q))

    def __rmul__(self, q):
        return self.__mul__(q)
        
    def __div__(self, q):
        if isinstance(q, self.__class__):
            return self * q.inv()
        elif type(q) in _types:
            scalar = self.scalar/q
            vec    = self.vec/q
            return Quaternion(scalar=scalar,vec=vec)
        else:
            raise TypeError, "unsupported operand type(s) for /: '{}' and '{}'".format(self.__class__, type(q))
            
    def __rdiv__(self, q):
        if type(q) in _types:
            return q * self.inv()
        else:
            raise TypeError, "unsupported operand type(s) for /: '{}' and '{}'".format(self.__class__, type(q))
            
    def __pow__(self, q):
        if isinstance(q, self.__class__) or (type(q) in _types):
            return (self.log() * q).exp()
        else:
            raise TypeError, "unsupported operand type(s) for **: '{}' and '{}'".format(self.__class__, type(q))

    def conj(self):
        return self.conjugate()

    def conjugate(self):
        return Quaternion(scalar=self.scalar, vec=-self.vec)

    def norm(self):
        return np.sqrt(self.scalar**2 + self.vec.dot(self.vec))        

    def inner(self,q):
        if isinstance(q, self.__class__):
            # return self*q.conj()
            return 0.5*(self.conj()*q + q.conj()*self)
        else:
            raise TypeError, "unsupported operand type(s) for inner(): '{}' and '{}'".format(type(Quaternion), type(q))
            
    def outer(self,q):
        if isinstance(q, self.__class__):
            return 0.5*(self.conj()*q - q.conj()*self)
        else:
            raise TypeError, "unsupported operand type(s) for inner(): '{}' and '{}'".format(type(Quaternion), type(q))
                                                            
    def inv(self):
        return self.conj()/self.norm()**2

    def exp(self):
        qvn = np.linalg.norm(self.vec)
        q0  = np.exp(self.scalar)
        scalar = q0*np.cos(qvn)
        vec    = q0*np.sin(qvn)/qvn * self.vec
        return Quaternion(scalar=scalar,vec=vec)
        
    def log(self):
        scalar = np.log(self.norm())
        vec    = self.vec/np.linalg.norm(self.vec)*np.arccos(self.scalar/self.norm())
        return Quaternion(scalar=scalar,vec=vec) 

    def sqrt(self):
        return self**0.5 
        
    def matp(self):
        A = self.scalar
        B = -self.vec
        C = self.vec
        D = self.scalar*np.eye(3)+skew(self.vec)
        mat = np.empty((4,4))
        mat[0,0]     = A
        mat[0,1:4]   = B
        mat[1:4,0]   = C
        mat[1:4,1:4] = D
        return mat
        
    def matm(self):
        A = self.scalar
        B = -self.vec
        C = self.vec
        D = self.scalar*np.eye(3)-skew(self.vec)
        mat = np.empty((4,4))
        mat[0,0]     = A
        mat[0,1:4]   = B
        mat[1:4,0]   = C
        mat[1:4,1:4] = D
        return mat
                                        
    def dcm(self):
        """Description:  Compute body to inertial direction cosine matrix."""
        out = np.zeros((3,3))
        out = (2.0*self.scalar**2 - 1.0)*np.eye(3) + \
            2.0 * np.outer(self.vec,self.vec) + \
            2.0 * self.scalar * skew(self.vec)
        return out        

    def wmap(self):
        out = np.zeros((3,4))
        out = np.array([[-self.vec[0],  self.scalar,  self.vec[2], -self.vec[1]],
                        [-self.vec[1], -self.vec[2],  self.scalar,  self.vec[0]],
                        [-self.vec[2],  self.vec[1], -self.vec[0],  self.scalar]])
        out = 2.0*out
        return out
        
    def len(self):
        return 4
        
    def real(self):
        return self.scalar
        
    def imag(self):
        return self.vec
