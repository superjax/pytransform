import numpy as np
from quaternion import *
from common import *

I_3x3 = np.eye(3)

class Transform:
    def __init__(self, *argv):
        if isinstance(argv[0], Quaternion):
            self.q = argv[0]
            self.t = argv[1]
        else:
            self.q = Quaternion(argv[0][:4])
            self.t = argv[0][4:]

    def __str__(self):
        return self.q.__str__() + ", [ " + str(self.t[0,0]) + ", " + \
                str(self.t[1,0]) + ", " + str(self.t[2,0]) + " ]"

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        return self.otimes(other)

    def __imul__(self, other):
        pass

    def __add__(self, other):
        pass

    def __iadd__(self, other):
        pass

    def __sub__(self, other):
        pass

    # Return the passive quaternion associated with this transform
    @property
    def rotation(self):
        return self.q

    # Return the translation associated with this transform
    @property
    def translation(self):
        return self.t

    # Return the active rotation matrix associated with this transform
    @property
    def R(self):
        return self.q.R.T

    # Return the 7x1 array of the underlying representation
    # This is a passive quaternion and the translation vector
    @property
    def elements(self):
        return np.vstack((self.q.elements, self.t))

    # Return the 4x4 representation of this transform
    @property
    def T(self):
        return np.vstack((np.hstack((self.q.R.T, self.t)),
                          np.hstack((np.zeros((1,3)), np.ones((1,1))))))

    # Invert the transform
    @property
    def inverse(self):
        qinv = self.q.inverse
        return Transform.from_q_and_t(qinv, -qinv.rotp(self.t))

    # Provide a passive quaternion and a translation
    @classmethod
    def from_q_and_t(cls, q, t):
        return cls(q, t)

    # Provide an active R and a translation
    @classmethod
    def from_R_and_t(cls, R, t):
        return cls(Quaternion.from_R(R.T), t)

    # Provide an active quaternion and translation in the form of a 7x1 array
    @classmethod
    def from_array(cls, arr):
        return cls(Quaternion(arr[:4]), arr[4:])

    @staticmethod
    def exp(v):
        assert v.shape == (6,1)
        omega = v[:3]
        u = v[3:].copy()
        th = norm(omega)
        wx = skew(omega)

        # exponentiate the rotation
        q_exp = Quaternion.exp(omega)
        if th > 1e-16:
            B = (1. - np.cos(th)) / (th * th)
            C = (th - np.sin(th)) / (th * th * th)
            t_exp = (I_3x3 + B*wx + C*wx.dot(wx)).dot(u)
        else:
            t_exp = u
        return Transform(q_exp, t_exp)

    @staticmethod
    def log(T):
        assert isinstance(T, Transform)
        omega = Quaternion.log(T.q)
        th = norm(omega)
        if th > 1e-16:
            wx = skew(omega)
            A = np.sin(th)/th
            B = (1.-np.cos(th)) / (th*th)
            V = I_3x3 - (1./ 2.)*wx + (1./(th*th)) * (1.-(A/(2.*B)))*(wx.dot(wx))
        else:
            V = I_3x3
        u = V.dot(T.t)
        return np.vstack((omega, u))


    @staticmethod
    def Identity():
        return Transform(Quaternion.Identity(), np.zeros((3,1)))

    @staticmethod
    def random():
        pass

    @staticmethod
    def between_two_poses(T1, T2):
        pass

    def copy(self):
        return Transform(self.q.copy(), self.t.copy())

    def transforma(self, v):
        return self.q.rotp(v) + self.t

    def transformp(self, v):
        return self.q.rota(v - self.t)

    def invert(self):
        self.q.inv()
        self.t = -self.q.rotp(self.t)

    def otimes(self, T):
        return Transform(self.q * T.q, self.t + self.q.rotp(T.t))

    def boxplus(self, delta):
        pass

    def boxminus(self, T2):
        pass

    def plotlines(self):
        points = np.zeros((3,5))
        points[:,0::2] =  self.transforma(I_3x3)
        points[:,1::2] = self.t
        return points


if __name__ == '__main__':
    # Check Initialization and Print Functions
    q = Quaternion.random()
    t = np.random.random((3, 1))
    T1 = Transform(q, t)

    ## Check Inverse and Multiply functions
    # Check a Known transform and its inverse
    T_known = Transform(Quaternion.from_axis_angle(np.array([[0, 0, 1]]).T, np.pi/4.0),
                        np.array([[1.0, 1.0, 0]]).T)
    T_known_inv = Transform(Quaternion.from_axis_angle(np.array([[0, 0, 1.]]).T , -np.pi/4.0),
                            np.array([[0.0, -(2**0.5), 0]]).T)
    assert norm(T_known_inv.elements - T_known.inverse.elements) < 1e-8

    T2 = T1.inverse
    T3 = T1 * T2
    assert norm(T3.q - Quaternion.Identity()) < 1e-8, "Inverse didn't work"
    assert norm(T3.t) < 1e-8, "Inverse didn't work"

    # Transforming a Vector
    p = np.random.random((3,1))
    assert norm(T1.transformp(T1.inverse.transformp(p)) - p) < 1e-8
    assert norm(T1.inverse.transformp(T1.transformp(p)) - p) < 1e-8
    assert norm(T1.transforma(T1.inverse.transforma(p)) - p) < 1e-8
    assert norm(T1.inverse.transforma(T1.transforma(p)) - p) < 1e-8

    if False:
        # Plot an active and passive transformation
        p = np.array([[1, 0.2, 0]]).T
        p2 = np.array([[0.5, 1, 0]]).T
        import matplotlib.pyplot as plt
        p_active = T_known.transforma(p)
        p2_passive = T_known.inverse.transformp(p2) # represent p in the origin frame
        origin = Transform.Identity().plotlines()
        frame1 = T_known.plotlines()
        plt.plot(origin[0,:], origin[1,:], '-k')
        plt.plot(frame1[0,:], frame1[1,:], '-b')
        plt.plot([0,p_active[0,0]], [0, p_active[1,0]], '-m')
        plt.plot([0, p2_passive[0,0]], [0, p2_passive[1,0]], '-c')
        plt.show()

    # Check Log and Exp
    xi = np.random.random((6,1))
    assert(norm(Transform.log(Transform.exp(xi)) - xi) < 1e-8)

    



    print "Transform test [PASSED]"
    pass