from trajopt import Trajopt
from pendulum import Pendulum
from math import pi
import numpy as np
import sys

XMAX_MIN = pi
YMAX_MIN = 10

X_START = [0,0]
X_GOAL = [pi,0]

# How did I get these numbers? A little intuition and a lot of guess and check.
QF = np.array([[30,0],[0,10]])
Q = np.array([[3,0],[0,1]])
R = np.array([0.08])

N = 32

pend = Pendulum()
pend.set_Q(Q)
pend.set_R(R)
pend.set_QF(QF)
pend.set_goal(X_GOAL)

x0 = np.zeros([pend.get_state_size(),N])
u0 = np.zeros([pend.get_control_size(),N-1])

trajopt_obj = Trajopt(pend, X_START, X_GOAL, N, XMAX_MIN, YMAX_MIN)

error = trajopt_obj.solve(x0, u0, N, True)