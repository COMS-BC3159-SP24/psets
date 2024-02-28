from math import sin, cos, pi
import numpy as np
import copy

class Pendulum():
    def __init__(self, timestep = 0.15, gravity = 9.81, damping = 0.01, control_min = -5, control_max = 5, control_step = 0.25):
        self.timestep = timestep
        self.gravity = gravity
        self.damping = damping
        self.control_min = control_min
        self.control_max = control_max
        self.control_step = control_step
        self.Q = None
        self.R = None
        self.QF = None
        self.goal = None

    ######################
    #                    #
    # High level helpers #
    #                    #
    ######################

    # return the state and control size
    def get_state_size(self):
        return 2
    def get_control_size(self):
        return 1

    # clamp the input n to the range [smallest, largest]
    def clamp(self, n, smallest, largest): 
        return max(smallest, min(n, largest))

    # ensure angles are between -pi and pi
    def wrap_angles(self, angle):
        upper_bound = pi
        lower_bound = -pi
        while (angle < lower_bound):
            angle += 2*pi
        while (angle > upper_bound):
            angle -= 2*pi
        return np.array(angle)

    # compute the difference between two states making sure to wrap angles
    def state_delta(self, state1, state2):
        return np.array([self.wrap_angles(state1[0]-state2[0]), state1[1]-state2[1]])

    #####################################################
    #                                                   #
    #  Helper functions you need to fill out start here #
    #                                                   #
    #####################################################

    ######################
    #                    #
    #  Physics helpers   #
    #                    #
    ######################

    # apply the physics of the pendulum to compute acceleration
    def dynamics(self, x, u):
        position = x[0]
        velocity = x[1]
        qdd = u - self.gravity*sin(position) - self.damping*velocity
        return np.array(qdd)

    # compute the gradient of applying physics of the pendulum to compute acceleration
    def dynamics_gradient(self, x, u):
        position = x[0]
        velocity = x[1]
        # qdd = u - self.gravity*sin(position) - self.damping*velocity
        dqdd_dq = -self.gravity*cos(position)
        dqdd_dqd = -self.damping
        dqdd_du = 1
        return np.array([dqdd_dq, dqdd_dqd, dqdd_du])

    # compute the next state using euler integration
    def next_state(self, x, u):
        position = np.array(x[0])
        velocity = np.array(x[1])

        # first compute acceleration using dynamics
        # note: in this case we also clamp the input to control_min/max
        #       to avoid unrealistic steps (this can be safely ignored in the gradient)
        u_clamp = self.clamp(u, self.control_min, self.control_max)
        acceleration = self.dynamics(x,u_clamp)

        #
        # TODO
        #
        # Euler integrator [q', qd'] = [q, qd] + dt * [qd, qdd]
        # note: make sure to wrap_angles where appropriate
        #
        return np.array([0,0])

    # compute the gradient of the next state function using euler integration
    def next_state_gradient(self, x, u):
        position = x[0]
        velocity = x[1]
        A = np.array([[0., 0.], [0., 0.]])
        B = np.array([[0.], [0.]])

        #
        # TODO
        #
        # Euler integrator [q', qd'] = [q, qd] + dt * [qd, qdd]
        # Return the partial derivative matricies:
        #     A = [[dq'/dq, dq'/dqd],    and B = [[dq/du],
        #          [dqd'/dq, dqd'/dqd]]           [dqd/du]]
        #       
        #
        return A, B

    ######################
    #                    #
    #    Cost helpers    #
    #                    #
    ######################

    def set_Q(self, Q):
        self.Q = Q
    def set_R(self, R):
        self.R = R
    def set_QF(self, QF):
        self.QF = QF
    def set_goal(self, goal):
        self.goal = goal

    # compute the cost of the form 0.5(x-xg)^TQ(x-xg) + 0.5u^TRu
    # note that if u=None then it is the final state and there is no control (use QF)
    def cost_value(self, x, u = None):
        cost = 0
        # first compute the error between the current state and the goal
        delta = self.state_delta(x, self.goal)
        # then compute the cost for that error
        #
        # TODO
        #
        # hint: np.matmul may be useful!
        return 0

    # compute the gradient of the cost of the form 0.5(x-xg)^TQ(x-xg) + 0.5u^TRu
    # note that if u=None then it is the final state and there is no control (use QF)
    #
    # return the vector [dcost/dq, dcost/dqd, dcost/du]
    def cost_gradient(self, x, u = None):
        grad = []
        # first compute the error between the current state and the goal
        delta = self.state_delta(x, self.goal)
        # then compute the cost gradient with respect to the state
        #
        # TODO
        #
        if u is not None:
            return np.array([0,0,0])
        else:
            return np.array([0,0])

    # compute the hessian of the cost of the form 0.5(x-xg)^TQ(x-xg) + 0.5u^TRu
    # note that if u=None then it is the final state and there is no control (use QF)
    #
    # return the matrix [[d^2cost/dq^2,   d^2cost/dq dqd, d^2cost/dq du], 
    #                    [d^2cost/dqd dq, dcost/dqd^2,    d^2cost/dqd du],
    #                    [d^2cost/du dq,  dcost/du dqd,   d^2cost/du^2]]
    #
    # Hint: you may find the block_diag helper function helpful! (But you don't need to use it)
    #
    def cost_hessian(self, x, u = None):
        H = [[]]
        #
        # TODO
        #
        # hint: if u is None then you only need the hessian with respect to q, qd
        #
        if u is not None:
            return np.zeros((3,3))
        else:
            return np.zeros((2,2))

    def cost_gradient_hessian(self, x, u = None):
        return self.cost_hessian(x,u), self.cost_gradient(x,u)

    def block_diag(self, *arrs):
        """Create a block diagonal matrix from the provided arrays.

        Given the inputs `A`, `B` and `C`, the output will have these
        arrays arranged on the diagonal::

            [[A, 0, 0],
             [0, B, 0],
             [0, 0, C]]

        If all the input arrays are square, the output is known as a
        block diagonal matrix.

        Parameters
        ----------
        A, B, C, ... : array-like, up to 2D
            Input arrays.  A 1D array or array-like sequence with length n is
            treated as a 2D array with shape (1,n).

        Returns
        -------
        D : ndarray
            Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
            same dtype as `A`.

        References
        ----------
        .. [1] Wikipedia, "Block matrix",
               http://en.wikipedia.org/wiki/Block_diagonal_matrix

        Examples
        --------
        >>> A = [[1, 0],
        ...      [0, 1]]
        >>> B = [[3, 4, 5],
        ...      [6, 7, 8]]
        >>> C = [[7]]
        >>> print(block_diag(A, B, C))
        [[1 0 0 0 0 0]
         [0 1 0 0 0 0]
         [0 0 3 4 5 0]
         [0 0 6 7 8 0]
         [0 0 0 0 0 7]]
        >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
        array([[ 1.,  0.,  0.,  0.,  0.],
               [ 0.,  2.,  3.,  0.,  0.],
               [ 0.,  0.,  0.,  4.,  5.],
               [ 0.,  0.,  0.,  6.,  7.]])

        """
        if arrs == ():
            arrs = ([],)
        arrs = [np.atleast_2d(a) for a in arrs]

        bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
        if bad_args:
            raise ValueError("arguments in the following positions have dimension "
                                "greater than 2: %s" % bad_args) 

        shapes = np.array([a.shape for a in arrs])
        out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

        r, c = 0, 0
        for i, (rr, cc) in enumerate(shapes):
            out[r:r + rr, c:c + cc] = arrs[i]
            r += rr
            c += cc
        return out