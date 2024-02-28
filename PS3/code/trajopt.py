# trajopt.py
# This program runs the main algorithms built on top of your util functions
# (Do not modify, but please read)
#
# Brian Plancher - Spring 2024
# Adapted from code written by Rus Tedrake and Scott Kuindersma

import sys, math, pygame, copy
from pygame.locals import *
from time import sleep
import numpy as np
import scipy as sp
np.set_printoptions(precision=3) # so things print nicer

class Trajopt:
    def __init__(self, robot_object, start_node, goal_node, N, XMAX_MIN, YMAX_MIN, \
                       MAX_ITER = 100, EXIT_TOL = 1e-3, ALPHA_FACTOR = 0.5, ALPHA_MIN = 1e-4, MU = 5):
        self.robot_object = robot_object # the robot_object with physics and cost functions
        self.MAX_ITER = MAX_ITER         # total ddp loops to try
        self.EXIT_TOL = EXIT_TOL         # This is the convergence criterion. We will declare success when the trajectory
                                         # is updated by a norm of less that 1e-4. DO NOT MODIFY.
        self.N = N                       # Number of nodes in a trajectory
        self.start_node = start_node
        self.goal_node = goal_node
        self.XMAX_MIN = XMAX_MIN         # max for drawing locically
        self.YMAX_MIN = YMAX_MIN         # max for drawing locically
        self.canvas_max = 640            # max for drawing pixels
        # set global line search parameters
        self.alpha_factor = ALPHA_FACTOR # how much to reduce alpha by each deeper search
        self.alpha_min = ALPHA_MIN       # minimum alpha to try
        self.mu = MU                     # merit function weighting        

    def draw_circle(self, node, size, color):
        percent_x = (node[0] + self.XMAX_MIN) / 2 / self.XMAX_MIN
        percent_y = (node[1] + self.YMAX_MIN) / 2 / self.YMAX_MIN
    
        scaled_x = int(self.canvas_max*percent_x)
        scaled_y = int(self.canvas_max*percent_y)    

        pygame.draw.circle(self.screen, color, (scaled_x, scaled_y), size)

    def draw_line(self, node1, node2, color):
        percent_x1 = (node1[0] + self.XMAX_MIN) / 2 / self.XMAX_MIN
        percent_y1 = (node1[1] + self.YMAX_MIN) / 2 / self.YMAX_MIN
        percent_x2 = (node2[0] + self.XMAX_MIN) / 2 / self.XMAX_MIN
        percent_y2 = (node2[1] + self.YMAX_MIN) / 2 / self.YMAX_MIN
    
        scaled_x1 = int(self.canvas_max*percent_x1)
        scaled_y1 = int(self.canvas_max*percent_y1)
        scaled_x2 = int(self.canvas_max*percent_x2)
        scaled_y2 = int(self.canvas_max*percent_y2)

        # check if angle wrapping (and don't draw lines across the screen)
        if node1[0] - node2[0] > math.pi:
            pass
        elif node1[0] - node2[0] < -math.pi:
            pass
        else:
            pygame.draw.line(self.screen, color, (scaled_x2, scaled_y2), (scaled_x1, scaled_y1))

    def init_screen(self):
        # initialize and prepare screen
        pygame.init()
        self.screen = pygame.display.set_mode((self.canvas_max,self.canvas_max))
        pygame.display.set_caption('PS3 - DDP')
        black = 20, 20, 40
        blue = 0, 0, 255
        green = 0, 255, 0
        self.screen.fill(black)
        self.draw_circle(self.start_node, 5, blue)
        self.draw_circle(self.goal_node, 5, green)
        pygame.display.update()

    def draw_trajectory(self, x):
        white = 255, 240, 200
        red = 255, 0, 0
        self.draw_circle(x[:,0], 2, white)
        for k in range(1,self.N):
            self.draw_circle(x[:,k], 2, white)
            self.draw_line(x[:,k-1], x[:,k], red)
        pygame.display.update()
        sleep(0.4)

    def wait_to_exit(self, x, u, DISPLAY_MODE = True):
        # if in test mode return the path
        if not DISPLAY_MODE:
            return x, u, K
        # Else wait for the user to see the solution to exit
        else:
            while(1):
                for e in pygame.event.get():
                    if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                        sys.exit("Leaving because you requested it.")


    def solve(self, x, u, N, DISPLAY_MODE = False):
        # start up the graphics
        self.init_screen()

        # get constants
        nx = self.robot_object.get_state_size()
        nu = self.robot_object.get_control_size()

        # solve
        self.iLQR(x, u, nx, nu, N, DISPLAY_MODE)


    #####################################################
    #                                                   #
    #                    iLQR Helpers                   #
    #                                                   #
    #####################################################

    # Compute the total cost along the trajectory
    def compute_total_cost(self, x, u, nx, nu, N):
        J = 0
        for k in range(N-1):
            J += self.robot_object.cost_value(x[:,k], u[:,k])
        J += self.robot_object.cost_value(x[:,N-1])
        return J

    # compute the dynamics and cost gradient and hessians
    def compute_approximation(self, xk, uk, nx, nu, k):
        Gk, gk = self.robot_object.cost_gradient_hessian(xk, uk)
        Ak, Bk = self.robot_object.next_state_gradient(xk, uk)
        return Ak, Bk, Gk, gk

    # put most of the above backward pass functions together and get from
    # the inputs to return the current kappa, K, Vxx, Vx
    def backward_pass_iterate(self, A, B, H, g, Vxx_kp1, Vx_kp1, nx, nu, k):
        # Backpropogate the CTG through the dynamics
        Hxx = np.matmul(A.transpose(),np.matmul(Vxx_kp1,A)) + H[0:nx,0:nx]
        Huu = np.matmul(B.transpose(),np.matmul(Vxx_kp1,B)) + H[nx:,nx:]
        Hux = np.matmul(B.transpose(),np.matmul(Vxx_kp1,A)) + H[nx:,0:nx]
        gx = np.matmul(A.transpose(),Vx_kp1) + g[0:nx]
        gu = np.matmul(B.transpose(),Vx_kp1) + g[nx:]

        # Invert Huu block
        HuuInv = np.linalg.inv(Huu)

        # Compute feedback and feedforward updates
        K = -np.matmul(HuuInv,Hux)
        kappa = -np.matmul(HuuInv,gu)
        
        # Compute the next CTG estimate
        Vxx = Hxx - np.matmul(Hux.transpose(),K)
        Vx = gx - np.matmul(Hux.transpose(),kappa)

        return kappa, K, Vxx, Vx

    # return u_new based on the current x_new, k, kappa, alpha, and original x
    def compute_control_update(self, x, x_new, K, kappa, alpha, nx, nu, N):
        delta = self.robot_object.state_delta(x_new, x)
        return alpha*kappa + np.matmul(K,delta)

    # rollout the full trajectory to produce x_new, u_new based on
    # x, u (the original trajectory), as well as K, kappa, alpha (the updates)
    def rollout_trajectory(self, x, u, K, kappa, alpha, nx, nu, N):
        x_new = copy.deepcopy(x)
        u_new = copy.deepcopy(u)
        for k in range(N-1):
            u_new[:,k] = u[:,k] + self.compute_control_update(x[:,k], x_new[:,k], K[:,:,k], kappa[:,k], alpha, nx, nu, k)
            x_new[:,k+1] = self.robot_object.next_state(x_new[:,k], u_new[:,k])
        return x_new, u_new

    # main iLQR function
    def iLQR(self, x, u, nx, nu, N, DISPLAY_MODE = False):
        
        # allocate memory for things we will compute
        kappa = np.zeros([nu,N-1])     # control updates
        K = np.zeros((nu,nx,N-1))   # feedback gains

        # compute initial cost
        J = self.compute_total_cost(x, u, nx, nu, N)
        J_prev = J
        if DISPLAY_MODE:
            print("Initial Cost: ", J)

        # start the main loop
        iteration = 0
        failed = False
        while 1:

            # Do backwards pass to compute new control update and feedback gains: kappa and K
            # start by initializing the cost to go
            Vxx, Vx = self.robot_object.cost_gradient_hessian(x[:,N-1])
            for k in range(N-2,-1,-1):
                # then compute the quadratic approximation at that point
                A, B, H, g = self.compute_approximation(x[:,k], u[:,k], nx, nu, k)

                # then compute the control update and new CTG estimates
                kappak, Kk, Vxx, Vx = self.backward_pass_iterate(A, B, H, g, Vxx, Vx, nx, nu, k)

                # save kappa and K for forward pass
                kappa[:,k] = kappak.tolist()
                K[:,:,k] = Kk.tolist()

            # Do forwards pass to compute new x, u, J (with line search)
            alpha = 1
            while 1:
                # rollout new trajectory
                x_new, u_new = self.rollout_trajectory(x, u, K, kappa, alpha, nx, nu, N)

                # compute new cost
                J_new = self.compute_total_cost(x_new, u_new, nx, nu, N)
                    
                # simple line search criteria
                if J_new < J:
                    J_prev = J
                    x = x_new
                    u = u_new
                    J = J_new
                    if DISPLAY_MODE:
                        print("Iteration[", iteration, "] with cost[", round(J,4), "] and x final:")
                        print(x[:,N-1])
                        self.draw_trajectory(x)
                    break

                # failed try to continue the line search
                elif alpha > self.alpha_min:
                    alpha *= self.alpha_factor
                    if DISPLAY_MODE:
                        print("Deepening the line search")
                
                # failed line search
                else:
                    if DISPLAY_MODE:
                        print("Line search failed")
                    failed = True
                    break

            # Check for exit (or error)
            if failed: # need double break to get out of both loops here if line search fails
                break

            delta_J = J_prev - J
            if delta_J < self.EXIT_TOL:
                if DISPLAY_MODE:
                    print("Exiting for exit_tolerance")
                break
            
            if iteration == self.MAX_ITER - 1:
                if DISPLAY_MODE:
                    print("Exiting for max_iter")
                break
            else:
                iteration += 1

        if DISPLAY_MODE:
            print("Final Trajectory")
            print(x)
            print(u)

        self.wait_to_exit(x, u, DISPLAY_MODE)