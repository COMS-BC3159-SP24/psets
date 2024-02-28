# Exploring Taylor Expansions and DDP

**NOTE: You need to install some python pacakges `pip install -r requirements.txt`**

**Files in this assignment:**
`pendulum.py`   Defines a series of (very useful) helper functions related to the robot (the pendulum) we are working with.
                You will need to finish some of the implementation here (and thus submit to Gradescope for Autograding).

`trajopt.py`    Defines the high level trajectory optimzation functions that call the helper dynamics functions in `pendulum` in order to solve trajopt problems. You DO NOT need to submit this file.

`runPend.py`    Sets up some constants and calls the trajopt functions to run and displays the solve steps graphically (if you are running it locally). You DO NOT need to submit this file.

## Part 1 - Taylor Approximations of Dynamics and Cost Functions (8 Points)

We'll be working in the `pendulum.py` file and we'll be implementing a few dynamics and cost helper functions that we'll use in the larger trajopt functions (but we won't implement those today).

Functions to update and their points:
* `next_state` 2 points
* `next_state_gradient` 3 points
* `cost_value` 1 point
* `cost_gradient` 1 point
* `cost_hessian` 1 point

We'll begin by converting the pendulum physics functions which compute acceleration (`dynamics`) and its gradient (`dynamics_gradient`) into `next_state` and `next_state_gradient` functions through the use of Euler integration. Recall from class that Euler integration adds a small amount of acceleration and velocity to the original position and velocity:
```
`[q', qd'] = [q, qd] + dt * [qd, qdd]`
```
Also recall that the dynamics derivative matricies are:
```
             A = [[dq'/dq, dq'/dqd],    and B = [[dq/du],
                  [dqd'/dq, dqd'/dqd]]           [dqd/du]]
```

Next we'll move onto the cost functions. Here we need to implement the cost value, gradient, and hessian. For this problem we are going to use the "standard simple quadratic cost":
```
0.5(x-xg)^TQ(x-xg) + 0.5u^TRu
```
As that outputs a scalar value the cost gradient and hessian therefore are of the form:
```
g = [dcost/dq, dcost/dqd, dcost/du]
H = [[d^2cost/dq^2,   d^2cost/dq dqd, d^2cost/dq du], 
     [d^2cost/dqd dq, dcost/dqd^2,    d^2cost/dqd du],
     [d^2cost/du dq,  dcost/du dqd,   d^2cost/du^2]]
```

Note that most things are of type [`np.array`](https://numpy.org/doc/stable/user/quickstart.html) which hopefully should simplify your linear algebra and there are a few more hints left in the code comments! Also, make sure to take into account both the states and the controls!

## Part 2 - Explore
Now that you have a working DDP implementation run the `runPend.py` file. What happens? How does this work as compared to the SQP algorithm? Is it better? Worse? Slower? Faster? Does the trajectory go all the way to the goal? If you change the parameters for the cost function by adjusting `Q` and `R` in that file what happens? Play around with it a little bit. Hopefully it will give you a little better intuition for what the math is doing!