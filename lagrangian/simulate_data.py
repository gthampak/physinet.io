# +
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial # reduces arguments to function by making some subset implicit

from jax.experimental import stax
from jax.experimental import optimizers
# -

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


# We need to get a way to write the analytical solutions to the double pendulum so we can get data

# general form of lagrangian
def lagrangian(q, q_dot, m1, m2, l1, l2, g):
  t1, t2 = q     # theta 1 and theta 2
  w1, w2 = q_dot # omega 1 and omega 2

  # kinetic energy (T)
  T1 = 0.5 * m1 * (l1 * w1)**2
  T2 = 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 +
                    2 * l1 * l2 * w1 * w2 * jnp.cos(t1 - t2))
  T = T1 + T2
  
  # potential energy (V)
  y1 = -l1 * jnp.cos(t1)
  y2 = y1 - l2 * jnp.cos(t2)
  V = m1 * g * y1 + m2 * g * y2

  return T - V


# t1 and t2 correspond to the angles theta 1 theta 2, and w1 w2 correspond to the angular velocities omega 1 omega 2

# analytical solution for dbl pendulum
def f_analytical(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
  t1, t2, w1, w2 = state
  a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(t1 - t2)
  a2 = (l1 / l2) * jnp.cos(t1 - t2)
  f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * jnp.sin(t1 - t2) - \
      (g / l1) * jnp.sin(t1)
  f2 = (l1 / l2) * (w1**2) * jnp.sin(t1 - t2) - (g / l2) * jnp.sin(t2)
  g1 = (f1 - a1 * f2) / (1 - a1 * a2)
  g2 = (f2 - a2 * f1) / (1 - a1 * a2)
  return jnp.stack([w1, w2, g1, g2])


# equations of motion
def equation_of_motion(lagrangian, state, t=None):
  q, q_t = jnp.split(state, 2)
  q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
          @ (jax.grad(lagrangian, 0)(q, q_t)
             - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
  return jnp.concatenate([q_t, q_tt])


# solving the lagrangian
def solve_lagrangian(lagrangian, initial_state, **kwargs):
  @partial(jax.jit, backend='cpu')
  def f(initial_state):
    return odeint(partial(equation_of_motion, lagrangian),
                  initial_state, **kwargs)
  return f(initial_state)


# +
# Double pendulum dynamics via the rewritten Euler-Lagrange
@partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times, m1=1, m2=1, l1=1, l2=1, g=9.8):
  L = partial(lagrangian, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
  return solve_lagrangian(L, initial_state, t=times, rtol=1e-10, atol=1e-10)

# Double pendulum dynamics via analytical forces taken from Diego's blog
@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state, times):
  return odeint(f_analytical, initial_state, t=times, rtol=1e-10, atol=1e-10)

# used to keep polar coordinates from getting big and weird
def normalize_dp(state):
  # wrap generalized coordinates to [-pi, pi]
  return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])

# used to get solutions of differential equations
def rk4_step(f, x, t, h):
  # one step of runge-kutta integration
  k1 = h * f(x, t)
  k2 = h * f(x + k1/2, t + h/2)
  k3 = h * f(x + k2/2, t + h/2)
  k4 = h * f(x + k3, t + h)
  return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)


# -

def generate_train_ideal():
    print("generating data...")
    time_step = 0.01
    N = 2000
    # 20 seconds of data
    
    analytical_step = jax.jit(jax.vmap(partial(rk4_step, f_analytical, t=0.0, h=time_step)))
    
    # initialize some state with some theta and 0 angular velocities
    x0 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
    t = np.arange(N, dtype=np.float32) # time steps 0 to N
    
    # dynamics for first N time steps that describes how the pendulum moves
    x_train = jax.device_get(solve_analytical(x0, t))
    
    # time derivatives for each of the x_train states
    xt_train = jax.device_get(jax.vmap(f_analytical)(x_train))
    
    # the next part of the time step that you get by performing a step of runge kutta integration
    # this time evolves the double pendulum and basically gets where it will be next
    y_train = jax.device_get(analytical_step(x_train))
    
    # same thing, now testing data but for further time
    t_test = np.arange(N, 2*N, dtype=np.float32) # time steps N to 2N
    x_test = jax.device_get(solve_analytical(x0, t_test)) # dynamics for next N time steps
    xt_test = jax.device_get(jax.vmap(f_analytical)(x_test)) # time derivatives of each state
    y_test = jax.device_get(analytical_step(x_test)) # analytical next step
    
    # x_train are the actual dynamics, xt_train are the time derivatives, and y_train are the next steps
    return x_train, xt_train, y_train, x_test, xt_test, y_test


# x_0 is some random starting sequence

def generate_train_noisy():
    print("generating noisy data...")
    
    time_step = 0.01
    N = 2000
    # 20 seconds of data
    
    analytical_step = jax.jit(jax.vmap(partial(rk4_step, f_analytical, t=0.0, h=time_step)))
    
    # initialize some state with some theta and 0 angular velocities
    x0 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
    t = np.arange(N, dtype=np.float32) # time steps 0 to N
    
    noise = np.random.RandomState(0).randn(x0.size)
    noise_coeff = 1e-3
    
    # dynamics for first N time steps that describes how the pendulum moves
    
    x_train = jax.device_get(solve_analytical(x0 + noise_coeff * noise, t))
    
    xt_train = jax.device_get(jax.vmap(f_analytical)(x_train))
    
    # the next part of the time step that you get by performing a step of runge kutta integration
    # this time evolves the double pendulum and basically gets where it will be next
    y_train = jax.device_get(analytical_step(x_train))
    
    # same thing, now testing data but for further time
    t_test = np.arange(N, 2*N, dtype=np.float32) # time steps N to 2N
    x_test = jax.device_get(solve_analytical(x0 + noise_coeff * noise, t_test)) # dynamics for next N time steps
    xt_test = jax.device_get(jax.vmap(f_analytical)(x_test)) # time derivatives of each state
    y_test = jax.device_get(analytical_step(x_test)) # analytical next step
    
    # x_train are the actual dynamics, xt_train are the time derivatives, and y_train are the next steps
    return x_train, xt_train, y_train, x_test, xt_test, y_test



generate_train_noisy()

generate_train_ideal()
