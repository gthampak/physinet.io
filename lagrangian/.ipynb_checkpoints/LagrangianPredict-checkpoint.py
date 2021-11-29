# +
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import pickle
from functools import partial # reduces arguments to function by making some subset implicit

from jax.experimental import stax
from jax.experimental import optimizers

# visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from moviepy.editor import ImageSequenceClip
from functools import partial
import proglog
from PIL import Image

from simulate_data import generate_train_ideal, solve_lagrangian, init_nn, normalize_dp, solve_analytical


# +
@jax.jit
def update_timestep(i, opt_state, batch):
  params = get_params(opt_state)
  return opt_update(i, jax.grad(loss)(params, batch, time_step), opt_state)

@jax.jit
def update_derivative(i, opt_state, batch):
  params = get_params(opt_state)
  return opt_update(i, jax.grad(loss)(params, batch, None), opt_state)


# +
# replace the lagrangian with a parameteric model
def learned_lagrangian(params):
  def lagrangian(q, q_t):
    assert q.shape == (2,)
    state = normalize_dp(jnp.concatenate([q, q_t]))
    return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
  return lagrangian

# define the loss of the model (MSE between predicted q, \dot q and targets)
@jax.jit
def loss(params, batch, time_step=None):
  state, targets = batch
  if time_step is not None:
    f = partial(equation_of_motion, learned_lagrangian(params))
    preds = jax.vmap(partial(rk4_step, f, t=0.0, h=time_step))(state)
  else:
    preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(params)))(state)
  return jnp.mean((preds - targets) ** 2)


# +
def make_plot(i, cart_coords, l1, l2, max_trail=30, trail_segments=20, r = 0.05):
    # Plot and save an image of the double pendulum configuration for time step i.
    plt.cla()

    x1, y1, x2, y2 = cart_coords
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k') # rods
    c0 = Circle((0, 0), r/2, fc='k', zorder=10) # anchor point
    c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10) # mass 1
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10) # mass 2
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # plot the pendulum trail (ns = number of segments)
    s = max_trail // trail_segments
    for j in range(trail_segments):
        imin = i - (trail_segments-j)*s
        if imin < 0: continue
        imax = imin + s + 1
        alpha = (j/trail_segments)**2 # fade the trail into alpha
        ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                lw=2, alpha=alpha)

    # Center the image on the fixed anchor point. Make axes equal.
    ax.set_xlim(-l1-l2-r, l1+l2+r)
    ax.set_ylim(-l1-l2-r, l1+l2+r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.savefig('./frames/_img{:04d}.png'.format(i//di), dpi=72)

def radial2cartesian(t1, t2, l1, l2):
  # Convert from radial to Cartesian coordinates.
  x1 = l1 * np.sin(t1)
  y1 = -l1 * np.cos(t1)
  x2 = x1 + l2 * np.sin(t2)
  y2 = y1 - l2 * np.cos(t2)
  return x1, y1, x2, y2

def fig2image(fig):
  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return image


# -

params = pickle.load(open('LNN_Params', 'rb'))

init_random_params, nn_forward_fn = init_nn()

# choose an initial state
x1 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
t2 = np.linspace(0, 20, num=301)

# predictions from LNN
x1_model = jax.device_get(solve_lagrangian(learned_lagrangian(params), x1, t=t2)) 

# analytical solution
x1_analytical = jax.device_get(solve_analytical(x1, t2))

L1, L2 = 1, 1
theta1_ana, theta2_ana = x1_analytical[:, 0], x1_analytical[:, 1]
cart_coords_ana = radial2cartesian(theta1_ana, theta1_ana, L1, L2)

L1, L2 = 1, 1
theta1_model, theta2_model = x1_model[:, 0], x1_model[:, 1]
cart_coords_model = radial2cartesian(theta1_model, theta1_model, L1, L2)


def cart_coords_over_time(cart_coords):
    x1, y1, x2, y2 = cart_coords

    plt.scatter(x1, y1, marker="s", label='first')
    plt.scatter(x2, y2, marker="o", label='second')
    
    plt.show()


cart_coords_over_time(cart_coords_model)



cart_coords_over_time(cart_coords_ana)


def cart_error_over_time(cart_coords_model, cart_coords_ana):
    x11, y11, x21, y21 = cart_coords_ana
    x12, y12, x22, y22 = cart_coords_model
    
    distance_1 = np.sqrt((np.square(np.absolute(x11 - x12)) + np.square(np.absolute(y11 - y12))))
    distance_2 = np.sqrt((np.square(np.absolute(x21 - x22)) + np.square(np.absolute(y21 - y22))))
    
    index = np.arange(np.size(distance_1))
    
    plt.plot(index, distance_1, label='first')
    plt.plot(index, distance_2, label='second')
    
    plt.show

    plt.show()


cart_error_over_time(cart_coords_model, cart_coords_ana)


