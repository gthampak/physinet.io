# Comparing Reservoir Computing, Lagrangian, and Hamiltonian Neural Netowrks' Forecasts of Chaotic Systems in Physics (to be revised)

### Team Members

- Nathan Paik
- Guy Thampakkul
- Tai Xiang
- Ziang Xue

### Introduction

Lagrangian and Hamiltonian neural networks (LNN and HNN) output the Lagrangian and Hamiltonian equations for a system in motion. They were both developed for extremely physics-specific tasks, and this makes them relatively narrow in their scope. We seek to explore if they are capable of outperforming a more general-purpose neural network, Reservoir Computing (RC), at the mechanics-based task of forecasting the motion of a chaotic double pendulum. To verify this, we will train these networks on a dataset involving the initial conditions of a double pendulum and its subsequent path of motion. If RC outperforms the two physics-specific networks, then the utility of these networks substantially decreases. However, in this case, we expect LNN and HNN to surpass RC, as the most common mathematical way to solve for the equations of motion for a double pendulum is by first solving the Lagrangian or Hamiltonian.

### Plan of Action

### Possible Datasets

- https://ibm.github.io/double-pendulum-chaotic-dataset/



### Ethics Discussion (Possible Ethical Implications/ideas)

- Ethics of brute forcing (waste?)
- Ethics of "black-box side" of NNs
