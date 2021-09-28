
# PeNNdulum

## Comparing Reservoir Computing, Lagrangian, and Hamiltonian Neural Networks' Forecasts of Chaotic Systems in Physics

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

- Brute forcing: The project might require a massive video-based dataset. When training a model with such dataset it takes huge computing power to do so and will consume a considerable amount of energy. The energy used and the carbon-foot print of training such model could be a ethical concern.
- The uncertainty of NN: The predictions from the models are, after all, only predictions, and there's no guarantee of correctness. If some party put the trained model in real-world use and something goes wrong, causing damage either physically or financially, who should take the responsibility?
- The possibility of a NN "learning wrong": Neural networks have the capability to "learn", but their results may not always implicate successful learning. For example, in a real world scenario, if a NN learns a certain behavior, it may reflect racist or sexist tendencies that exist in the world without intentionally doing so. This includes areas like ad selection, image detection (mislabelling), resume classification, etc.

### Literature Review
