
# PeNNdulum

## Comparing Reservoir Computing, Lagrangian, and Hamiltonian Neural Networks' Forecasts of Chaotic Systems in Physics

### Team Members

- Nathan Paik
- Guy Thampakkul
- Tai Xiang
- Ziang Xue

### Introduction

Lagrangian and Hamiltonian neural networks (LNN and HNN) output the Lagrangian and Hamiltonian equations for a system in motion. They were both developed for extremely physics-specific tasks, and this makes them relatively narrow in their scope. We seek to explore if they are capable of outperforming a more general-purpose neural network that is highly successful at predicting the behavior of chaotic systems, Reservoir Computing (RC). The task we will be using for comparison is the mechanics-based problem of forecasting the motion of a chaotic double pendulum. To verify this, we will train these networks on a dataset involving the initial conditions of a double pendulum and its subsequent path of motion. If RC outperforms the two physics-specific networks, then the utility of these networks substantially decreases. However, in this case, we expect LNN and HNN to surpass RC, as the most common mathematical way to solve for the equations of motion for a double pendulum is by first solving the Lagrangian or Hamiltonian. To get an understanding of the performance of each of these neural networks against a more common baseline model, we will also be comparing all three of these models against a recurrent-neural network that will serve as the control.

### Ethics Discussion (Possible Ethical Implications/ideas)

- Brute forcing: The project might require a massive video-based dataset. When training a model with such dataset it takes huge computing power to do so and will consume a considerable amount of energy. The energy used and the carbon-foot print of training such model could be a ethical concern.
- The uncertainty of NN: The predictions from the models are, after all, only predictions, and there's no guarantee of correctness. If some party put the trained model in real-world use and something goes wrong, causing damage either physically or financially, who should take the responsibility?
- The possibility of a NN "learning wrong": Neural networks have the capability to "learn", but their results may not always implicate successful learning. For example, in a real world scenario, if a NN learns a certain behavior, it may reflect racist or sexist tendencies that exist in the world without intentionally doing so. This includes areas like ad selection, image detection (mislabelling), resume classification, etc.

### Update 1

#### Software Used: 

For training our reservoir computing model, we will be using the [*easyESN*](https://github.com/kalekiu/easyesn) python library. To train our Lagrangian Neural Network, we will be writing some functions from scratch and imported pre-trained hyperparameters indicated by the Lagrangian Neural Network research paper. For the Hamiltonian Neural Networks, we can import the pre-trained models released by the team that published the paper on Hamiltonian Neural Networks, or train our own with PyTorch. We will train the recurrent neural network with PyTorch.

#### Dataset:

We will be using the [*IBM Double Pendulum Dataset*](https://ibm.github.io/double-pendulum-chaotic-dataset/). This is a dataset that consists of four frames of input for a double pendulum system, followed by 200 predicted frames. Furthermore, IBM lists of the camera parameters used for the construction of this dataset, thus we can create further training data. The physics department has a double pendulum filming setup as well as some matlab code that is capable of translating video inputs into cartesian coordinate outputs, and thus we can create more data if desired.

Quick note on getting the dataset: create a new folder `Data` in the repo, `cd` into it, and run:

```bash
wget https://dax-cdn.cdn.appdomain.cloud/dax-double-pendulum-chaotic/2.0.1/double-pendulum-chaotic.tar.gz
tar xzf double-pendulum-chaotic.tar.gz
```
I've already added a `.gitignore` file so it won't be pushed to the repo as long as the folder is called `Data` and it's right under `/physinet.io`.

##### Contents and Dimensions of Dataset:

We will mainly be using the coordinate info in the dataset and put aside the `.mkv` videos for now. The coordinates are stored in csv files and each frame has 6 entries: the `x` and `y` coordinate of the 3 axis of the double pendulum. We will be defining method that convert data for each frame into a `1x4` vector of: `sin_angle_green_red`, `cos_angle_green_red`, `sin_angle_blue_green`, 
`cos_angle_blue_green`. This vector is the sample.

There is an WIP Jupyter notebook that tries to make a baseline RNN out of these. I'm still going through the PyTorch docs to learn how to create RNNs.

#### General Overview:

We will be using recurrent neural networks, echo state networks (reservoir computing), Lagrangian neural networks, and Hamiltonian neural networks. In the case of the recurrent and echo state network, our data will be input as a vector describing the initial conditions. Our output for those two models will be a set of x and y coordinates corresponding to the path of the pendulum. These will just be a vector of floating-point values. We can then visualize these positions to get a general sense of the pendulum's travel over time.

For the Hamiltonian and Lagrangian neural networks, we will input a set of initial conditions. However, our corresponding outputs will be transformed to be a vector of the potential and kinetic energy of the system over time. With these outputs and the corresponding Hamiltonian and Lagrangian to the double pendulum system, we can construct the change in x and y coordinates over time to properly compare against the recurrent and echo state neural networks.

### Literature Review

Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020, July 30). [*Lagrangian neural networks*](https://arxiv.org/abs/2003.04630).

- Compared to regular NN, HNN are better in learning symmetries and conservation law, but requires coordinates of system to be canoncial. LNN does not require this. Theories about lagrangian is included and 3 different experiments are performed, including double pendulum. Codes are provided [here](https://github.com/MilesCranmer/lagrangian_nns). It also has a really helpful reference list.

Bollt, E. (2021, January 4). [*On explaining the surprising success of reservoir computing forecaster of chaos? The universal machine learning dynamical system with contrast to VAR and DMD*](https://aip.scitation.org/doi/abs/10.1063/5.0024890). American Association of Physics Teachers.

- An explanation on why reservoir computing succeeds at forecasting dynamical systems. Explains some of the foundational mathematics behind reservoir computing and benchmarks various iterations of reservoir computing on tasks involving dynamical systems and compares them against other architectures.

Shinbrot, T., Grebogi, C., Wisdom, J., & Yorke, J. A. (1992, June 1). [*Chaos in a double pendulum*](https://aapt.scitation.org/doi/10.1119/1.16860). American Association of Physics Teachers.

- This is a very basic introduction on double pendulums and some math included. Explains some of the fundamental mechanics behind the double pendulum and derives the the governing equation of motion for the problem with the Lagrangian formalism. Covers the exponentiation and mathematics behind why the system varies so drastically in response to changes in initial conditions.

Woolley, Jonathan W., P. K. Agarwal, and John Baker. [*Modeling and prediction of chaotic systems with artificial neural networks*](https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.2117). International journal for numerical methods in fluids 63.8 (2010): 989-1004.

- A paper about chaotic systems in general (not specific to double pendulum or phenomena in physics). Chaotic systems such as earthquakes, laser systems, epileptic seizures, combustion, and weather patterns are very difficult to predict. This study attempts to develop a system for training artificial neural networks t predict the future data of processes. Data set was obtained by solving Lorenz's equations. Backpropagation algorithm is used to train the network. "A correlation of 94% and a negative Lyapunov exponent indicate that the results obtained from ANN are in good agreement with the actual values."

Zhang, H., Fan, H., Wang, L., & Wang, X. (2021). [*Learning Hamiltonian dynamics with reservoir computing*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.024205). Physical Review E, 104(2), 024205.

- RC is good for state evolution prediction in Hamiltonian dynamics. They used parameter-aware RC to reconstruct the KAM diagram (something previously done with HNNs). RC is used in learning the behavior of a double pendulum. Compared to HNN, whose output depends solely on input at the current time, RC also takes into account the past states of the system. However, RC makes training simpler, since the Hamiltonian mechanisms are no longer pre-requisites.

Lutter, M., Ritter, C., &; Peters, J. (2019, July 10). [Deep lagrangian networks: Using physics as model prior for deep learning](https://arxiv.org/abs/1907.04490). 

- A network structure called Deep Lagrangian Networks is presented. Previous works seldomly combined NN and differential equations. The paper gives an introduction on Lagrangian mechanics and the math of fitting it into NN. The team did a 2-degree-of-freedom robot arm simulation, and the Deep Lagrangian Network learnt the physical model of the system. The double pendulum is a similar 2-degree-of-freedom problem that is based within classical mechanics.
