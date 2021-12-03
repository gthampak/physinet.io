

# PeNNdulum

## Comparing Reservoir Computing, Lagrangian, and Hamiltonian Neural Networks' Forecasts of Chaotic Systems in Physics

### Team Members

- Nathan Paik
- Guy Thampakkul
- Tai Xiang
- Ziang Xue

***

### Introduction

Lagrangian neural networks (LNN) output the Lagrangian for a system in motion. The Lagrangian characterizes the total energy in the system, and is a useful tool for solving for the mechanics of a system where a wider and more general range of dynamics are needed. The LNN was developed for extremely physics-specific tasks, and this makes it relatively narrow in scope. We seek to explore if they are capable of outperforming a more general-purpose neural network that is highly successful at predicting the behavior of chaotic systems, Reservoir Computing (RC), as well as a baseline fully-connected neural network model (FC), the long-short term memory neural network (LSTM). 

The task we will be using for comparison is the mechanics-based problem of forecasting the motion of a chaotic double pendulum. To verify this, we will train these networks on a dataset that is the analytical solution of a double pendulum over time. If RC and LSTM outperforms the physics-specific network, the LNN, then the utility of these networks substantially decreases. However, in this case, we expect LNN to surpass RC and LSTM, as the most common mathematical way to solve for the equations of motion for a double pendulum is by first solving the Lagrangian. Furthermore, we expect the LNN to yield better results than both RC and LSTM as its output has been proven to abide by the laws of physics and conserve energy, whereas RC and LSTM may not necessarily yield outputs that fall within physical constraints. 

To get an understanding of the performance of each of these neural networks against a more common baseline model, we will also be comparing all three of these models against a generic feed-forward network that will serve as the control.

Finally, to test the validity of each model on new data, we will test each model on a set of simulated data, giving them each the same initial conditions. We can then both quantitatively and qualitatively evaluate how each model does to get a better understanding of their respective strengths and weaknesses.

#### The Double Pendulum

We choose double pendulum forecasting as our task as it is a dynamical system that is governed by a set of ordinary differential equations that yield substantial chaos. The system consists of a single pendulum with another one attached to the first pendulum's end.

`TODO:` add multiple side by side video comparisons of different initial conditions

From the above videos, we observe that slight differences in the initial conditions of the system result in drastically different paths and trajectories. This makes the system extraordinarily hard to forecast, for slight deviations in path result in substantial error. We can observe this in the equations of motion, which are reached through a derivation with the Lagrangian:

<center><img src="plots/dbl_pend_sample.png" width="325" /></center>

<br>

<center>

$\ddot{\theta_1} = \frac{-m_2cos(\theta_1 - \theta_2)l_1 \dot{\theta_1}^2sin(\theta_1 - \theta_2) + m_2cos(\theta_1 - \theta_2)gsin(\theta_2) - m_2 l_2\dot{\theta_2}^2sin(\theta_1-\theta_2)-(m_1+m_2)gsin(\theta_1)}{l_1(m_1+m_2-m_2cos^2(\theta_1 - \theta_2))}$

<br>

$\ddot{\theta_2} = \frac{(m1+m2)[l_1 \dot{\theta_1}^2 + \frac{\dot{\theta_2}^2sin(\theta_1 - \theta_2)cos(\theta_1 - \theta_2)m_2l_2}{m1+m2} + cos(\theta_1 - \theta_2)gsin(\theta_1) - gsin(\theta_2)]}{l_2 (m_1 + m_2 sin^2(\theta_1 - \theta_2))}$

</center>

where $\theta_1$ and $\theta_2$ describes the angles created between the pendulum arms and the vertical plane, $m_1$ and $m_2$ describe the masses of the first and second pendulum respectively, and $l_1$ and $l_2$ describe the arm lengths of the first and second pendulum directly. Observing the exponential terms in both equations, it can be inferred that any error will propogate through the system at a large scale, making accurate forecasting of the path to be quite difficult.

### Related works

Similar work on chaotic systems and the double pendulum has been done before. Klinkachorn and Parmar at Stanford characterized the performance of neural networks on double pendulum's as the starting angle between the two pendulum arm's began to vary. However, they tested a range of machine learning algorithms and models, including linear regression, autoregression, feed-forward neural networks, and long-short term memory networks. Rudy et al. also demonstrated a novel method to train models that seek to fit dynamical systems on noisy data, and in this paper compare increasing levels of variance that arise when a neural network is used to predict an increasingly noisy double pendulum input.

Although sharing multiple similarities, our work primarily differs in that we seek to test models that hypothetically ought to perform quite well on this task. The RNN, a more primitive analog to the LSTM, is simply used as a baseline instead of as the most advanced model, and we extend upon prior work by testing LNN, HNN, and RC on the double pendulum task.

### Methods Overview

The dataset used is a double pendulum simulation dataset that generates 1500 analytical solutions to the double pendulum problem given a set of initial conditions. This is done computationally and represents the ground truth solution that is derived through the double pendulum's equations of motion. Initially, we planned on building our own dataset with some computer vision code and a double pendulum setup provided by Pomona College Physics Department to test our trained networks on noisy systems (real world double pendulum), however we did not have time to complete this task. The simulated dataset, the IBM dataset, and computer vision dataset provides us with data on the same system with increasing levels of noise.

We trained and optimized a baseline fully-connected neural network (FC) long-short term neural network (LSTM), an echo state network (reservoir computing), and a Lagrangian Neural Network. We used PyTorch to implement and train our recurrent neural network and [ReservoirPy](https://github.com/reservoirpy/reservoirpy) to implement and train our echo state network. ReservoirPy is a library on github based on Python scientific libraries used as a tool to help implement efficient Reservoir Computing Neural Networks, specifically Echo State Networks. In the process of exploring ESN libraries, we also looked at "easy-esn", "pytorch-esn", or "EchoTorch". Because Lagrangian Neural Networks are more physics and mathematics intensive and unique than mainstream neural networks, ours was constructed from scratch using existing examples online and on github. The work we relied relatively heavily on in the construction of our Lagrangian Neural Network is Miles Cranmer et al.’s [paper](https://arxiv.org/abs/2003.04630) and [github repo]((https://github.com/MilesCranmer/lagrangian_nns)) on LNNs with dependencies on more mainstream Python libraries including Jax, NumPy, MoviePy, and celluloid, with the latter two used for visualization purposes.

For analysis, we wrote graphing functions that emulates paths of the double pendulums under different initial conditions over time consistent with the laws of physics. We overlaid the theoretical paths with our network-generated paths to get a clear visual representation of how the different networks perform. We also plotted the differences in predicted angles and analytically calculated angles, as well as the distance between pendulum endpoints for the predicted cartesian coordinates and analytical cartesian coordinates. Traditionally, F1 score, accuracy, precision, recall, etc. would be considered in the evaluation of model performance, but due to the nature of error propogation for the double pendulum, these metrics are not a good grounds for comparison.

- Data Wrangling and Preprocessing

Pre-processing involves unpacking and preparing the dataset for training uses. The datasets contain raw data of initial conditions and 2000 frames of the pendulum path. To make this usable for our training model, we first converted the raw coordinates to pixel (cartesian) coordinates, and then transformed those to polar coordinates. We decided to use polar coordinates as they encode information on both position and angle, which is especially effective for coupled oscillators. The data can now be fed into the network for training. We then separated our datasets into training data (for feeding into our networks) and training data for final network evaluations and network comparisons.

- Long-Short Term Neural Network

The LSTM was set up using PyTorch libraries. We fed training data into the network and performed hyperparameter optimization to attain a baseline model.

- Reservoir Computing

We trained the echo state network through ReservoirPy Python libraries. We also performed hyperparameter optimization on this model as well. The hyperparameter process differed from traditional neural networks (LSTM), as we iterated through ESN specific parameters such as number of reservoirs, leaking rate, spectral radius, and regression parameters.

- Lagrangian Neural Network

The Lagrangian Neural Network was written from scratch as their underlying mathematical equations and layers differ from conventional neural networks. Our code was written based on existing notebooks that have implemented these networks, most of which accompany their research papers from the same authors (see literature review of papers and links to code). For these networks, optimal hyperparameters are provided by the researchers and authors, so we did not have to optimize the parameters ourselves.

- Comparison and Results

To compare these networks, we looked at validation loss and accuracy, and comparing how well they perform on the testing set that was generated through simulation of a different set of initial conditions.

***

### Discussion

#### Results and Network Comparisons

To compare these 3 networks, we looked at validation loss and accuracy, and comparing how well they perform on the testing set that was segmented from the IBM dataset. After building a LSTM model, a LNN model and trying to build an ESN model, we have find out the following:

**General Overview**

We observe the predicted trajectories of each of the models overtime as well as an analytically solved solution:

![Analytical Results](plots/LNN_pendulum_positions_analytic.png) ![LNN Results](plots/LNN_pendulum_positions_model.png) ![ESN Results](plots/ESN_pendulum_positions_model.png) ![LSTM Results](plots/LSTM_pendulum_positions_standard.png) ![FC Results](plots/FC_pendulum_positions.png)

We can also visualize each of these positions continuously in a video:

<p float="left">
  <img src="plots/analytic.gif" width="500" />
  <img src="plots/lnn.gif" width="500" />
  <img src="plots/esn.gif" width="500" />
  <img src="plots/lstm.gif" width="500" />
  <img src="plots/fc.gif" width="500" />
</p>
<center>From left to right, the analytical video, LNN predicted, ESN predicted, LSTM predicted, FC predicted </center>
<br>

From the video, one clear advantage of the LNN in the case of this system is its preservation of physics. Though the path traced does not perfectly align with the analytical solution, the LNN conserves energy in the system, and the total amount of potential and kinetic energy are balanced accordingly. However, the ESN does not abide by the laws of physics, and tends to move about randomly and disobey gravity as well as conservation of energy.

From this broad overview, it is quite obvious that the LNN seems to match the analytical solution the most, while the ESN completely departs from the analytical solution. Taking a look at difference between the distance between x and y positions of the analytical solution and each model prediction, as well as the difference between the angles produced by the analytical solution and the model predictions, we see the following:

![LNN Error - Distance Between Points](plots/LNN_error_distance.png) ![ESN Error - Distance Between Points](plots/ESN_error_distance.png)

![LNN Error - Difference Between Angles](plots/LNN_theta_error.png) ![ESN Error - Difference Between Angles](plots/ESN_theta_error.png)

Visualizations of error report large differences between analytical solutions and model predictions for both models. Since the double pendulum is a highly chaotic system, any small error in the system will propogate over time, rendering all later predictions to be somewhat different from the analytical solution.

**LSTM Model**

We note that the LSTM is not really up to the task. With a 3 layer LSTM model with decreasing hidden layer sizes `[32, 16, 8]`, after training 10 epochs on 100 generated analytical samples, an example trial of generation looks like this:

![Model Output](plots/LSTM_pendulum_positions_model.png)![Ground Truth](plots/LSTM_pendulum_positions_analytic.png)

We observe that, although the model learns the spirit of a "chaotic system", it does not really learn the pattern. The outputs are far off from target, and des not even show similar pattern.

Also, the training statistics shows severe overfitting of the data:

![Training and Validating Loss Comparison](plots/LSTM_train_val.png)

We see that over time, the training loss decreases as we expected, but when we validated the model using other simulated data, it does not show similar decrease, hence we can conclude that overfitting happened.

Altering the `learning_rate` does not seem to solve the problem.

`TODO in final version`: add comparison from LSTM.

**LNN Model**

The LNN model demonstrates an accurate prediction of the pendulum for the first few prediction frame, but once there is considerable error, the error propagates quickly throughout the system and make the prediction unreliable. For single frame predictions however, we still witness reasonable loss decrease over the course of training:

![LNN Loss](plots/LNN_loss.png)

However, this error during multi-timestep forecasting makes sense since double pendulum is such a chaotic system and error propagates easily. Also, LNN prediction seems to follow the actual physical rules and conserve energy and momentum while making the prediction, which is a virtue most neural networks lack.

**ESN Model**

Our ESN model takes as input the triangular functions of the angles formed by the arms and the vertical line. The key hyperparameters for our network are as follows:

```python
leak_rate = 0.1         #Decides the "memory size". higher value -> shorter memory
spectral_radius = 25.0  #Higher values -> chaotic system
input_scaling = 0.5     #Smaller (-> 0) -> free behavior, larger (-> 1) -> input-driven
regularization = 1e-7   #ridge optimization parameter.
forecase = 1            #use the next following frame as label.
```

The ESN trains on entire time series and use the same serie (but one frame later) as label. The network turns out to be extremely inaccurate in learning and predicting the movement of the end joint of the double pendulum.

When we train the model, we noticed that there are 3 sample sequences that induced extremely high MSE error. To protect the network from such "pollution", we had to remove the 3 sequences by hand.

First of all, the prediction accuracy does not depend on the size of the training data. We first trained the ESN on each sample sequence, tested the prediction error (MSE), and then reset the model to untrained state. We then trained another model on the entire dataset (40 sequence) without reset. Figure 1(a) shows the MSE with/without resetting. Figure 1(b) compares the errors from the two models.

![Figure 1](plots/train_size_comp.png)

We see from the figures that the MSE loss are relatively similar for both models, which implies that size of training data does affect model precision.

This could be accounted to the limited "memory" for an ESN network. When training on large dataset, new incoming data takes away memory space of the network and make it "forgets" earlier inputs that it has learned.

We observe that this is not due to the specific sequence we tested the ESN upon. The sequences varies in their difficulty to train, but the the difference is within range of the error predicting one sequence can produce. This is shown in figure *2*

![Figure 2](plots/MSE_from_each_sequence.png)

Changing the `leak_rate` parameter for a longer term of memory will alter the effect of training size. We trained 2 new models with `leak_rate=0.05` and `leak_rate=0.01`, and compared it with the previous `leak_rate=0.1` model interms of difference of MSE with respect of resetting, using the following formula:

$Diff = MSE_{No\ reset}-MSE_{Reset}$

The results is shown below:
![Figure 3](plots/LR_comp.png)

We see that extremrely small `leak_rate` induce significantly larger MSE difference both above and beyond 0, which means it is highly unstable. Comparing `0.05` and `0,1`, we find that lowering `leak_rate` results in difference mostly below 0, which means in that case larger training size generally gives smaller error. However, It is also most unstable since there are cases where larger training dataset induced significantly high error.

Another way we can see the ineffectiveness of larger training data is that, in the no resetting case, our $x$ axis can be treated as time passed, and we see that the MSE does not shown any decrease pattern (see fig 1a).

The discussion above shows how the ESN model is not making valid predictions. Our main assumption is that the motion of the double pendulumn is overall circular. When more data are fed into the network, it learns nothing else but the circular motion. When less data are fed, either there are not enough data to learn from, or that it can only learn the local movement (which is often quite linear). The latter case also explaines why sometimes the prediction gives a linear trajectory that goes way beyond reasonable value -- It could have learnt a linear trend.

The thing about ESN and double pendulum is that, since double pendulum is a second order system, the error propagates throughout the system quickly, and combined with a flawed dataset, predicting become extremely difficult.

The discussion section will be presenting the plots made from the LSTM and LNN models, and provide the corresponding loss.

The interpretation goes in the paragraphs above.

This result proves our assumption that all 3 of RNN (LSTM), ESN and LNN are not very accurate at predicting double pendulum, but overall ESN< RNN < LNN . This also proves our assumption that LNN makes a physically credible prediction and is able to learn the actual physical rules of the system.

***

### Ethical Implication and Discussion

A possible ethical issue in this instance is the use of overideal training data. The IBM double pendulum dataset that is used for training is a dataset that is clean and not noisy: it is filmed with a high speed camera in a controlled environment, with carefully measured axis markers and angular values. However, it is possible that the models we train with this dataset are incapable of handling a noisy system, such as validation on a user-generated double pendulum path. We will analyze this issue in our validation of each model with the Pomona College double pendulum. 
 
This ethical implication has far reaching issues in multiple areas. If a trained model is only able to operate in the space of clean data, then in certain edge-cases or uncommon cases, the model will experience a high error rate. This is especially alarming in areas such as facial or speech recognition, where there may be high amounts of variation in the noisiness of images or audio, and error may result in a range of consequences from inconvenience to life-changing.

***

### Reflection

What would you do differently next time?

- Reach out to authors and researchers much earlier in the process and ask for help. It turned out that they were pretty responsive and excited that there are people working on extensions to their projects. If we did it earlier, we think our jobs would have been much easier. We also should have started training our networks earlier so we would have more time to make adjustments and tune our hyperparameters more.
- Do more readings earlier on.
- Spend more time training each network.
- Ask more questions.
- Do more research on how to compare different kinds of Neural Networks and how to make sure we're not putting in more time or effort into one or the other. Example: theoretically LNNs are supposed to perform the best. But did it perform the best in our research because of reseacher bias or because it actually did?

How would you continue this work (e.g., what extensions would you pursue)?

- Hamiltonian network.
- Work on all three kinds of datasets more completely.
- Use more statistical metrics for network comparison.

***

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

***

### Update 2

#### Progress

Thus far, we have explored our double pendulum dataset and have looked through notebooks about the general status of the dataset as well as training examples. We currently have a very rudimentary Long Short Term Memory neural network that is trained on the data, and it is capable of making predictions on a double pendulums path, though it is quite inaccurate. This LSTM is mostly adopted from the IBM example notebook, and we will soon be altering it to be our baseline RNN model and also performing hyperparameter optimization on that RNN model to make sure it performs to the best of its ability.

Another item we have updated is an improved ethics question - considering whether the clean IBM dataset we use for training causes us problems when noisy data is used as input. To test this, we have obtained a less optimal double pendulum from the Pomona College Physics Department and we plan on writing computer vision code to track and map the path of the pendulum so that we can validate it with our model.

We have also began working on hyperparameter optimization for the baseline networks.

#### Issues
We initially encountered some issues with the example notebooks provided by the IBM dataset and what exact parameters we would train on. However, we resolved those issues by studying the notebook more and understanding the coordinate axis on which we train on better. Aside from that, our progress has been fairly smooth and we understand our next steps well.

***

### Literature Review

Asseman, A., Kornuta, T., & Ozcan, A.S. (2018). Learning beyond simulated physics.

- This paper introduces a dataset for a double pendulum that consists of inputs of the positions and angles of the arms, and outputs of the path. It also utilizes an LSTM to train on this dataset and predict the motion of a double pendulum.

Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020, July 30). [*Lagrangian neural networks*](https://arxiv.org/abs/2003.04630).

- Compared to regular NN, HNN are better in learning symmetries and conservation law, but requires coordinates of system to be canoncial. LNN does not require this. Theories about lagrangian is included and 3 different experiments are performed, including double pendulum. Codes are provided [here](https://github.com/MilesCranmer/lagrangian_nns). It also has a really helpful reference list.

Bollt, E. (2021, January 4). [*On explaining the surprising success of reservoir computing forecaster of chaos? The universal machine learning dynamical system with contrast to VAR and DMD*](https://aip.scitation.org/doi/abs/10.1063/5.0024890). American Association of Physics Teachers.

- An explanation on why reservoir computing succeeds at forecasting dynamical systems. Explains some of the foundational mathematics behind reservoir computing and benchmarks various iterations of reservoir computing on tasks involving dynamical systems and compares them against other architectures.

Klinkachorn, S., & Parmar, J. (2019). Evaluating Current Machine Learning Techniques On Predicting Chaotic Systems CS.

- A study on the ability of different forms of ML and deep learning algorithms to fit the path of a double pendulum. The researchers found that at small angles where chaotic motion was not present,  a simple linear regression with a polynomial feature map performed best, while a LSTM was the most accurate when chaotic motion began to occur.

Rudy, S.H., Kutz, J.N., & Brunton, S.L. (2019). Deep learning of dynamics and signal-noise decomposition with time-stepping constraints. J. Comput. Phys., 396, 483-506.

- A study on the variance of the performance of deep learning models on complex and dynamical systems when there are variations in noise, as well as a new method to circumvent issues caused by noisy data. They treat measurement error and noisiness as part of the unknowns that the neural network must deal with, instead of de-noising data early on. The double pendulum is used as an example in this paper.

Shinbrot, T., Grebogi, C., Wisdom, J., & Yorke, J. A. (1992, June 1). [*Chaos in a double pendulum*](https://aapt.scitation.org/doi/10.1119/1.16860). American Association of Physics Teachers.

- This is a very basic introduction on double pendulums and some math included. Explains some of the fundamental mechanics behind the double pendulum and derives the the governing equation of motion for the problem with the Lagrangian formalism. Covers the exponentiation and mathematics behind why the system varies so drastically in response to changes in initial conditions.

Woolley, Jonathan W., P. K. Agarwal, and John Baker. [*Modeling and prediction of chaotic systems with artificial neural networks*](https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.2117). International journal for numerical methods in fluids 63.8 (2010): 989-1004.

- A paper about chaotic systems in general (not specific to double pendulum or phenomena in physics). Chaotic systems such as earthquakes, laser systems, epileptic seizures, combustion, and weather patterns are very difficult to predict. This study attempts to develop a system for training artificial neural networks t predict the future data of processes. Data set was obtained by solving Lorenz's equations. Backpropagation algorithm is used to train the network. "A correlation of 94% and a negative Lyapunov exponent indicate that the results obtained from ANN are in good agreement with the actual values."

Zhang, H., Fan, H., Wang, L., & Wang, X. (2021). [*Learning Hamiltonian dynamics with reservoir computing*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.024205). Physical Review E, 104(2), 024205.

- RC is good for state evolution prediction in Hamiltonian dynamics. They used parameter-aware RC to reconstruct the KAM diagram (something previously done with HNNs). RC is used in learning the behavior of a double pendulum. Compared to HNN, whose output depends solely on input at the current time, RC also takes into account the past states of the system. However, RC makes training simpler, since the Hamiltonian mechanisms are no longer pre-requisites.

Lutter, M., Ritter, C., &; Peters, J. (2019, July 10). [Deep lagrangian networks: Using physics as model prior for deep learning](https://arxiv.org/abs/1907.04490). 

- A network structure called Deep Lagrangian Networks is presented. Previous works seldomly combined NN and differential equations. The paper gives an introduction on Lagrangian mechanics and the math of fitting it into NN. The team did a 2-degree-of-freedom robot arm simulation, and the Deep Lagrangian Network learnt the physical model of the system. The double pendulum is a similar 2-degree-of-freedom problem that is based within classical mechanics.


### Useful Hamilton Neural Networks Papers

Greydanus, S., Dzamba, M., &amp; Yosinski, J. (2019, September 5). [Hamiltonian neural networks.](https://arxiv.org/abs/1906.01563.)

github pages/codes
- https://greydanus.github.io/2019/05/15/hamiltonian-nns/ (pages)
- https://github.com/greydanus/hamiltonian-nn (code)
- https://github.com/MilesCranmer/lagrangian_nns (code)

### Useful Lagrangian Neural Networks Papers

Lutter, M., Ritter, C., &amp; Peters, J. (2019, July 10). [Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning](https://arxiv.org/pdf/1907.04490.pdf)

Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). [Lagrangian neural networks](https://arxiv.org/pdf/2003.04630.pdf)

- github pages/codes
- https://greydanus.github.io/2020/03/10/lagrangian-nns/ (pages)
- https://github.com/MilesCranmer/lagrangian_nns (code)
