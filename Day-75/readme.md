## Monte Carlo (MC) Method <a name="MonteCarlo"></a>:
- Demo Code: [monte_carlo_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/monte_carlo_demo.ipynb)
- MC methods learn directly from episodes of experience.
- MC is model-free :  no knowledge of MDP transitions / rewards.
- MC uses the simplest possible idea: value = mean return.
- Episode must terminate before calculating return.
- Average return is calculated instead of using true return G.
- First Visit MC: The first time-step t that state s is visited in an episode.
- Every Visit MC: Every time-step t that state s is visited in an episode.


### MC Calculating Returns (with Pseudocode) <a name="MCCalculatingReturns"></a>:
![mc-calculating-returns](https://user-images.githubusercontent.com/10358317/49827998-cca62980-fd9b-11e8-999b-150aac525870.jpg)

### First-Visit MC (with Pseudocode) (MC Prediction Problem) <a name="FirstVisitMC"></a>:
![first-visit-mc](https://user-images.githubusercontent.com/10358317/49827884-73d69100-fd9b-11e8-9623-16890aa3bbcb.jpg)

### MC Exploring-Starts (with Pseudocode) (MC Control Problem) <a name="MCExploringStarts"></a>:
- Demo Code: [monte_carlo_es_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/monte_carlo_es_demo.ipynb)
- State s and Action a is randomly selected for all starting points.
- Use Q instead of V 
- Update the policy after every episode, keep updating the same Q in-place.

![mc-control1](https://user-images.githubusercontent.com/10358317/49828847-fbbd9a80-fd9d-11e8-9286-dee68c6fa1a2.jpg)

### MC Epsilon Greedy (without Exploring Starts) <a name="MCEpsilonGreedy"></a>:
- Demo Code: [monte_carlo_epsilon_greedy_demo.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/monte_carlo_epsilon_greedy_demo.ipynb)
- MC Exploring start is infeasible, because in real problems we can not calculate all edge cases (ex: in self-driving car problem, we can not calculate all cases).
- Randomly selection for all starting points in code is removed.
- Change policy to sometimes be random.
- This random policy is Epsilon-Greedy (like multi-armed bandit problem)

## Temporal Difference (TD) Learning Method <a name="TDLearning"></a>:
- Demo Code: [td0_prediction.ipynb](https://github.com/omerbsezer/rl-tutorial-with-demo/blob/master/td_prediction.ipynb)
- TD methods learn directly from episodes of experience.
- TD updates a guess towards a guess
- TD learns from incomplete episodes, by bootstrapping.
- TD uses bootstrapping like DP, TD learns experience like MC (combines MC and DP).

### MC - TD Difference <a name="MCTDDifference"></a>:
- MC and TD learn from experience.
- TD can learn before knowing the final outcome.
- TD can learn online after every step. MC must wait until end of episode before return is known.
- TD can learn without the final outcome.
- TD can learn from incomplete sequences. MC can only learn from complete sequences.
- TD works in continuing environments. MC only works for episodic environments.
- MC has high variance, zero bias. TD has low variance, some bias.

![mc-td-dif1](https://user-images.githubusercontent.com/10358317/49805899-60a9ce00-fd67-11e8-900e-38662cf36a54.jpg)
![mc-td-dif2](https://user-images.githubusercontent.com/10358317/49805902-61dafb00-fd67-11e8-8033-b06f8a3ed1c1.jpg)
![mc-td-dif3](https://user-images.githubusercontent.com/10358317/49810084-758b5f00-fd71-11e8-8b67-b1d8da52e45a.jpg)
[David Silver Lecture Notes]

### MC - TD - DP Difference in Visual <a name="MCTDDifferenceinVisual"></a>:
![mc-td-dp](https://user-images.githubusercontent.com/10358317/49806522-01e55400-fd69-11e8-92a6-9bff14bb4c80.jpg)
[David Silver Lecture Notes]