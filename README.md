## [](#header-2)Analysis of Proximal Policy Optimization algorithm Using OpenAI Gym

### [](#header-3) Goals:
*   Comparing the algorithm performance with other baseline techniques for OpenAI game environment
*   Exploring performance based on input data preprocessing , using different Neural Network architectures & CPU vs GPU training
*   Modifying different hyperparameters to analyze their impact on the overall performance of the algorithm


### [](#header-3) Implementation:
*   The model is developed using TensorFlow and input data is collected from OpenAI GYM's _MS-PACMAN_ environment.

*   Performance of different neural network architectures is explored:

!["CNN vs LSTM - Reward function"](https://github.com/jadhavhninad/Analysis-of-Proximal-Policy-Optimization-using-OpenAI-gym/blob/master/Plots/Reward_CNN_LSTM.png)


*   GPU based training was done using [Google Collaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)

*   Reference : [OpenAI GYM Baselines](https://github.com/openai/baselines)

### [](#header-3) Output:
*   Different models based on the modified hyperparemeters, CPU training & GPU training.
*   Performance comparison(rewards & loss function) plots.
