# Deep-Q-Learning-Network-from-Scratch-in-Python-TensorFlow-and-OpenAI-Gym

These code files implement the Deep Q-learning Network (DQN) algorithm from scratch by using Python, TensorFlow (Keras), and OpenAI Gym. The codes are tested in the OpenAI Gym Cart Pole (v1) environment. These code files are a part of the reinforcement learning tutorial I am developing. The tutorial webpage explaining the codes is given here: 

https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/

Uploaded files:

- "driverCode_final.py" - this is the driver code for training the model. This code import a class definition that implements the Deep Q Network from "functions_final.py". You should start from here.

- "functions_final.py" - this is the file that implements the class called "DeepQLearning" that implements the Deep Q Network.

- "simulateTrainedModel.py" - this file loads the trained TensorFlow model stored in the TensorFlow model file "trained_model.h5"  and creates a video that shows the control performance. Note that the video is saved in the folder "stored_video" (if such a folder does not exist it will be created)

- "trained_model.h5"  - this is the model I trained for one day on my computer - you can use this model to visualize the performance. You can also improve the model by loading its weights and resuming the training process. DO NOT OVERWRITE THIS MODEL. IF YOU DO SO, YOU WILL ERASE THE TRAINED MODEL.

- "trained_model_temp.h5" - this is just a temporary model obtained after several training episodes to illustrate the performance of untrained model, and to use it as a baseline


LICENSE: THIS CODE CAN BE USED FREE OF CHARGE ONLY FOR ACADEMIC AND EDUCATIONAL PURPOSES. THAT IS, IT CAN BE USED FREE OF CHARGE ONLY IF THE PURPOSE IS NON-COMMERCIAL AND IF THE PURPOSE IS NOT TO MAKE PROFIT OR EARN MONEY BY USING THIS CODE.

IF YOU WANT TO USE THIS CODE IN THE COMMERCIAL SETTING, THAT IS, IF YOU WORK FOR A COMPANY OR IF YOU ARE AN INDEPENDENT
CONSULTANT AND IF YOU WANT TO USE THIS CODE, THEN WITHOUT MY PERMISSION AND WITHOUT PAYING THE PROPER FEE, YOU ARE NOT ALLOWED TO USE THIS CODE. YOU CAN CONTACT ME AT

aleksandar.haber@gmail.com

TO INFORM YOURSELF ABOUT THE LICENSE OPTIONS AND FEES FOR USING THIS CODE.
ALSO, IT IS NOT ALLOWED TO 
(1) MODIFY THIS CODE IN ANY WAY WITHOUT MY PERMISSION.
(2) INTEGRATE THIS CODE IN OTHER PROJECTS WITHOUT MY PERMISSION.

 DELIBERATE OR INDELIBERATE VIOLATIONS OF THIS LICENSE WILL INDUCE LEGAL ACTIONS AND LAWSUITS. 



