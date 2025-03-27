# Deep-Q-Learning-Network-from-Scratch-in-Python-TensorFlow-and-OpenAI-Gym

IMPORTANT NOTE: The code files are released under "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International Public License (CC BY-NC-ND 4.0.)"  

**Brief explanation of the license (read the complete license):** 
**Attribution (BY):** You must give appropriate credit and reference to the creator and code (citation). You need to provide a link to the license and link to the code files. 
**NonCommercial (NC):** You may not use the material for commercial purposes. 
**NoDerivatives (ND):** You cannot remix, transform, or build upon the material, meaning you can only share the original work without any adaptations. If you plan to use the code for commercial purposes, contact the author at ml.mecheng@gmail.com

These code files implement the Deep Q-learning Network (DQN) algorithm from scratch by using Python, TensorFlow (Keras), and OpenAI Gym. The codes are tested in the OpenAI Gym Cart Pole (v1) environment. These code files are a part of the reinforcement learning tutorial I am developing. The tutorial webpage explaining the codes is given here: 

https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/

Uploaded files:

- "driverCode_final.py" - this is the driver code for training the model. This code import a class definition that implements the Deep Q Network from "functions_final.py". You should start from here.

- "functions_final.py" - this is the file that implements the class called "DeepQLearning" that implements the Deep Q Network.

- "simulateTrainedModel.py" - this file loads the trained TensorFlow model stored in the TensorFlow model file "trained_model.h5"  and creates a video that shows the control performance. Note that the video is saved in the folder "stored_video" (if such a folder does not exist it will be created)

- "trained_model.h5"  - this is the model I trained for one day on my computer - you can use this model to visualize the performance. You can also improve the model by loading its weights and resuming the training process. DO NOT OVERWRITE THIS MODEL. IF YOU DO SO, YOU WILL ERASE THE TRAINED MODEL.

- "trained_model_temp.h5" - this is just a temporary model obtained after several training episodes to illustrate the performance of untrained model, and to use it as a baseline



