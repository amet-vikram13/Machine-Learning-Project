import gym
import numpy as np
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.layers import Dense,Dropout,Input

# modifying the default parameters of np.load because numpy is depcrepted
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

env = gym.make("CartPole-v1")
max_episode_steps = 500
initial_episodes = 250000
score_requirement = 60

def some_random_episodes() :
	for _ in range(5) :
		env.reset()
		for _ in range(max_episode_steps) :
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done :
				break

#some_random_episodes()
#print("All Done!")

def generate_training_data() :
	training_data = []
	scores = []
	accepted_scores = []
	for episode in range(initial_episodes) :
		score = 0
		game_memory = []
		prev_observation = []
		env.reset()
		for _ in range(max_episode_steps) :
			action = np.random.randint(0,2)
			observation, reward, done, info = env.step(action)
			if len(prev_observation)>0 :
				game_memory.append([prev_observation,action])
			prev_observation = observation
			score += reward
			if done :
				break
		#print("score: ",score)
		if score >= score_requirement :
			accepted_scores.append(score)
			for data in game_memory :
				if  data[1]==1 :
					output = [0,1]
				elif data[1]==0 :
					output = [1,0]
				training_data.append([data[0],output])
		scores.append(score)
		print("Episode {} done!".format(episode))
	
	training_data_save = np.array(training_data)
	np.save('task3_training_data/training_data.npy',training_data_save)
	
	if len(accepted_scores)>0 :
		print("Average accepted score: ",np.mean(accepted_scores))
		print("Median score for accepted scores: ",np.median(accepted_scores))

	return training_data

#training_data = generate_training_data()
#print("length of training data: ",len(training_data))
#print("All Done!")

#training_data = np.load("task3_training_data/training_data_8827.npy")

def neural_network_model(input_shape) :
	input_layer  = Input(shape=input_shape)
	hidden_layer = Dense(200,activation="tanh")(input_layer)
	hidden_layer = Dropout(0.2)(hidden_layer)
	hidden_layer = Dense(300,activation="tanh")(hidden_layer)
	hidden_layer = Dropout(0.2)(hidden_layer)
	hidden_layer = Dense(200,activation="tanh")(hidden_layer)
	hidden_layer = Dropout(0.2)(hidden_layer)
	hidden_layer = Dense(300,activation="tanh")(hidden_layer)
	hidden_layer = Dropout(0.4)(hidden_layer)
	#hidden_layer = Dense(1024,activation="relu")(hidden_layer)
	#hidden_layer = Dense(256,activation="relu")(hidden_layer)
	#hidden_layer = Dense(128,activation="relu")(hidden_layer)
	#hidden_layer = Dense(64,activation="relu")(hidden_layer)
	#hidden_layer = Dense(32,activation="relu")(hidden_layer)
	output_layer = Dense(2,activation="softmax")(hidden_layer)
	model = Model(inputs = input_layer,outputs = output_layer)
	adam_optimizer = Adam(learning_rate=0.005)
	model.compile(loss="categorical_crossentropy",optimizer=adam_optimizer,metrics=["accuracy"])
	return model

def train_data(training_data,model=False):
	inputs =  np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
	outputs = np.array([i[1] for i in training_data]).reshape(-1,2)
	model = neural_network_model((inputs.shape[1],))
	print(model.summary())
	model.fit(x=inputs,y=outputs,batch_size=500,epochs=20,verbose=1)
	return model

#model = train_data(training_data)
#model.save("models/task3_model_2.h5")
#print("model trained!")

model = load_model("models/final_task3_model_2.h5")

def evaluate_model(model) :
	scores = []
	for episode in range(100) :
		score = 0
		prev_observation = []
		env.reset()
		for _ in range(500) :
			#env.render()
			if len(prev_observation)==0 :
				action = np.random.randint(0,2)
			else :
				action = np.argmax(model.predict(np.array(prev_observation).reshape(-1,len(prev_observation)),batch_size=1)[0])
			new_observation,reward,done,info = env.step(action)
			prev_observation = new_observation
			score += reward
			if done :
				break
		scores.append(score)
		print("Episode {} done!".format(episode))
	print("Length Scores: ",len(scores))
	print(scores)
	print("Average Score: ",np.mean(scores))

evaluate_model(model)