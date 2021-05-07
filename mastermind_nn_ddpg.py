from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Input, Concatenate
from keras.optimizers import Adam
from rl.agents.ddpg import DDPGAgent
from rl.memory import SequentialMemory
import numpy as np
import gym
import logging
logging.basicConfig(level=logging.DEBUG)

#############################################################################

# Env initialization.
env = gym.make('gym_mastermind:Mastermind-v0')
# env = gym.make('gym_tdh:Tdh-v0')
np.random.seed(1235)
env.seed(1235)
nb_actions = env.action_space.shape[0]

# Hyperparameters settings.
WINDOW_LENGTH = 6
GAMMA = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 80000
NB_STEPS_WARMUP = 10000

actor = Sequential()
actor.add(Flatten(input_shape=(WINDOW_LENGTH,) + (env.observation_space.shape)))
actor.add(Dense(108))
actor.add(Activation('relu'))
actor.add(Dense(48))
actor.add(Activation('relu'))
actor.add(Dense(12))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(WINDOW_LENGTH,) + (env.observation_space.shape), name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(108)(x)
x = Activation('relu')(x)
x = Dense(48)(x)
x = Activation('relu')(x)
x = Dense(12)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

memory = SequentialMemory(limit=MEMORY_SIZE, window_length=WINDOW_LENGTH)  # bylo 15 000
agentddpg = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=NB_STEPS_WARMUP, nb_steps_warmup_actor=NB_STEPS_WARMUP,
                  gamma=GAMMA, batch_size=BATCH_SIZE, target_model_update=1e-3)

agentddpg.compile(Adam(lr=0.001), metrics=['mae'])
agentddpg.fit(env, verbose=2, nb_steps = 10000000)
agentddpg.save_weights("model43.h5")



"""testy kazdej mozliwej kombinacji po kolei"""
