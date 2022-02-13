from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Input, Concatenate
from keras.optimizers import Adam
from ddpg import DDPGAgent
from rl.memory import SequentialMemory
import numpy as np
import gym


# Env initialization.
env = gym.make("gym_mastermind:Mastermind-v0")
np.random.seed(1235)
env.seed(1235)
NB_ACTIONS = env.action_space.shape[0]

# Hyperparameters settings.
WINDOW_LENGTH = 7
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 500000
LEARNING_RATE = 0.00003
NB_STEPS_WARMUP = 30000
NB_STEPS = 9000000

actor = Sequential()
actor.add(Flatten(input_shape=(WINDOW_LENGTH,) + (env.observation_space.shape)))
actor.add(Dense(2048))
actor.add(Activation("relu"))
actor.add(Dense(512))
actor.add(Activation("relu"))
actor.add(Dense(128))
actor.add(Activation("relu"))
actor.add(Dense(NB_ACTIONS))
actor.add(Activation("sigmoid"))
print(actor.summary())

action_input = Input(shape=(NB_ACTIONS,), name="action_input")
observation_input = Input(
    shape=(WINDOW_LENGTH,) + (env.observation_space.shape), name="observation_input"
)
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(2048)(x)
x = Activation("relu")(x)
x = Dense(512)(x)
x = Activation("relu")(x)
x = Dense(128)(x)
x = Activation("relu")(x)
x = Dense(1)(x)
x = Activation("linear")(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

memory = SequentialMemory(limit=MEMORY_SIZE, window_length=WINDOW_LENGTH)
agentddpg = DDPGAgent(
    nb_actions=NB_ACTIONS,
    actor=actor,
    critic=critic,
    critic_action_input=action_input,
    memory=memory,
    nb_steps_warmup_critic=NB_STEPS_WARMUP,
    nb_steps_warmup_actor=NB_STEPS_WARMUP,
    gamma=GAMMA,
    batch_size=BATCH_SIZE,
    target_model_update=1e-3,
)

agentddpg.compile(Adam(lr=LEARNING_RATE), metrics=["mae"])
# agentddpg.load_weights("model40.hd5f")
agentddpg.test(env, nb_episodes=15000, visualize=False, verbose=2)
agentddpg.fit(env, verbose=2, nb_steps=NB_STEPS)
# agentddpg.save_weights("model43.h5")
