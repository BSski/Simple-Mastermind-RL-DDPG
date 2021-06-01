"""Mastermind"""
from collections import Counter
import logging
import random
import copy
import os
import itertools

import numpy as np
import gym
from gym.utils import seeding
from gym import spaces
from matplotlib import pyplot as plt
logger = logging.getLogger(__name__)

class MastermindEnv(gym.Env):
    """
    Guess a 'self.SIZE'-digits long password where each digit is between 0 and 'self.VALUES'.

    After each step the agent is provided with a 'self.SIZE'-digits long tuple:
    - '2' indicates that a digit has been correclty guessed at the correct position.
    - '1' indicates that a digit has been correclty guessed but the position is wrong.
    - '0' otherwise.

    The rewards at the end of the episode are:
    -20 if the agent's guess is incorrect
    75 if the agent's guess is correct
    + rewards for certain actions, such as next guess penalty and repeating action penalty

    The episode terminates after the agent guesses the target or
    'self.GUESS_MAX' steps have been taken.
    """

    """ # TODO:
    - dodaj obliczany próg random nagrody: 1/ilość dostępnych kombinacji * nagroda max, ale z ujemną
    wartością to chyba będzie inaczej wyglądać, przemyśl
    - zwiększaj negatywną nagrodę za każdy kolejny guess:
    1:1 2:2 3:3 4:4 5:5 6:6 7:7
    """

    VALUES = 4
    SIZE = 4  # 3 or 4
    GUESS_MAX = 8
    def __init__(self):
        self.target = None
        self.guess_count = None
        self.observation = None
        self.recent_rewards_history = []
        self.recent_mean_rewards_history = []
        self.amount_of_recent_rewards_in_history = 3000
        for i in range(self.amount_of_recent_rewards_in_history):
            self.recent_mean_rewards_history.append(0)
        self.guesses_list = []
        self.actions_tried_this_epizode = []

        self.feedback_pegs_to_binary_dict = self.generate_feedback_pegs_to_binary_dict()

        self.values_range = ""
        for i in range(self.VALUES):
            self.values_range += str(i)
        self.possible_codes = ["".join(item) for item in itertools.product(self.values_range, repeat=self.SIZE)]


        self.number_of_digits_in_action = len(self.decimalToBinary(len(self.possible_codes)-1))
        self.possible_codes_dict = self.generate_possible_codes_dict(self.possible_codes)
        self.possible_feedback_pegs = ["".join(item) for item in set(itertools.combinations(str(
                                       "2"*self.SIZE + "1"*self.SIZE + "0"*self.SIZE), self.SIZE))]
        self.amount_of_possible_feedback_pegs = len(self.possible_feedback_pegs)-1
        self.amount_of_possible_binary_feedback_pegs = len(self.decimalToBinary(self.amount_of_possible_feedback_pegs))


        self.obs_low = []
        self.obs_low.append(0)  # current step
        for slot in range(self.number_of_digits_in_action):
            self.obs_low.append(0)
        for peg in range(self.amount_of_possible_binary_feedback_pegs):
            self.obs_low.append(0)
        #print(self.obs_low)
        self.obs_high = []
        self.obs_high.append(self.GUESS_MAX)  # current step
        for slot in range(self.number_of_digits_in_action):
            self.obs_high.append(1)
        for peg in range(self.amount_of_possible_binary_feedback_pegs):
            self.obs_high.append(1)
        #print(self.obs_high)
        self.observation_space = spaces.Box(low=np.array(self.obs_low),
                                            high=np.array(self.obs_high), dtype=np.int)

        self.action_low = []
        for slot in range(self.number_of_digits_in_action):
            self.action_low.append(0)
        #print("action_low;", self.action_low)
        self.action_high = []
        for slot in range(self.number_of_digits_in_action):
                self.action_high.append(1)
        #print("action_high;", self.action_high)
        self.action_space = spaces.Box(low=np.array(self.action_low), high=np.array(self.action_high), dtype=np.int)
        self.seed()

        self.testing = 0
        self.nr_of_test_epizode = 0
        self.test_log = {}
        self.test_guesses_list = []

        self.reset()
        self.counter = -1
        self.interval = 0
        self.actions_list = []


    def generate_code(self):
        if self.nr_of_test_epizode == len(self.possible_codes):
            self.testing = 0
            print("\n\n\n\n\n\n\n\n\n\n")
            for i in self.test_log:
                print(i, ":", self.test_log[i])
            print("\n\n\n\n\n\n\n\n\n\n")

        if self.testing == 1:
            counter = 0
            for i in range(200000):
                counter += 3
                counter -= 3
            counter = 0

            code = []
            for i in self.possible_codes[self.nr_of_test_epizode]:
                if i != " ":
                    code.append(int(i))
            self.nr_of_test_epizode  += 1
            return code
        else:
            return [random.randint(0, self.VALUES-1) for _ in range(self.SIZE)]

    def generate_possible_codes_dict(self, possible_codes):
        # WORKS ONLY FOR BINARY CODES  # TODO EXPAND IT TO MORE VALUES
        #possible_codes_original = copy.deepcopy(possible_codes)

        # add spaces between characters
        for i in range(len(possible_codes)):
            nr = 0
            for j in range(len(possible_codes[i])-1):
                nr += 1
                possible_codes[i] = possible_codes[i][:j+nr] + " " + possible_codes[i][j+nr:]

        # convert string to list of ints
        temp_characters_list = []
        possible_codes_separate = []
        for i in possible_codes:
            temp_characters_list = i.split()
            temp_characters_list_int = map(int, temp_characters_list)
            temp_characters_list_int_list = list(temp_characters_list_int)
            possible_codes_separate.append(temp_characters_list_int_list)

        # create list of binary indexes in amount of len(possible_codes)
        temp_binary_list = [0 for i in range(self.number_of_digits_in_action)]
        binary_indexes_for_actions = []
        for code in range(len(possible_codes)):
            binary_code = ""
            for digit in temp_binary_list:
                binary_code += str(digit)
            binary_indexes_for_actions.append(binary_code)
            for i in range(len(temp_binary_list)):
                if i == 0:
                    temp_binary_list[len(temp_binary_list)-1] += 1
                else:
                    if temp_binary_list[len(temp_binary_list)-i] > 1:
                        try:
                            temp_binary_list[len(temp_binary_list)-i-1] += 1
                            temp_binary_list[len(temp_binary_list)-i] = 0
                        except "TryingToIncrementateNonExistantIndex":
                            print("\n\n\n\n\n\n\n\n\n\n\n\n\n SOMETHING WENT WRONG in generate_possible_codes_dict \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

        # create dict
        possible_codes_dict = {}
        zip_iterator = zip(binary_indexes_for_actions, possible_codes_separate)
        possible_codes_dict = dict(zip_iterator)
        return possible_codes_dict

    def generate_feedback_pegs_to_binary_dict(self):
        bag_of_peg_combinations = ["".join(item) for item in set(itertools.combinations(str(
                                   "2"*self.SIZE + "1"*self.SIZE + "0"*self.SIZE),
                                   self.SIZE))]
        pegs_to_binary_dict = {}
        a = 0
        b = 0
        c = 0
        d = 0
        for i in bag_of_peg_combinations:
            pegs_to_binary_dict[i] = str(d) + str(c) + str(b) + str(a)
            a += 1
            if a > 1:
                a = 0
                b += 1
            if b > 1:
                b = 0
                c += 1
            if c > 1:
                c = 0
                d += 1
            assert d < 2, "Error: feedback pegs combinations space is too big for only 4 binary digits."

        possible_peg_combinations = ["".join(item) for item in set(itertools.product("012", repeat=self.SIZE))]
        new_pegs_to_binary_dict = {}
        for i in possible_peg_combinations:
            for j in pegs_to_binary_dict:
                if Counter(j) == Counter(i):
                    new_pegs_to_binary_dict[i] = pegs_to_binary_dict[j]

        # Merging dicts
        # TODO: yyyy chyba starcza sam 'new_pegs_to_binary_dict', po co jest reszta skoro on zawiera w sobie pegs_to_binary_dict? czytaj \/
        # zahashowałem ten zmergowany dict, skoro to ten sam /\
        # ready_feedback_pegs_combinations_dict = {**new_pegs_to_binary_dict, **pegs_to_binary_dict}
        print(pegs_to_binary_dict)
        print("\n\n\n", new_pegs_to_binary_dict)
        # print("\n\n\n", ready_feedback_pegs_combinations_dict)
        # return ready_feedback_pegs_combinations_dict
        return new_pegs_to_binary_dict

    def encode_feedback_pegs_as_binary(self, feedback_pegs):
        feedback_pegs_str = ""
        for i in feedback_pegs:
            feedback_pegs_str += str(i)
        return self.feedback_pegs_to_binary_dict[str(feedback_pegs_str)]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calculate_feedback_pegs(self, action):
        match_idxs = set(idx for idx, ai in enumerate(action) if ai == self.target[idx])
        n_correct = len(match_idxs)
        g_counter = Counter(self.target[idx] for idx in range(self.SIZE) if idx not in match_idxs)
        a_counter = Counter(action[idx] for idx in range(self.SIZE) if idx not in match_idxs)
        n_white = sum(min(g_count, a_counter[k]) for k, g_count in g_counter.items())
        return tuple([0] * (self.SIZE - n_correct - n_white) + [1] * n_white + [2] * n_correct)

    def insert_new_step_to_the_observation(self, observation, feedback_pegs_binary, action_discretized):
        updated_observation = observation
        # dodawanie numeru aktualnego kroku do obserwacji:
        updated_observation[0] = self.guess_count
        # dodawanie akcji do obserwacji:
        for i in range(len(action_discretized)):
            updated_observation[i+1] = action_discretized[i]
        # dodawanie feedback pegów do obserwacji:
        for i in range(len(feedback_pegs_binary)):
            updated_observation[len(action_discretized)+i+1] = int(feedback_pegs_binary[i])
        return updated_observation

    def get_observation(self, action, action_discretized):
        feedback_pegs = self.calculate_feedback_pegs(action)
        feedback_pegs_binary = self.encode_feedback_pegs_as_binary(feedback_pegs)
        ## #print("---   BINARY PEGS:", feedback_pegs_binary, "  PEGS:", feedback_pegs)

        #self.observation = self.shift_observation_to_the_right(self.observation)
        self.observation  = self.insert_new_step_to_the_observation(self.observation,
                                                                    feedback_pegs_binary,
                                                                    action_discretized)
        ## #print("\n-   OBSERVATION:", self.observation)
        return self.observation

    def decimalToBinary(self, n):
        return bin(n).replace("0b", "")

    def binaryToDecimal(self, n):
        return int(n, 2)

    def discretize_action(self, action):
        #print("---   ACTION TO DISCRETIZE:", action)
        action_discretized = []
        for i in list(action):
            assert i >= 0 and i <= 1, "value outside of sigmoid range"
            if i < 0.5:
                action_discretized.append(0)
            else:
                action_discretized.append(1)
        return action_discretized

    def step(self, action):
        reward = 0
        self.counter += 1
        self.guess_count += 1
        action = list(action)
        #print("---------   GUESS " + str(self.guess_count) + ":")
        #print("-------   original ddpg action:", [round(i, 2) for i in action])

        action_discretized = self.discretize_action(action)

        action_discretized_str = ""
        for i in action_discretized:
            action_discretized_str += str(i)
        action = action_discretized_str
        #print("-----   BINARY ACTION: " + '\'' + str(action) + '\'')
        action = self.possible_codes_dict[action]
        ## #print("----   ACTION:", action)

        # check if the agent guessed at first try; if yes, change the code
        if self.testing == 0:
            if self.guess_count == 1 and action == self.target:
                new_target = self.generate_code()
                while self.target == new_target:
                    new_target = self.generate_code()
                self.target = new_target
                ## #print("############# ZMIANA TARGETU NA:", self.target)

        # check if agent guessed the code
        done = action == self.target or self.guess_count >= self.GUESS_MAX
        if done:
            if action == self.target:
                reward = 100
                ## #print("\n\nZgadnięte!\n\n\n")
            else:
                reward = -12
        else:
            reward = -1

        self.observation = self.get_observation(action, action_discretized)
        if action in self.actions_tried_this_epizode:
            reward += -5
        self.actions_tried_this_epizode.append(action)

        self.push_reward_to_list(reward)
        if os.path.exists("C:\\Users\Bartek\\Desktop\\gyms\\gym-mastermind\\gym_mastermind\\envs\\generate_graph_Y.txt"):
            self.draw_rewards_history()

        if self.testing == 1:
            if not done:
                self.test_guesses_list.append(action)
            if done:
                self.test_guesses_list.append(action)
                self.test_log[str(self.target)] = self.test_guesses_list
                self.test_guesses_list = []

        return self.observation, reward, done, {}

    def reset(self):
        if os.path.exists("C:\\Users\Bartek\\Desktop\\gyms\\gym-mastermind\\gym_mastermind\\envs\\generate_txt.txt"):
            #print("podaj po kolei wszystkie kombinacje i wyrzuć je jako output do pliku")
            #print("w sumie to weź je streść elegancko do słownika i na jego podstawie licz %")
            self.testing = 1

        self.guesses_list.append(self.guess_count)
        self.target = self.generate_code()
        ## #logger.debug("TARGET: %s", self.target)
        self.guess_count = 0
        self.interval = 0
        # Creating empty observation
        self.observation = []
        self.observation.append(0)  # current step
        self.actions_tried_this_epizode = []

        # dodawanie miejsc na akcję w obserwacji:
        for slot in range(self.number_of_digits_in_action):
            self.observation.append(0)

        for peg in range(self.amount_of_possible_binary_feedback_pegs):
            self.observation.append(0)
        return self.observation

    def push_reward_to_list(self, reward):
        if len(self.recent_rewards_history) > self.amount_of_recent_rewards_in_history:
            self.recent_rewards_history.pop(0)
        self.recent_rewards_history.append(reward*self.guess_count)
        a = np.array(self.recent_rewards_history)
        self.recent_mean_rewards_history.append(np.mean(a))

    def draw_rewards_history(self):
        plt.plot(self.recent_mean_rewards_history[::300])
        plt.axline((0, 25), (600, 25), alpha = 0.6)
        plt.ylim(-50, 110)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    mm = MastermindEnv()
