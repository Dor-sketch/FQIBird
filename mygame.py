"""
This is a custom Flappy Bird game class for the agent to interact with
"""

import asyncio
import random
from collections import deque
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tqdm import tqdm
import src.flappy
tqdm.disable = True

COLLISION = 0
CROSS = 1
NOR_COLLISION_CROSS = -1
JUMPS_PADDING = 11
TOP_PIPE_PADDING = 50
LOWER_PIPE_PADDING = 20
FLOOR = 411
BIG_NUMBER = 999999999.0
LESS_BIG_NUMBER = 10000000.0


def clamp(value, min_value, max_value):
    """
    Clamp a value between a minimum and maximum value
    :param value: the value to clamp
    :param min_value: the minimum value
    :param max_value: the maximum value
    :return: the clamped value
    """
    return max(min_value, min(value, max_value))


class QLearningAgent:
    """
    Q-learning agent
    """

    def __init__(self, actions: list = [0, 1], alpha: float = 0.1, gamma: float = 0.2, epsilon: float = 0.01):
        """
        Initialize the Q-learning agent
        :param actions: list of possible actions
        :param alpha: learning rate
        :param gamma: discount factor
        :param epsilon: exploration rate
        """
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def get_action_from_q(self, state: dict, game: src.flappy.Flappy):
        """
        Get the best action from the Q-table
        :param state: the current state of the game
        :return: the best action
        """
        state_key = str(state)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0 for action in self.actions}
        else:
            for a in self.actions:
                if a not in self.q_table[state_key]:
                    self.q_table[state_key][a] = 0
        # Predict future collision and penalize the action leading to it
        for a in self.actions:
            if self.predict_collision_or_cross(game, a) == CROSS:
                return a
        return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def get_action(self, state: dict, game: src.flappy.Flappy):
        """
        Get the best action for the current state based on reflexes
        :param state: the current state of the game
        :param game: the game object
        :return: the best action if the agent is using reflexes or the best action from the Q-table
        """
        upper_pipes = [pipe.get_state() for pipe in game.pipes.upper]
        lower_pipes = [pipe.get_state() for pipe in game.pipes.lower]
        upper_pipes = [pipe.get_state() for pipe in game.pipes.upper]
        lower_pipes = [pipe.get_state() for pipe in game.pipes.lower]
        player_pos = game.player.get_state()
        player_pos = game.player.get_state()

        # Find the closest pipe that the player hasn't passed yet
        closest_pipe_index = min(
            (i for i in range(len(upper_pipes)) if upper_pipes[i][0] > 50),
            default=0,
            key=lambda i: upper_pipes[i][0] - player_pos[0]
        )
        closest_upper_pipe = upper_pipes[closest_pipe_index]
        closest_lower_pipe = lower_pipes[closest_pipe_index]

        def between_pipes():
            return player_pos[0] + player_pos[1] > upper_pipe_bottom + JUMPS_PADDING and player_pos[0] < lower_pipe_top - TOP_PIPE_PADDING

        def below_pipes():
            return player_pos[0] + player_pos[1] > upper_pipe_bottom + JUMPS_PADDING and player_pos[1] > 0 and player_pos[0] + player_pos[1] < lower_pipe_top - LOWER_PIPE_PADDING

        def close_and_above():
            return closest_upper_pipe[0] - game.player.x > JUMPS_PADDING and player_pos[0] + player_pos[1] < upper_pipe_bottom

        def going_down_and_close():
            return player_pos[0] + player_pos[1] > lower_pipe_top + JUMPS_PADDING and player_pos[1] > 0

        def hitting_floor():
            return player_pos[0] + player_pos[1] > FLOOR

        upper_pipe_bottom = closest_upper_pipe[1] + closest_upper_pipe[3]
        lower_pipe_top = closest_lower_pipe[1]

        if between_pipes() or close_and_above():
            return 0
        elif below_pipes() or going_down_and_close() or hitting_floor():
            return 1

        return self.get_action_from_q(state, game)

    def learn(self, state, action: int, reward: float, new_state: dict, game: src.flappy.Flappy):
        """
        Learn from the game state
        :param state: the current state of the game
        :param action: the action taken
        :param reward: the reward received
        :param new_state: the new state of the game
        """
        if state is None:
            return
        state_key = str(state)
        new_state_key = str(new_state)
        if new_state_key not in self.q_table:
            self.q_table[new_state_key] = {a: 0 for a in self.actions}
        # If a future collision is predicted, increase the reward for avoiding it
        if self.predict_collision_or_cross(game, action) == COLLISION:
            reward -= LESS_BIG_NUMBER
        self.q_table[state_key][action] += \
            self.alpha * (reward + self.gamma
                          * max(self.q_table[new_state_key].values())
                          - self.q_table[state_key][action])

    def predict_collision_or_cross(self, flappy, action):
        # Create a copy of the game state
        player_x = flappy.player.x
        player_y = flappy.player.y
        player_w = flappy.player.w
        player_h = flappy.player.h
        player_vel_y = flappy.player.vel_y * action
        player_acc_y = flappy.player.acc_y
        player_rot = flappy.player.rot
        player_vel_rot = flappy.player.vel_rot

        pipes = []
        for pipe in flappy.pipes.upper:
            pipes.append(pipe.x)
            pipes.append(pipe.y)
            pipes.append(pipe.w)
            pipes.append(pipe.h)
        for pipe in flappy.pipes.lower:
            pipes.append(pipe.x)
            pipes.append(pipe.y)
            pipes.append(pipe.w)
            pipes.append(pipe.h)

        floor = flappy.floor.y

        # simulate 3 game ticks
        for i in range(10):
            # Apply the action to the future state
            player_y += player_vel_y
            player_vel_y += player_acc_y
            player_rot = clamp(player_rot + player_vel_rot, flappy.player.rot_min, flappy.player.rot_max)

            # Check if the player collides with the floor
            if player_y + player_h >= floor:
                return COLLISION

            # check for the top of the screen
            if player_y <= 0:
                return COLLISION

            # Check if the player collides with a pipe
            for i in range(0, len(pipes), 4):
                pipe_x = pipes[i] - (52 * i)
                pipe_y = pipes[i + 1]
                pipe_w = pipes[i + 2]
                pipe_h = pipes[i + 3]
                if player_x + player_w >= pipe_x and player_x <= pipe_x + pipe_w:
                    if player_y <= pipe_y + pipe_h or player_y + player_h >= pipe_y:
                        return COLLISION
                    elif player_y <= pipe_y + pipe_h or player_y + player_h >= pipe_y:
                        if player_x + player_w >= pipe_x and player_x <= pipe_x + pipe_w:
                            return CROSS

        return NOR_COLLISION_CROSS


class DQNAgent:
    """
    Deep Q-learning agent
    """

    def __init__(self, state_size, action_size):
        """
        Initialize the DQN agent
        :param state_size: the size of the state
        :param action_size: the size of the action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.6    # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.2
        self.action_space = 2
        self.model = self.build_model()
        self.counter = 0

    def build_model(self):
        """
        Build the neural network model
        :return: the neural network model
        """
        model = Sequential()
        model.add(Dense(6, input_shape=(6,), activation='relu'))  # Change the input_shape to (26,)
        model.add(Dense(36, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        try:
            self.load("weights")
            return self.model
        except Exception as e:
            print("Creating a new model")
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Remember the state, action, reward, next state, and done
        :param state: the current state
        :param action: the action taken
        :param reward: the reward received
        :param next_state: the next state
        :param done: whether the game is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Get the best action for the current state
        :param state: the current state
        :return: the best action
        """
        state = np.reshape(state, [1, -1])  # Reshape the state
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)  # No need to reshape the state here
        return np.argmax(act_values[0])

    def flatten(self, d):
        """
        Flatten a dictionary
        :param d: the dictionary to flatten
        :return: the flattened dictionary
        """
        flat_list = []
        for value in d.values():
            if isinstance(value, list):
                flat_list.extend(value)
            else:
                flat_list.append(value)
        return flat_list

    def replay(self, batch_size):
        """
        Replay the memory
        :param batch_size: the size of the batch
        """
        minibatch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, minibatch_size)

        states = np.zeros((minibatch_size, 6))
        next_states = np.zeros((minibatch_size, 6))
        actions, rewards, dones = [], [], []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state if not isinstance(state, dict) else np.array(list(state.values()), dtype=np.float32)
            next_states[i] = next_state if not isinstance(next_state, dict) else np.array(list(next_state.values()), dtype=np.float32)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        # Predict Q(s,a) and Q(s',a') for all a
        target = self.model.predict(states)
        target_next = self.model.predict(next_states)

        for i in range(minibatch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        # Train the neural network with all samples in the minibatch at once
        self.model.fit(states, target, epochs=10, verbose=0)

        # Increment the counter each time replay is called
        self.counter += 1

        # Save the model's weights every N iterations
        if self.counter % 100 == 0:
            self.save("weights")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name + ".weights.h5")
        print("Loaded model weights")

    def save(self, name):
        self.model.save_weights(name + ".weights.h5")

    def learn(self, state, action, reward, new_state):
        self.remember(state, action, reward, new_state, False)
        self.replay(32)

    def get_action(self, state):
        return self.act(state)


class MyGame:
    """
    A custom Flappy Bird game class for the agent to interact with

    """

    def __init__(self, agent=QLearningAgent):
        self.flappy = src.flappy.Flappy()
        self.agent = agent()

    async def start(self):
        """
        Start the game and the agent
        """
        state = self.get_initial_state()
        iterations = 100

        while iterations > 0:
            async for new_state, action, game in self.flappy.start(self):
                if state is not None:
                    reward = self.get_reward(new_state, game)
                    self.agent.learn(state, action, reward, new_state, game)
                state = new_state
                if self.flappy.player.crashed:
                    iterations -= 1
                    state = self.get_initial_state()

    def get_initial_state(self):
        """
        Get the initial state of the game
        :return: the initial state of the game
        """
        player = np.array([56, 260.5, 34])
        pipes = np.array([444, -144])
        time = np.array([0])
        self.agent.q_table[np.array2string(np.concatenate((player, pipes, time)))] = {0: 0, 1: 0}
        return np.array([56., 260.5, 34., 444., -144., 0.])        # return {

    def getPipesmiddle(self, game):
        """ return the middle of the pipes """
        return game.pipes.upper[0].x + game.pipes.upper[0].w / 2

    def get_reward(self, new_state, game):
        """ return the reward based on the new state """
        reward = FLOOR - abs(new_state[0] + new_state[1] - self.getPipesmiddle(game))
        for i, pipe in enumerate(game.pipes.upper):
            if game.player.crossed(pipe):
                return BIG_NUMBER
        if game.player.crashed:
            return -BIG_NUMBER
        return reward


def main():
    """
    Main function
    """
    game = MyGame()
    asyncio.run(game.start())


if __name__ == "__main__":
    main()
