# ##############################################################################
#  Aprendizagem Profunda, TP2 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense

from TP2.snake_game import SnakeGame


def agent(state_shape, action_shape, learning_rate=0.001):
    """The agent maps states to actions, computing the  q-function value"""

    init = tf.keras.initializers.HeUniform()
    inputs = Input(shape=state_shape, name='inputs')
    layer = Dense(32, activation='relu', kernel_initializer=init)(inputs)
    layer = Dense(16, activation='relu', kernel_initializer=init)(layer)
    output = Dense(action_shape, activation='linear', kernel_initializer=init)(layer)

    model = Model(inputs, output)
    model.summary()
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['mse'])
    return model


def train(replay_memory, model, target_model, discount_factor, batch_size):
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([action[0] for action in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([action[3] for action in mini_batch])
    future_qs_list = target_model.predict(new_current_states)
    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = max_future_q
        X.append(observation)
        Y.append(current_qs)

    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def main(max_epsilon=1, min_epsilon=0.01, decay=0.01,
         MIN_REPLAY_SIZE=1000,
         train_episodes=300,
         discount_factor=0.618,
         batch_size=200):
    """
    Args:
        batch_size:
        discount_factor:
        train_episodes:
        MIN_REPLAY_SIZE:
        max_epsilon: You can't explore more than 100% of the time
        min_epsilon: At a minimum, we'll always explore 1% of the time
        decay:

    Returns:
    """
    action_space = [-1, 0, 1]
    epsilon = max_epsilon  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    snake_game = SnakeGame(width=32, height=32, food_amount=1,
                           border=0, grass_growth=0,
                           max_grass=0)
    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent((32, 32, 3), 3)
    # Target Model (updated every 100 steps)
    target_model = agent((32, 32, 3), 3)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        total_training_rewards = 0
        board, reward, done, score_dict = snake_game.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            snake_game.board_state()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                # action = env.action_space.sample()
                action = random.sample(action_space, 1)[0]
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                reshaped = board.reshape([1, board.shape[0]])
                predicted = model.predict(reshaped).flatten()
                action = action_space[np.argmax(predicted)]
            new_board_state, reward, done, score_dict = snake_game.step(action)
            replay_memory.append([board, action, reward, new_board_state, done])

            # 3. Update the Main Network using the Bellman Equation

            if len(replay_memory) >= MIN_REPLAY_SIZE and \
                    (steps_to_update_target_model % 4 == 0 or done):
                train(replay_memory, model, target_model, discount_factor=discount_factor, batch_size=batch_size)

            board = new_board_state
            total_training_rewards += reward

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
                        total_training_rewards, episode, reward))
                total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)


if __name__ == '__main__':
    main()
