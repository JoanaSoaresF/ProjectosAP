# ##############################################################################
#  Aprendizagem Profunda, TP2 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################
import random
from collections import deque
from datetime import datetime

import numpy as np
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import tensorflow as tf
from tensorflow.keras.models import load_model

from Utils import convolution_stack_layer, dense_block, plot_statistics, create_folders
from game_demo import plot_board
from snake_game import SnakeGame

# PATHS
DRIVE_PATH = '/content/drive/MyDrive/2ºSemestre/Aprendizagem Profunda/ProjectosAP/TP2'
now = datetime.utcnow().strftime("_%Y-%m-%d_%Hh%Mmin")
WEIGHTS_PATH = './weights/'
GAME_NAME = 'cnn_agent_game'
GAMES_PATH = './games/'
TRAIN_PATH = './train/'
PLOTS_PATH = './plots/'

# PLOT OPTION
PLOT_TRAIN = False

# Parameters
MAX_EPSILON = 1
MIN_EPSILON = 0.01
DECAY = 0.01
MIN_REPLAY_SIZE = 1028
TRAIN_EPISODES = 400
DISCOUNT_FACTOR = 0.4
BATCH_SIZE = 512
LEARNING_RATE = 0.001

# GAME PARAMETERS
BOARD_DIM = 14
BORDER = 9
FOOD = 5
GRASS_GROW = 0
MAX_GRASS = 0


def agent(state_shape, action_shape):
    """The agent maps states to actions, computing the  q-function value"""

    inputs = Input(shape=state_shape, name='inputs')

    # convolutional layers stacked
    convolutional_layer = convolution_stack_layer(inputs, 64, (2, 2))
    convolutional_layer = convolution_stack_layer(convolutional_layer, 32, (2, 2))
    convolutional_layer = convolution_stack_layer(convolutional_layer, 16, (2, 2))

    # Flatten before dense layers
    layer = Flatten()(convolutional_layer)

    # Dense layers
    # layer = dense_block(layer, 256, dropout=False)
    layer = dense_block(layer, 128, dropout=False)
    layer = dense_block(layer, 64, dropout=False)

    output = Dense(3)(layer)

    model = Model(inputs, output)
    model.summary()
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=Huber(),
                  optimizer=opt,
                  metrics=['mse'])
    return model


def train(replay_memory, model, target_model):
    mini_batch = random.sample(replay_memory, BATCH_SIZE)

    current_states = np.array([action[0] for action in mini_batch])
    current_qs_list = model.predict(current_states)

    new_current_states = np.array([action[3] for action in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + DISCOUNT_FACTOR * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action + 1] = max_future_q
        X.append(observation)
        Y.append(current_qs)

    model.fit(np.array(X), np.array(Y), batch_size=BATCH_SIZE, verbose=1, shuffle=True)


def play_trained_model(action_space):
    path = f'{WEIGHTS_PATH}{GAME_NAME}{now}'
    model = load_model(path)
    done = False
    snake_game = SnakeGame(width=BOARD_DIM, height=BOARD_DIM, food_amount=FOOD,
                           border=BORDER, grass_growth=GRASS_GROW,
                           max_grass=MAX_GRASS)
    step = 1
    board = snake_game.board_state()
    total_reward = 0

    game_folder = f"{GAMES_PATH}{GAME_NAME}{now}/"
    create_folders(game_folder)

    while not done:
        reshaped = np.reshape(board, (-1, 32, 32, 3))
        predicted = model.predict(reshaped).flatten()
        action = action_space[np.argmax(predicted)]
        board, reward, done, _ = snake_game.step(action)
        plot_board(f"{game_folder}{step}.png", board, f"step = {step}")
        total_reward += reward
        step += 1
        if step == 1000:
            break
    print("****** Game ended ******")
    print(f"Steps took = {step}\nTotal Reward = {total_reward}")


def main():
    action_space = [-1, 0, 1]
    epsilon = MAX_EPSILON  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    snake_game = SnakeGame(width=BOARD_DIM, height=BOARD_DIM, food_amount=FOOD,
                           border=BORDER, grass_growth=GRASS_GROW,
                           max_grass=MAX_GRASS)
    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent((32, 32, 3), 3)
    # Target Model (updated every 100 steps)
    target_model = agent((32, 32, 3), 3)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=100_000)

    steps_to_update_target_model = 0
    steps_statistics = []
    reward_statistics = []

    for episode in range(TRAIN_EPISODES):
        print(f"Training episode {episode}")
        episode_path = f'{TRAIN_PATH}{GAME_NAME}/e{episode}/'

        if PLOT_TRAIN:
            create_folders(episode_path)

        total_training_rewards = 0
        board, reward, done, score_dict = snake_game.reset()
        if PLOT_TRAIN:
            plot_board(f'{episode_path}{0}.png', board, f"Start")
        done = False
        step = 0
        decay_epsilon = False
        exploit_actions = []

        while not done:
            steps_to_update_target_model += 1
            # board = snake_game.board_state()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                # action = env.action_space.sample()
                tag = "Explore"
                action = random.sample(action_space, 1)[0]
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                tag = "Exploit"
                reshaped = np.reshape(board, (-1, 32, 32, 3))
                predicted = model.predict(reshaped).flatten()
                action = action_space[np.argmax(predicted)]
                exploit_actions.append(action)

            new_board_state, reward, done, score_dict = snake_game.step(action)
            replay_memory.append([board, action, reward, new_board_state, done])
            board = new_board_state
            total_training_rewards += reward
            step = step + 1
            if PLOT_TRAIN:
                plot_board(f'{episode_path}{step}.png', board, f"step{step}_{tag}")

            # 3. Update the Main Network using the Bellman Equation

            if len(replay_memory) >= MIN_REPLAY_SIZE and \
                    (steps_to_update_target_model % 4 == 0 or done):
                # The model is trained only once there are enough examples in the experience pool. Once that
                # condition is met, the model is trained every four steps

                decay_epsilon = True
                train(replay_memory, model, target_model)

            if done:
                # print(exploit_actions)
                print('\tTotal training rewards: {} after n steps = {}'.format(
                        total_training_rewards, step))

                if steps_to_update_target_model >= 100 and len(replay_memory) >= MIN_REPLAY_SIZE:
                    print('\tCopying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        if decay_epsilon:
            epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY * episode)

        steps_statistics.append(step)
        reward_statistics.append(total_training_rewards)

    tf.keras.models.save_model(model, f'{WEIGHTS_PATH}{GAME_NAME}{now}')
    # model.save(f'{WEIGHTS_PATH}{GAME_NAME}{now}.h5')
    path = f'{PLOTS_PATH}{GAME_NAME}{now}'
    create_folders(path)
    plot_statistics(steps_statistics, "Steps", f'{path}/steps.png')
    plot_statistics(reward_statistics, "Rewards", f'{path}/rewards.png')


if __name__ == '__main__':
    # now = '_2022-06-10_21h14min'
    now = datetime.utcnow().strftime("_%Y-%m-%d_%Hh%Mmin")
    main()
    print("******* Playing game *********")
    play_trained_model([-1, 0, 1])
