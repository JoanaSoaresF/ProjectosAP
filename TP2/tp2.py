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
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from Utils import plot_statistics, create_folders
from game_demo import plot_board
from network_utils import convolution_stack_layer, dense_block
from policy import policy
from snake_game import SnakeGame

# PATHS
DRIVE_PATH = '/content/drive/MyDrive/2ºSemestre/Aprendizagem Profunda/ProjectosAP/TP2'
now = datetime.utcnow().strftime("%Y-%m-%d_%Hh%Mmin")
WEIGHTS_PATH = './weights/'
GAME_NAME = 'menos_policy_fill_random'
GAMES_PATH = './games/'
TRAIN_PATH = './train/'
PLOTS_PATH = './plots/'

# PLOT OPTION
PLOT_TRAIN = False
TRAIN = True
NUM_GAMES = 10
CHOSEN_MODEL = f'2022-06-14_08h54minmenos_policy_fill_random'

# Parameters
TRAIN_PERIODICITY = 6
POLICY_PERIODICITY = 10
MAX_EPSILON = 1
MIN_EPSILON = 0.1
DECAY = 0.01
MIN_REPLAY_SIZE = 2048
TRAIN_EPISODES = 1000
DISCOUNT_FACTOR = 0.8
BATCH_SIZE = 1024
BATCH_RESIZE = 4
LEARNING_RATE = 0.003
UPDATE_MODEL = 100

# GAME PARAMETERS
BOARD_DIM = 14
BORDER = 9
FOOD = 10
GRASS_GROW = 0.001
MAX_GRASS = 0.05

VERSION_INFORMATION = f'\n*******************************************************************************************\n' \
                      f'Running on: {now}\n' \
                      f'Game: {GAME_NAME}\n' \
                      f'Epsilon: max:{MAX_EPSILON} min:{MIN_EPSILON} decay:{DECAY}\n' \
                      f'Episodes: {TRAIN_EPISODES}; Replay: {MIN_REPLAY_SIZE}; Batch size: {BATCH_SIZE}; Batch resize: {BATCH_RESIZE}; Learning rate:{LEARNING_RATE}\n' \
                      f'Train periodicity: {TRAIN_PERIODICITY};Policy periodicity: {POLICY_PERIODICITY}; update model: {UPDATE_MODEL} steps' \
                      f'Discount factor: {DISCOUNT_FACTOR}\n' \
                      f'Game: {BOARD_DIM}x{BOARD_DIM} with {BORDER} border; food: {FOOD}; grass: {MAX_GRASS} with grow {GRASS_GROW}\n'


def agent(state_shape):
    """The agent maps states to actions, computing the  q-function value"""

    inputs = Input(shape=state_shape, name='inputs')

    # convolutional layers stacked
    # convolutional_layer = convolution_stack_layer(inputs, 128, (2, 2))
    convolutional_layer = convolution_stack_layer(inputs, 64, (2, 2))
    convolutional_layer = convolution_stack_layer(convolutional_layer, 32, (2, 2))
    # convolutional_layer = convolution_stack_layer(convolutional_layer, 16, (2, 2))

    # Flatten before dense layers
    layer = Flatten()(convolutional_layer)

    # Dense layers
    # layer = dense_block(layer, 256, dropout=False)
    layer = dense_block(layer, 128, dropout=False)
    layer = dense_block(layer, 64, dropout=False)
    layer = dense_block(layer, 32, dropout=False)
    # layer = dense_block(layer, 16, dropout=False)

    output = Dense(3, activation='linear')(layer)

    model = Model(inputs, output)
    model.summary()
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss=Huber(),
                  optimizer=opt,
                  metrics=['mse'])
    return model


def train_agent(replay_memory, model, target_model):
    mini_batch = random.sample(replay_memory, BATCH_SIZE)

    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)

    new_current_states = np.array([transition[3] for transition in mini_batch])
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

    model.fit(np.array(X), np.array(Y), batch_size=int(BATCH_SIZE / BATCH_RESIZE), verbose=0, shuffle=True)


def training_episodes(replay_memory):
    action_space = [-1, 0, 1]
    epsilon = MAX_EPSILON  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    snake_game = SnakeGame(width=BOARD_DIM, height=BOARD_DIM, food_amount=FOOD,
                           border=BORDER, grass_growth=GRASS_GROW,
                           max_grass=MAX_GRASS)
    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent((32, 32, 3))
    # Target Model (updated every 100 steps)
    target_model = agent((32, 32, 3))
    target_model.set_weights(model.get_weights())

    steps_to_update_target_model = 0
    steps_statistics = []
    reward_statistics = []

    for episode in range(TRAIN_EPISODES):
        print(f"Training episode {episode}")
        episode_path = f'{TRAIN_PATH}{now}{GAME_NAME}/e{episode}/'

        if PLOT_TRAIN:
            create_folders(episode_path)

        total_training_rewards = 0
        board, reward, done, score_dict = snake_game.reset()

        if PLOT_TRAIN:
            plot_board(f'{episode_path}{0}.png', board, f"Start")

        done = False
        step = 0

        while not done:
            steps_to_update_target_model += 1
            # board = snake_game.board_state()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy

            if episode % POLICY_PERIODICITY == 0:
                # one game played by policy every ten episodes
                tag = "Policy"
                score, apple, head, tail, direction = snake_game.get_state()
                action = policy(score, apple, head, tail, direction)
            elif random_number <= epsilon:
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

            new_board_state, reward, done, score_dict = snake_game.step(action)
            replay_memory.append([board, action, reward, new_board_state, done])
            board = new_board_state
            total_training_rewards += reward
            step = step + 1
            if PLOT_TRAIN:
                plot_board(f'{episode_path}{step}.png', board, f"step{step}_{tag}")

            # 3. Update the Main Network using the Bellman Equation

            if len(replay_memory) >= MIN_REPLAY_SIZE and \
                    (steps_to_update_target_model % TRAIN_PERIODICITY == 0 or done):
                # The model is trained only once there are enough examples in the experience pool. Once that
                # condition is met, the model is trained every four steps
                train_agent(replay_memory, model, target_model)

            if done:
                # print(exploit_actions)
                print('\tTotal training rewards: {} after n steps = {}'.format(
                        total_training_rewards, step))

                if steps_to_update_target_model >= UPDATE_MODEL and len(replay_memory) >= MIN_REPLAY_SIZE:
                    print('\tCopying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY * episode)

        steps_statistics.append(step)
        reward_statistics.append(total_training_rewards)

    tf.keras.models.save_model(model, f'{WEIGHTS_PATH}{now}{GAME_NAME}')

    path = f'{PLOTS_PATH}{now}{GAME_NAME}'
    create_folders(path)
    plot_statistics(steps_statistics, "Steps", f'{path}/steps.png')
    plot_statistics(reward_statistics, "Rewards", f'{path}/rewards.png')


def play_trained_model(action_space, model_path, plot_path):
    model = load_model(model_path)
    done = False
    snake_game = SnakeGame(width=BOARD_DIM, height=BOARD_DIM, food_amount=FOOD / 2,
                           border=BORDER, grass_growth=0,
                           max_grass=0)
    step = 1
    board = snake_game.board_state()
    total_reward = 0

    create_folders(plot_path)

    while not done:
        reshaped = np.reshape(board, (-1, 32, 32, 3))
        predicted = model.predict(reshaped).flatten()
        action = action_space[np.argmax(predicted)]
        board, reward, done, _ = snake_game.step(action)
        plot_board(f"{plot_path}{step}.png", board, f"step = {step}")
        total_reward += reward
        step += 1
        if step == 500:
            break
    print("****** Game ended ******")
    game_result = f"Steps took = {step}\nTotal Reward = {total_reward}"
    print(game_result)
    return game_result


def fill_memory_with_policy_moves(replay_memory):
    action_space = [-1, 0, 1]
    snake_game = SnakeGame(width=BOARD_DIM, height=BOARD_DIM, food_amount=FOOD,
                           border=BORDER, grass_growth=GRASS_GROW,
                           max_grass=MAX_GRASS)
    while len(replay_memory) <= MIN_REPLAY_SIZE:

        board, reward, done, score_dict = snake_game.reset()
        done = False
        while not done:
            action = random.sample(action_space, 1)[0]
            new_board_state, reward, done, score_dict = snake_game.step(action)
            replay_memory.append([board, action, reward, new_board_state, done])
            board = new_board_state


if __name__ == '__main__':
    print(VERSION_INFORMATION)

    replay_memory = deque(maxlen=100000)

    if TRAIN:
        path = f'{WEIGHTS_PATH}{now}{GAME_NAME}'
        fill_memory_with_policy_moves(replay_memory)

        training_episodes(replay_memory)

        for game in range(NUM_GAMES):
            print(f"******* Playing game {game}*********")
            path_plot = f'{GAMES_PATH}{now}{GAME_NAME}-{game}/'
            game_result = play_trained_model([-1, 0, 1], path, path_plot)

            game_string = f"Playing game result:\n{game_result}\n"
        end = f'*******************************************************************************************\n'
        f = open("runs_info", "a")
        f.write(f"{VERSION_INFORMATION}\n{game_string}\n{end}")
        f.close()
    else:
        for game in range(NUM_GAMES):
            print(f"******* Playing game {game} *********")
            game_result = play_trained_model([-1, 0, 1], f'{WEIGHTS_PATH}{CHOSEN_MODEL}/',
                                             f'{GAMES_PATH}{CHOSEN_MODEL}-{game}/')
            print(f'********************************************\n')
