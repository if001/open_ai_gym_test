import gym
import numpy as np
import predictor
import math
import matplotlib.pyplot as plt

DEBUG = False


def get_status(_observation):
    env_low = env.observation_space.low  # 位置と速度の最小値
    env_high = env.observation_space.high  # 　位置と速度の最大値
    env_dx = (env_high - env_low) / 40  # 40等分
    # 0〜39の離散値に変換する
    position = int((_observation[0] - env_low[0])/env_dx[0])
    velocity = int((_observation[1] - env_low[1])/env_dx[1])
    return position, velocity


def get_status_zero_one(_observation):
    env_low = env.observation_space.low  # 位置と速度の最小値
    env_high = env.observation_space.high  # 　位置と速度の最大値

    position = 1/(env_high[0] - env_low[0]) * _observation[0] + \
        (-env_low[0]/(env_high[0] - env_low[0]))

    velocity = 1/(env_high[1] - env_low[1]) * _observation[1] + \
        (-env_low[1]/(env_high[1] - env_low[1]))
    return position, velocity


def update_q_table(_q_table, _action,  _observation, _next_observation, _reward, _episode):
    alpha = 0.2  # 学習率
    gamma = 0.99  # 時間割引き率

    # 行動後の状態で得られる最大行動価値 Q(s',a')
    next_position, next_velocity = get_status(_next_observation)
    next_max_q_value = max(_q_table[next_position][next_velocity])

    # 行動前の状態の行動価値 Q(s,a)
    position, velocity = get_status(_observation)
    q_value = _q_table[position][velocity][_action]

    # 行動価値関数の更新
    _q_table[position][velocity][_action] = q_value + \
        alpha * (_reward + gamma * next_max_q_value - q_value)

    return _q_table


def get_action(_env, _q_table, _observation, _episode):
    epsilon = 0.002
    if np.random.uniform(0, 1) > epsilon:
        position, velocity = get_status(observation)
        _action = np.argmax(_q_table[position][velocity])
    else:
        _action = np.random.choice([0, 1, 2])
    return _action


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    predictor = predictor.Predictor()

    # Qテーブルの初期化
    q_table = np.zeros((40, 40, 3))

    observation = env.reset()
    rewards = []
    loss = []
    errors = []
    steps = []
    loop = 10000
    loop = 50

    # 10000エピソードで学習する
    for episode in range(loop):

        total_reward = 0
        observation = env.reset()
        # env.render()
        each_errors = []
        train_x = []
        train_y = []
        for step in range(200):
            # env.render()
            # ε-グリーディ法で行動を選択
            action = get_action(env, q_table, observation, episode)
            _pos, _vel = get_status(observation)
            predict = predictor.predict(np.array([[_pos, _vel, action]]))

            # 車を動かし、観測結果・報酬・ゲーム終了FLG・詳細情報を取得
            next_observation, reward, done, _ = env.step(action)
            _pos, _vel = get_status_zero_one(next_observation)
            error = math.sqrt(
                math.pow(predict[0][0] - _pos, 2) + math.pow(predict[0][1] - _vel, 2))
            each_errors.append(error)

            if DEBUG:
                print("predict:", predict)
                print("actual:", _pos, _vel)
                print("error:", error)

            # Qテーブルの更新
            q_table = update_q_table(
                q_table, action, observation, next_observation, reward, episode)
            total_reward += (reward + error)

            _pos, _vel = get_status_zero_one(observation)
            next_pos, next_vel = get_status_zero_one(next_observation)
            train_x.append([_pos, _vel, action/2.0])
            train_y.append([next_pos, next_pos])

            if DEBUG:
                print("train")
                print(_pos, _vel, action/2.0)
                print(next_pos, next_pos)
                print("------")

            observation = next_observation

            if done:
                print(reward, step)
                # doneがTrueになったら１エピソード終了
                if episode % 100 == 0:
                    print('episode: {}, total_reward: {}'.format(
                        episode, total_reward))
                rewards.append(total_reward)
                break

        h = predictor.train(
            np.array(train_x),
            np.array(train_y)
        )

        loss.append(h.history['loss'][0])

        errors.append(np.array(each_errors).mean())

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(loss)
    ax.set_title('loss')
    plt.savefig("loss.png")

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(errors)
    ax.set_title('errors')
    plt.savefig("errors.png")

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.plot(steps)
    ax.set_title('step')
    plt.savefig("step.png")

    plt.show()
