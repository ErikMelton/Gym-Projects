import gym

env = gym.make('CartPole-v0')

best_score = 0

for i_episode in range(20):
    observation = env.reset()
    points = 0 # Our reward value

    while True:
        env.render()
        if observation[2] > 0:
            action = 1
        else:
            action = 0

        observation, reward, done, info = env.step(action)
        points += reward

        if done:
            if points > best_score:
                best_score = points
                break