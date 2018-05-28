import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

try:
    xrange = xrange
except:
    xrange = range

env = gym.make('CartPole-v0')