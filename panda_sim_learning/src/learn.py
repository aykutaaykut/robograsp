#!/usr/bin/env python

import os
import sys
import pwd
import math
import random
import datetime
import numpy as np
import rospy
import moveit_commander
from keras.models import load_model
from robot_env import RobotEnv
from dqn import Agent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPISODES = 10000
TIME_STEPS = 300
SAVE_NETWORK = True
LOAD_NETWORK = False
LOAD_PATH = '/home/' + pwd.getpwuid(os.getuid())[0] + '/catkin_ws/src/dqn_models/2019-05-16 11:47:11.009540/model_e_2_t_300.h5'

LOG_DIR = '/home/' + pwd.getpwuid(os.getuid())[0] + '/catkin_ws/src/dqn_log/'
LOG_FILE = 'dqn_log_' + str(datetime.datetime.now()) + '.txt'
MODEL_DIR = '/home/' + pwd.getpwuid(os.getuid())[0] + '/catkin_ws/src/dqn_models/' + str(datetime.datetime.now()) + '/'

def log(str):
    with open(LOG_DIR + LOG_FILE, 'a+') as f:
        f.write(str + '\n')

if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('robot_env', anonymous = True)
    rospy.sleep(1)

    env = RobotEnv()
    env.reset()
    state_dim = env.state_dim
    action_dim = env.action_dim
    dqn_agent = Agent(state_dim, action_dim)

    if SAVE_NETWORK:
        os.mkdir(MODEL_DIR)

    if LOAD_NETWORK:
        try:
            dqn_agent.model = load_model(LOAD_PATH)
            print '#################### MODEL IS LOADED SUCCESSFULLY! ####################'
        except:
            print '#################### NO SUCH FILE TO LOAD: ' + LOAD_PATH
            raise Exception("No such file to load")

    reward_list = []
    grasp_success_list = []
    grasp_success_rate_list = []

    for e in range(1, EPISODES+1):
        dist_list = []
        log('###################################################')
        log('#################### EPISODE ' + str(e) + ' ' + '#'*(20-int(math.log(e, 10))))
        log('###################################################')
        state = env.reset()
        state = np.reshape(state, [1, state_dim])
        total_reward = 0
        for t in range(1, TIME_STEPS+1):
            action = dqn_agent.act(state)
            # log('############### ITERATION ' + str(t) + ' ' + '#'*(15-int(math.log(t, 10))))
            state_next, reward, terminal, info, next_distance, successful_grasping = env.step(action)
            total_reward += reward
            dist_list.append(next_distance)
            state_next = np.reshape(state_next, [1, state_dim])
            # log('State: ' + str(state))
            # log('Action: ' + str(action))
            # log('Reward: ' + str(reward))
            # log('Next State: ' + str(state_next))
            # log('Done: ' + str(terminal))
            dqn_agent.remember(state, action, reward, state_next, terminal)
            state = state_next

            if terminal:
                log('##### END || episode: ' + str(e) + ' || time: ' + str(t) + ' || score: ' + str(total_reward) + ' || grasp_success: ' + str(successful_grasping))
                break
            dqn_agent.experience_replay()
        if SAVE_NETWORK and e%10 == 0:
            SAVE_PATH = MODEL_DIR + 'model_e_' + str(e) + '.h5'
            dqn_agent.model.save(SAVE_PATH)
        plt.figure()
        plt.plot(dist_list)
        plt.savefig(LOG_DIR + 'distance_figs/distance_' + str(e) + '.jpg')
        plt.close()
        reward_list.append(total_reward)
        if successful_grasping:
            grasp_success_list.append(1)
        else:
            grasp_success_list.append(0)
        grasp_success_rate = sum(grasp_success_list[-100:]) / float(len(grasp_success_list[-100:]))
        grasp_success_rate_list.append(grasp_success_rate)
        with open(LOG_DIR + 'reward_list.txt', 'a+') as f:
            f.write(str(total_reward) + '\n')
        with open(LOG_DIR + 'grasp_success_rate.txt', 'a+') as f:
            f.write(str(grasp_success_rate) + '\n')

    plt.figure()
    plt.plot(reward_list)
    plt.savefig(LOG_DIR + 'reward.jpg')
    plt.close()

    plt.figure()
    plt.plot(grasp_success_rate_list)
    plt.savefig(LOG_DIR + 'grasp_success_rates.jpg')
    plt.close()
