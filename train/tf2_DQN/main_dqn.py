import copy

import numpy as np

from agent.fix_rule.agent import Agent
from interface import Environment
from train.tf2_DQN import dqn

MAP_PATH = 'maps/1000_1000_fighter1v1.map'

RENDER = True
MAX_EPOCH = 1000
BATCH_SIZE = 200
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
DETECTOR_NUM = 0
FIGHTER_NUM = 1
COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM
LEARN_INTERVAL = 100

if __name__ == "__main__":
    blue_agent = Agent()
    red_agent_obs_ind = 'simple'  # ???
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []
    blue_detector_action=[]    #////
    fighter_model = dqn.RLFighter(ACTION_NUM)

    # 执行
    for x in range(MAX_EPOCH):
        step_cnt = 0
        env.reset()
        while True:
            obs_list = []
            action_list = []
            red_fighter_action = []

            if step_cnt == 0:
                red_obs_dict, blue_obs_dict = env.get_obs()

            _, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)
            obs_got_ind = [False] * red_fighter_num
            for y in range(red_fighter_num):
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                if red_obs_dict['fighter'][y]['alive']:
                    obs_got_ind[y] = True
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                    tmp_info_obs = red_obs_dict['fighter'][y]['info']
                    tmp_action = fighter_model.choose_action(tmp_img_obs, tmp_info_obs)
                    obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                    action_list.append(tmp_action)

                    true_action[0] = int(360 / COURSE_NUM * int(tmp_action[0] / ATTACK_IND_NUM))
                    true_action[3] = int(tmp_action[0] % ATTACK_IND_NUM)
                red_fighter_action.append(true_action)
            red_fighter_action = np.array(red_fighter_action)

            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

            # get reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            detector_reward = red_detector_reward + red_game_reward
            fighter_reward = red_fighter_reward + red_game_reward

            # save replay
            red_obs_dict, blue_obs_dict = env.get_obs()
            for y in range(red_fighter_num):
                if obs_got_ind[y]:
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                    tmp_info_obs = red_obs_dict['fighter'][y]['info']
                    fighter_model.store_transition(obs_list[y], action_list[y], fighter_reward[y],
                                                   {'screen': copy.deepcopy(tmp_img_obs),
                                                    'info': copy.deepcopy(tmp_info_obs)})

            if env.get_done():
                fighter_model.learn()
                break

            if (step_cnt > 0) and (step_cnt % LEARN_INTERVAL == 0):
                fighter_model.learn()
            step_cnt += 1