# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import pickle
import pandas as pd

# 数据中每个片段的长度
EPISODE_LENGTH = 400
original = True

class DataManager(object):
    POS_RANGE_MAX = 4.5
    POS_RANGE_MIN = -4.5

    def __init__(self):
        if original:
            data = pickle.load(open("data/data.p", "rb"))
            input_x = data["x"]  # (49999, 7)
            # 角度(-1~1), 速度_x, 速度_z, 角速度, 速度大小, 朝向?, 速度?
            # 速度_x和速度_z可以得出速度大小。已经确认了这一点。
            # 朝向不太清楚。
            # 最后的速度与速度大小的1/5相等。
            input_y = data["y"]  # (49999, 2)
            # 位置_x, 位置_z
            # 位置的范围大约在-4.03 ~ +4.03之间。
            self.pos_xs = input_y[:, 0]  # PositionX 对应列
            self.pos_ys = input_y[:, 1]  # PositionY 对应列
            
            self.input_y = input_y  # (49999, 2)
            self.linear_velocities = input_x[:, 4]  # (49999,)
            self.angular_velocities = input_x[:, 3]  # (49999,)
            self.angles = input_x[:, 0] * np.pi  # (49999,) -pi~pi
        else:
            file_path = 'data\\vehicle_data.csv'
            data = pd.read_csv(file_path)
            input_x = data[['LinearVelocityX', 'LinearVelocityY', 'AngularVelocityZ', 'Yaw']].values
            input_y = data[['PositionX', 'PositionY']].values

            # 提取线速度
            self.linear_velocities_x = input_x[:, 0]  # LinearVelocityX 对应列
            self.linear_velocities_y = input_x[:, 1]  # LinearVelocityY 对应列

            # 提取角速度
            self.angular_velocities = input_x[:, 2]  # AngularVelocityZ 对应列

            # 提取角度
            self.angles = input_x[:, 3]  # Yaw 对应列

            # 提取位置坐标
            self.pos_xs = input_y[:, 0]  # PositionX 对应列
            self.pos_ys = input_y[:, 1]  # PositionY 对应列

            self.input_y = input_y  # (49999, 2)

    def prepare(self):
        data_size = self.linear_velocities.shape[0]

        # 准备输入数据
        self.inputs = np.empty([data_size, 4], dtype=np.float32)
        self.inputs[:, 0] = self.angular_velocities
        self.inputs[:, 1] = self.angles
        if original:
            self.inputs[:, 2] = np.cos(self.linear_velocities)
            self.inputs[:, 3] = np.sin(self.linear_velocities)
        else:
            self.inputs[:, 2] = self.linear_velocities_x
            self.inputs[:, 3] = self.linear_velocities_y
        # 准备输出数据
        self.place_outputs = self.input_y

    def get_train_batch(self, batch_size, sequence_length):
        episode_size = (self.linear_velocities.shape[0] + 1) // EPISODE_LENGTH

        inputs_batch = np.empty([batch_size, sequence_length, self.inputs.shape[1]])
        place_outputs_batch = np.empty([batch_size, sequence_length, self.place_outputs.shape[1]])
        place_init_batch = np.empty([batch_size, self.place_outputs.shape[1]])

        for i in range(batch_size):
            episode_index = np.random.randint(0, episode_size)
            pos_in_episode = np.random.randint(0, episode_size - (sequence_length + 1))
            if episode_index == episode_size - 1 and pos_in_episode == episode_size - (sequence_length + 1) - 1:
                # 最后一个片段比其他片段短1步
                pos_in_episode -= 1
            pos = episode_index * EPISODE_LENGTH + pos_in_episode
            inputs_batch[i, :, :] = self.inputs[pos:pos + sequence_length, :]
            place_outputs_batch[i, :, :] = self.place_outputs[pos + 1:pos + sequence_length + 1, :]
            place_init_batch[i, :] = self.place_outputs[pos, :]

        return inputs_batch, place_outputs_batch, place_init_batch

    def get_confirm_index_size(self, batch_size, sequence_length):
        # 总片段数 (=125)
        episode_size = (self.linear_velocities.shape[0] + 1) // EPISODE_LENGTH
        # 每个片段的序列数 (=4)
        sequence_per_episode = EPISODE_LENGTH // sequence_length
        return (episode_size * sequence_per_episode // batch_size) - 1

    def get_confirm_batch(self, batch_size, sequence_length, index):
        inputs_batch = np.empty([batch_size, sequence_length, self.inputs.shape[1]])
        place_init_batch = np.empty([batch_size, self.place_outputs.shape[1]])
        place_pos_batch = np.empty([batch_size, sequence_length, 2])

        sequence_per_episode = EPISODE_LENGTH // sequence_length
        sequence_index = index * batch_size

        for i in range(batch_size):
            episode_index = sequence_index // sequence_per_episode
            pos_in_episode = (sequence_index % sequence_per_episode) * sequence_length
            pos = episode_index * EPISODE_LENGTH + pos_in_episode
            inputs_batch[i, :, :] = self.inputs[pos:pos + sequence_length, :]
            place_init_batch[i, :] = self.place_outputs[pos, :]
            place_pos_batch[i, :, 0] = self.pos_xs[pos + 1:pos + sequence_length + 1]
            place_pos_batch[i, :, 1] = self.pos_ys[pos + 1:pos + sequence_length + 1]
            sequence_index += 1

        return inputs_batch, place_init_batch, place_pos_batch
