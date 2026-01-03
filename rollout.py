import numpy as np
import torch
import math
from torch.distributions import one_hot_categorical
import time
import torch
import math
from highway_env import utils
from torch.distributions import one_hot_categorical
import time
from itertools import combinations
import numpy as np
import idm_controller
from mdp_controller import mdp_controller
from queue import PriorityQueue
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from idm_controller import idm_controller, generate_actions
import copy
from highway_env.vehicle.kinematics import Vehicle

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def find_matching_conflicts(self, vehicle_data, vehicle_data1, conflict_points):
        matches = []
        # 提取 vehicle1 的起点和终点
        vehicle_start = vehicle_data[1]  # vehicle1 的起点
        vehicle_end = vehicle_data[2]  # vehicle1 的终点

        # 提取 vehicle2 的起点和终点
        vehicle_start1 = vehicle_data1[1]  # vehicle2 的起点
        vehicle_end1 = vehicle_data1[2]  # vehicle2 的终点
        for point in conflict_points:
            # conflict_point 格式: (x, y, start1, start2, dest1, dest2)
            start1, start2 = point[2], point[3]
            dest1, dest2 = point[4], point[5]
            # 检查两辆车的起点和终点是否与冲突点的起点和终点匹配
            # 情况1: vehicle1 对应 start1/dest1，vehicle2 对应 start2/dest2
            condition1 = (
                    (vehicle_start == start1 and vehicle_end == dest1) and
                    (vehicle_start1 == start2 and vehicle_end1 == dest2)
            )

            # 情况2: vehicle1 对应 start2/dest2，vehicle2 对应 start1/dest1
            condition2 = (
                    (vehicle_start == start2 and vehicle_end == dest2) and
                    (vehicle_start1 == start1 and vehicle_end1 == dest1)
            )

            # 如果满足任意一种情况，则记录该冲突点
            if condition1 or condition2:
                matches.append(point[0])
                matches.append(point[1])
                break

        return matches

    def calculate_RAT(self, X_conf1_ij, X_conf1_ji, V_vehicle_i, V_vehicle_j):
        """
        计算相对到达时间差(RAT)

        参数:
            X_conf1_ij: 冲突点i和j的位置坐标 (数组或标量)
            X_vehicle_j: 车辆j的位置坐标 (数组或标量)
            V_vehicle_i: 车辆i的速度 (标量)
            V_vehicle_j: 车辆j的速度 (标量)

        返回:
            RAT_i: 相对到达时间差
        """
        term_i = (X_conf1_ij) / V_vehicle_i
        term_j = (X_conf1_ji) / V_vehicle_j
        RAT = np.abs(term_i - term_j)
        return RAT

    def check_safety_room(self, vehicle, action, surrounding_vehicles, env_copy, time_steps, conflict_pos):
        """
        para: vehicle: the ego vehicle
              surrounding_vehicles: [v_fl, v_rl, v_fr, v_rr]
              env_copy: copy of self
              vehicle.trajectories = [vehicle.position, vehicle.heading, vehicle.speed]
              return: the minimum safety room with surrounding vehicles in the trajectory
        """

        minrisk_list1 = []
        print("conflict_pos11111111",conflict_pos)
        # collect new trajectories
        for t in range(time_steps + 1):
            # obj = 0
            mdp_controller(vehicle, env_copy, action)
            minrisk_list = []
            for vj in surrounding_vehicles:
                if self.has_passed_conflict(vehicle.trajectories[t][0],conflict_pos,vehicle.destination) and self.has_passed_conflict(vj.trajectories[t][0],conflict_pos,vj.destination):
                    other = np.sqrt((conflict_pos[0] - vj.trajectories[t][0][0]) ** 2 + (
                            conflict_pos[1] - vj.trajectories[t][0][1]) ** 2)
                    ego = np.sqrt((conflict_pos[0] - vehicle.trajectories[t][0][0]) ** 2 + (
                            conflict_pos[1] - vehicle.trajectories[t][0][1]) ** 2)
                    TDTC_other = other / vj.trajectories[t][2]
                    TDTC_ego = ego / vehicle.trajectories[t][2]
                    TDTC = abs(TDTC_ego - TDTC_other)
                    minrisk_list.append(TDTC)
                    minrisk = min(minrisk_list)
                    minrisk_list1.append(minrisk)
        return min(minrisk_list1) if minrisk_list1 else float("inf")

    def has_passed_conflict(self, vehicle_pos, conflict_pos, end_pos):
        # 计算当前位置到终点的距离
        dist_vehicle_to_end = np.linalg.norm(np.array(vehicle_pos) - np.array(end_pos))
        # 计算冲突点到终点的距离
        dist_conflict_to_end = np.linalg.norm(np.array(conflict_pos) - np.array(end_pos)) - 5
        # 如果车辆距离终点更近，说明已经驶过冲突点
        return dist_vehicle_to_end > dist_conflict_to_end

    def _is_colliding(self, vehicle, other, other_trajectories):
        # Fast spherical pre-check
        # other_trajectories: [vehicle.position, vehicle.heading, vehicle.speed]

        # Euclidean distance
        if np.linalg.norm(other_trajectories[0] - vehicle.position) > 3 * vehicle.LENGTH:  ##3
            return False

        # Accurate rectangular check
        return utils.rotated_rectangles_intersect(
            (vehicle.position, 2 * vehicle.LENGTH, 2 * vehicle.WIDTH, vehicle.heading),
            (other_trajectories[0], 2 * other.LENGTH, 2 * other.WIDTH, other_trajectories[1]))  ##2

    def check_collision(self, vehicle, other, other_trajectories):
        """
        Check for collision with another vehicle.

        :param other: the other vehicle' trajectories or object
        other_trajectories: [vehicle.position, vehicle.heading, vehicle.speed]
        """
        if vehicle.crashed or other is vehicle:
            return

        if isinstance(other, Vehicle):
            if self._is_colliding(vehicle, other, other_trajectories):
                vehicle.speed = other_trajectories[2] = min([vehicle.speed, other_trajectories[2]], key=abs)
                vehicle.crashed = other.crashed = True

    def safety_supervisor(self, actions, conflit_array, vehicle_listtotal, avail_actions):
        """"
        implementation of safety supervisor
        """
        actions = list(actions)
        env_copy = copy.deepcopy(self.env)
        n_points = 12 ##12
        """compute the priority of controlled vehicles"""
        q = PriorityQueue()
        vehicles_and_actions = []  # original vehicle and action

        # reset the trajectories
        for v in env_copy.road.vehicles:
            v.trajectories = []
        vehicle_list1 = []
        vehicle_list2 = [1000, 1000, 1000, 1000]
        ###############################################################################################
        # 1. 建立车辆匹配关系（基于位置）
        for vehicle in self.env.road.vehicles:
            a = np.array(vehicle.route)
            if id(vehicle) not in [item[3] for item in vehicle_listtotal]:  # 检查 vehicle 是否已存在
                vehicle_listtotal.append([vehicle, a[0][0], a[2][1], id(vehicle)])  # 如果不存在，则加入
        # 2. 建立车辆匹配关系（基于位置）
        matching_pairs = []  # 存储 (original, copied) 对
        for orig in self.env.road.vehicles:
            for copied in env_copy.road.vehicles:
                if np.allclose(orig.position, copied.position, atol=0.1):
                    matching_pairs.append((orig, copied))
                    break

        # 3. 更新车辆列表（仅替换车辆对象）
        for item in vehicle_listtotal:
            current_vehicle = item  # 获取当前车辆对象
            # 查找是否需要替换
            for orig, copied in matching_pairs:
                if id(orig) == current_vehicle[3]:  # 使用原车辆的 ID 进行匹配
                    item[0] = copied
                    break

        # 5. 生成近距离车辆列表（距离原点小于 20）
        new_vehicle_list = [
            item[0] for item in vehicle_listtotal
            if math.sqrt(item[0].position[0] ** 2 + item[0].position[1] ** 2) < 25  ##30  ##25hao  40
        ]
        print("new_vehicle_list1111111",new_vehicle_list)
        ################################################################
        for i, vehicle in enumerate(new_vehicle_list):
            for j, vehicle1 in enumerate(new_vehicle_list):
                if i >= j:  # 只比较一次：确保vehicle1的索引大于vehicle的索引
                    continue  # 如果i >= j，跳过当前循环，不做比较
                # 然后继续后续的判断
                if vehicle in [item[0] for item in vehicle_listtotal] and vehicle1 in [item[0] for item in
                                                                                       vehicle_listtotal]:
                    for item in vehicle_listtotal:
                        if item[0] == vehicle:
                            vehicle_data = item  # 获取当前车辆完整数据
                            for item1 in vehicle_listtotal:
                                if item1[0] == vehicle1:
                                    vehicle_data1 = item1  # 获取对比车辆完整数据
                                    # 检查两车是否存在路径冲突
                                    matching_points = self.find_matching_conflicts(vehicle_data, vehicle_data1,
                                                                                   conflit_array)
                                    if matching_points != []:  # 如果存在冲突点
                                        matching_points = np.array(matching_points)
                                        a, b = map(float, matching_points[:2])  # 转换为浮点数
                                        conflict_pos = [a, b]
                                        # 计算两车到冲突点的距离
                                        conflit1 = np.sqrt(
                                            (a - vehicle.position[0]) ** 2 + (b - vehicle.position[1]) ** 2)
                                        conflit2 = np.sqrt(
                                            (a - vehicle1.position[0]) ** 2 + (b - vehicle1.position[1]) ** 2)
                                        # 计算两车速度
                                        speed1 = np.linalg.norm(vehicle.velocity)
                                        speed2 = np.linalg.norm(vehicle1.velocity)
                                        TTR1 = conflit1 / speed1
                                        TTR2 = conflit2 / speed2
                                        if self.has_passed_conflict(vehicle.position, conflict_pos,
                                                                    vehicle.destination) and self.has_passed_conflict(
                                            vehicle1.position, conflict_pos, vehicle1.destination):
                                            # 计算到达时间比
                                            RAT = self.calculate_RAT(conflit1, conflit2, speed1, speed2)
                                            if RAT < 4:  ###3  #4 hao ##5
                                                vehicle_list1.append([[vehicle, TTR1], [vehicle1, TTR2]])
        print("new_vehicle_list22222222", vehicle_list1)
        ###############################################################################################################
        index = 0
        for vehicle, action in zip(env_copy.controlled_vehicles, actions):
            vehicle.id = index + 1
            conflit = 10000
            for pair in vehicle_list1:
                # pair 的格式: [[vehicle_obj, conflit1], [vehicle1_obj, conflit2]]
                if vehicle == pair[0][0]:  # 检查是否是第一个车辆
                    conflit = pair[0][1]
                    break
                elif vehicle == pair[1][0]:  # 检查是否是第二个车辆
                    conflit = pair[1][1]
                    break
            if conflit < vehicle_list2[index]:
                vehicle_list2[index] = conflit
            priority_number = vehicle_list2[index]
            # speed = np.linalg.norm(vehicle.velocity)
            # priority_number = calculate_TTR(conf,speed)
            priority_number += np.random.rand() * 0.001  # to avoid the same priority number for two vehicles
            q.put((priority_number, [vehicle, action, index]))
            index += 1
        # q is ordered from large to small numbers
        while not q.empty():
            next_item = q.get()
            vehicles_and_actions.append(next_item[1])
        for i, vehicle_and_action in enumerate(vehicles_and_actions):
            first_change = True  # only do the first change

            # # if the vehicle is stepped before, reset it
            if len(vehicle_and_action[0].trajectories) == n_points:
                action = vehicle_and_action[1]
                index = vehicle_and_action[2]
                env_copy.controlled_vehicles[index] = copy.deepcopy(self.env.controlled_vehicles[index])
                vehicle = env_copy.controlled_vehicles[index]
                env_copy.road.vehicles[index] = vehicle
            else:
                vehicle = vehicle_and_action[0]
                action = vehicle_and_action[1]
                index = vehicle_and_action[2]
            # avail_actions
            available_actions = avail_actions[vehicle_and_action[2]]
            available_actions = [action for action in available_actions if action != -1]
            vehicle_array = []  ##chongtuche
            for pair in vehicle_list1:
                # pair 的格式: [[vehicle_obj, conflit1], [vehicle1_obj, conflit2]]
                if vehicle == pair[0][0]:  # 检查是否是第一个车辆
                    vehicle_array.append(pair[1][0])
                    continue
                elif vehicle == pair[1][0]:  # 检查是否是第二个车辆
                    vehicle_array.append(pair[0][0])
                    continue
            # propograte the vehicle for n steps
            for t in range(n_points):
                # consider the front vehicles first
                for v in vehicle_array:
                    if v is None:
                        continue
                    # skip if the vehicle has been stepped before
                    if len(v.trajectories) == n_points and i != 0 and v is not vehicle:
                        pass

                    # other surrounding vehicles
                    else:
                        if type(v) is IDMVehicle:
                            # determine the action in the first time step
                            if t == 0:
                                a = generate_actions(v, env_copy)
                                idm_controller(v, env_copy, a)
                            else:
                                idm_controller(v, env_copy, v.action)

                        elif type(v) is MDPVehicle and v is not vehicle:
                            # use the previous action: idle
                            mdp_controller(v, env_copy, actions[v.id - 1])
                        elif type(v) is MDPVehicle and v is vehicle:
                            if actions[index] == action:
                                mdp_controller(v, env_copy, action)
                            else:
                                # take the safe action after replace
                                mdp_controller(v, env_copy, actions[index])

                # check collision for every time step TODO: Check
                for other in vehicle_array:
                    if isinstance(other, Vehicle):
                        self.check_collision(vehicle, other, other.trajectories[t])

                if vehicle.crashed:
                    # TODO: check multiple collisions during n_points
                    # replace with a safety action
                    safety_rooms = []
                    updated_vehicles = []
                    candidate_actions = []
                    for a in available_actions:
                        vehicle_copy = copy.deepcopy(self.env.controlled_vehicles[index])
                        safety_room = self.check_safety_room(vehicle_copy, a, vehicle_array,
                                                             env_copy, t, conflict_pos)
                        updated_vehicles.append(vehicle_copy)
                        candidate_actions.append(a)
                        safety_rooms.append(safety_room)

                    # reset the vehicle trajectory associated with the new action
                    env_copy.controlled_vehicles[index] = updated_vehicles[safety_rooms.index(max(safety_rooms))]
                    vehicle = env_copy.controlled_vehicles[index]
                    env_copy.road.vehicles[index] = vehicle
                    if first_change:
                        first_change = False
                        actions[index] = candidate_actions[safety_rooms.index(max(safety_rooms))]
                    # actions[index] = 0
                    # TODO: check the collision after replacing the action
                    # reset its neighbor's crashed as False if True
                    for other in vehicle_array:
                        if isinstance(other, Vehicle) and other.crashed:
                            other.crashed = False

        return actions, vehicle_listtotal

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        queue = []  # 用来存储车辆的队列
        leavequeue = []  # 用来存储车辆的队列
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.seed(episode_num)
        sorigin = self.env.reset()
        print("s", sorigin)
        terminated = False
        win_tag = False
        has_arrived = [False for _ in range(4)]
        step = 0
        collision_punishment = -6
        totalspeed2 = 0
        conflit_point = [(-4.490373330306471, 0.07031759134898201, 'o0', 'o1', 'o1', 'o2'),
                         (0.06755501608435971, 4.713627842131126, 'o0', 'o3', 'o1', 'o0'),
                         (-1.5802851215919231, 2.188475532022709, 'o0', 'o1', 'o1', 'o3'),
                         (-1.813177479247075, 2.4097533165431253, 'o0', 'o2', 'o1', 'o0'),
                         (-10.125924674447237, -1.8137718414191917, 'o0', 'o3', 'o1', 'o1'),
                         (-14.169350588126612, -2.033259811370598, 'o0', 'o2', 'o1', 'o1'),
                         (-0.09719856082548263, -4.755552206153439, 'o1', 'o2', 'o2', 'o3'),
                         (1.8506321440697184, -11.059641971289594, 'o1', 'o0', 'o2', 'o2'),
                         (-2.0967369281994275, -1.7791844498701228, 'o1', 'o2', 'o2', 'o0'),
                         (-2.2793463293813305, -1.9004622343470863, 'o1', 'o3', 'o2', 'o1'),
                         (2.0312068172859483, -14.424327501145928, 'o1', 'o3', 'o2', 'o2'),
                         (4.439164412091117, -0.0954293591010804, 'o2', 'o3', 'o3', 'o0'),
                         (1.9843198455531708, -2.104236656586939, 'o2', 'o0', 'o3', 'o2'),
                         (9.921544119627566, 1.8089189535834849, 'o2', 'o1', 'o3', 'o3'),
                         (1.9017104443782848, -2.0118081595714536, 'o2', 'o3', 'o3', 'o1'),
                         (14.636454841036315, 2.0303196032628454, 'o2', 'o0', 'o3', 'o3'),
                         (2.092531927978712, 1.912056131395394, 'o3', 'o0', 'o0', 'o2'),
                         (2.1254242856338834, 1.9044846284108794, 'o3', 'o1', 'o0', 'o3'),
                         (-1.8264000856072304, 10.295249492219822, 'o3', 'o2', 'o0', 'o0'),
                         (-2.0341068655007883, 13.875554425414546, 'o3', 'o1', 'o0', 'o0'),
                         (2.0328923576551716, 2.007571502984516, 'o0', 'o1', 'o2', 'o3'),
                         (1.917390598825114, -2.0924284970154856, 'o0', 'o3', 'o2', 'o1'),
                         (2.1673752766293974, -11.85450188348776, 'o0', 'o3', 'o2', 'o2'),
                         (-2.067107642344859, 1.9212777845188027, 'o1', 'o2', 'o3', 'o0'),
                         (12.187285051617872, 2.1593136862721796, 'o1', 'o0', 'o3', 'o3'),
                         (-1.8826094011816439, -1.8787222155230365, 'o2', 'o3', 'o0', 'o1'),
                         (-2.1904825592615644, 11.429407094783446, 'o2', 'o1', 'o0', 'o0'),
                         (-12.713598681060244, -2.1315006225861106, 'o3', 'o2', 'o1', 'o1')]

        # 转换为NumPy数组
        conflit_array = np.array(conflit_point)
        vehicle_listtotal = []
        low_speed_punishment = 0.1
        arrived_reward = 3.0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        pro = []
        for vehicle in self.env.controlled_vehicles:
            a = np.linalg.norm(vehicle.destination - vehicle.position)
            pro.append(a)
        while not terminated and step < self.episode_limit:
            reward = 0
            # self.env.render()
            obs = np.squeeze(np.reshape(sorigin, (4, 30)))
            state = np.squeeze(np.reshape(sorigin, (120, 1)))
            actions, avail_actions, actions_onehot = [], [], []
            # actions, actions_onehot = [], []
            actions1 = [0, 0, 0, 0]
            ###############################################
            for agent_id in range(self.n_agents):
                min_value = min(pro)  # 找到最小值
                min_index = pro.index(min_value)  # 找到最小值的索引
                d = np.sqrt(obs[agent_id][1] ** 2 + obs[agent_id][2] ** 2)
                if d < 65 and min_index == agent_id: ##55
                    if agent_id not in queue and agent_id not in leavequeue:
                        pro[min_index] = 100000
                        queue.append(agent_id)
                if agent_id == 0:
                    if obs[agent_id][1] < -5:
                        if agent_id not in leavequeue:
                            leavequeue.append(agent_id)
                if agent_id == 1:
                    if obs[agent_id][2] < -5:
                        if agent_id not in leavequeue:
                            leavequeue.append(agent_id)
                if agent_id == 2:
                    if obs[agent_id][1] > 5:
                        if agent_id not in leavequeue:
                            leavequeue.append(agent_id)
                if agent_id == 3:
                    if obs[agent_id][2] > 5:
                        if agent_id not in leavequeue:
                            leavequeue.append(agent_id)
                if agent_id in leavequeue and agent_id in queue:
                    # 如果有 0，从 queue 中删除第一个 0
                    queue.remove(agent_id)
                if queue:
                    if (queue[0] == 0 or queue[0] == 1 or queue[0] == 2 or queue[0] == 3):
                        agents_id = queue[0]  # 取出第一辆车
                    if queue[0] == 0 or queue[0] == 1 or queue[0] == 2 or queue[0] == 3:
                        actions1[agents_id] = 2
                if agent_id in leavequeue:
                    actions1[agent_id] = 1
           ###########################################################################################
            if actions1 == [0, 0, 0, 0]:
                avail_action = [0, 1, 2]
            for agent_id in range(self.n_agents):
                if actions1[agent_id] == 2:
                    avail_action = [-1, -1, 2]
                if actions1[agent_id] == 0:
                    avail_action = [0, 1, -1]
                if actions1[agent_id] == 1:
                    avail_action = [0, 1, 2]
                ####################################################33
                if agent_id == 0:
                    front_vehicle00, rear_vehicle00 = self.env.road.neighbour_vehicles(
                        self.env.controlled_vehicles[0],
                        self.env.controlled_vehicles[0].lane_index)
                    # 获取 ego_vehicle 和 front_vehicle1 的位置
                    if front_vehicle00 != None:
                        speed00 = front_vehicle00.speed - self.env.controlled_vehicles[0].speed
                        d0 = self.env.controlled_vehicles[0].lane_distance_to(front_vehicle00)
                        # 计算两者之间的欧几里得距离
                        distance00 = abs(d0)
                        if speed00 != 0:
                            ttc00 = distance00 / speed00
                        else:
                            ttc00 = 10
                        if (distance00 < self.env.controlled_vehicles[1].speed) or ttc00 < 1.5:
                            avail_action = [0, -1, -1]
                    if self.env.controlled_vehicles[0].lane_index != self.env.controlled_vehicles[0].target_lane_index:
                        front_vehicle00, rear_vehicle00 = self.env.road.neighbour_vehicles(
                            self.env.controlled_vehicles[0], self.env.controlled_vehicles[0].target_lane_index)
                        if front_vehicle00 != None:
                            speed00 = front_vehicle00.speed - self.env.controlled_vehicles[0].speed
                            d0 = self.env.controlled_vehicles[0].lane_distance_to(front_vehicle00)
                            # 计算两者之间的欧几里得距离
                            distance00 = abs(d0)
                            # 计算两者之间的欧几里得距离
                            if speed00 != 0:
                                ttc00 = distance00 / speed00
                            else:
                                ttc00 = 10
                            if (distance00 < self.env.controlled_vehicles[1].speed) or ttc00 < 1.5:
                                avail_action = [0, -1, -1]
                        else:
                            pass
                if agent_id == 1:
                    front_vehicle10, rear_vehicle10 = self.env.road.neighbour_vehicles(
                        self.env.controlled_vehicles[1],
                        self.env.controlled_vehicles[1].lane_index)
                    # 获取 ego_vehicle 和 front_vehicle1 的位置
                    if front_vehicle10 != None:
                        speed10 = front_vehicle10.speed - self.env.controlled_vehicles[1].speed
                        d1 = self.env.controlled_vehicles[1].lane_distance_to(front_vehicle10)
                        # 计算两者之间的欧几里得距离
                        distance10 = abs(d1)
                        if speed10!=0:
                            ttc10 = distance10 / speed10
                        else:
                            ttc10 = 10
                        if (distance10 < self.env.controlled_vehicles[1].speed) or ttc10 < 1.5:
                            avail_action = [0, -1, -1]
                    if self.env.controlled_vehicles[1].lane_index != self.env.controlled_vehicles[1].target_lane_index:
                        front_vehicle10, rear_vehicle10 = self.env.road.neighbour_vehicles(
                            self.env.controlled_vehicles[1],
                            self.env.controlled_vehicles[
                                1].target_lane_index)
                        if front_vehicle10 != None:
                            speed10 = front_vehicle10.speed - self.env.controlled_vehicles[1].speed
                            d1 = self.env.controlled_vehicles[1].lane_distance_to(front_vehicle00)
                            # 计算两者之间的欧几里得距离
                            distance10 = abs(d1)
                            # 计算两者之间的欧几里得距离
                            if speed10 != 0:
                                ttc10 = distance10 / speed10
                            else:
                                ttc10 = 10
                            if (distance10 < self.env.controlled_vehicles[1].speed) or ttc10 < 1.5:
                                avail_action = [0, -1, -1]
                        else:
                            pass
                if agent_id == 2:
                    front_vehicle20, rear_vehicle20 = self.env.road.neighbour_vehicles(
                        self.env.controlled_vehicles[2],
                        self.env.controlled_vehicles[2].lane_index)
                    # 获取 ego_vehicle 和 front_vehicle1 的位置
                    if front_vehicle20 != None:
                        speed20 = front_vehicle20.speed - self.env.controlled_vehicles[2].speed
                        d2 = self.env.controlled_vehicles[2].lane_distance_to(front_vehicle20)
                        # 计算两者之间的欧几里得距离
                        distance20 = abs(d2)
                        if speed20 != 0:
                            ttc20 = distance20 / speed20
                        else:
                            ttc20 = 10
                        if (distance20 < self.env.controlled_vehicles[1].speed) or ttc20 < 1.5:
                            avail_action = [0, -1, -1]
                    if self.env.controlled_vehicles[2].lane_index != self.env.controlled_vehicles[2].target_lane_index:
                        front_vehicle20, rear_vehicle20 = self.env.road.neighbour_vehicles(
                            self.env.controlled_vehicles[2],
                            self.env.controlled_vehicles[
                                2].target_lane_index)
                        if front_vehicle20 != None:
                            speed20 = front_vehicle20.speed - self.env.controlled_vehicles[2].speed
                            d2 = self.env.controlled_vehicles[2].lane_distance_to(front_vehicle20)
                            # 计算两者之间的欧几里得距离
                            distance20 = abs(d2)
                            # 计算两者之间的欧几里得距离
                            if speed20 != 0:
                                ttc20 = distance20 / speed20
                            else:
                                ttc20 = 10
                            if (distance20 < self.env.controlled_vehicles[1].speed) or ttc20 < 1.5:
                                avail_action = [0, -1, -1]
                        else:
                            pass
                if agent_id == 3:
                    front_vehicle30, rear_vehicle30 = self.env.road.neighbour_vehicles(
                        self.env.controlled_vehicles[3],
                        self.env.controlled_vehicles[3].lane_index)
                    # 获取 ego_vehicle 和 front_vehicle1 的位置
                    if front_vehicle30 != None:
                        speed30 = front_vehicle30.speed - self.env.controlled_vehicles[3].speed
                        d3 = self.env.controlled_vehicles[3].lane_distance_to(front_vehicle30)
                        # 计算两者之间的欧几里得距离
                        distance30 = abs(d3)
                        if speed30 != 0:
                            ttc30 = distance30 / speed30
                        else:
                            ttc30 = 10
                        if (distance30 < self.env.controlled_vehicles[1].speed) or ttc30 < 1.5:
                            avail_action = [0, -1, -1]
                    if self.env.controlled_vehicles[3].lane_index != self.env.controlled_vehicles[3].target_lane_index:
                        front_vehicle30, rear_vehicle30 = self.env.road.neighbour_vehicles(
                            self.env.controlled_vehicles[3],
                            self.env.controlled_vehicles[
                                3].target_lane_index)
                        if front_vehicle30 != None:
                            speed30 = front_vehicle30.speed - self.env.controlled_vehicles[3].speed
                            d3 = self.env.controlled_vehicles[3].lane_distance_to(front_vehicle30)
                            # 计算两者之间的欧几里得距离
                            distance30 = abs(d3)
                            # 计算两者之间的欧几里得距离
                            if speed30 != 0:
                                ttc30 = distance30 / speed30
                            else:
                                ttc30 = 10
                            if (distance30 < self.env.controlled_vehicles[1].speed) or ttc30 < 1.5:
                                avail_action = [0, -1, -1]
                        else:
                            pass
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon)
                # generate onehot vector of th action
                # action_onehot = np.zeros(self.args.n_actions)
                # action_onehot[action] = 1
                actions.append(np.int(action))
                # actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                # last_action[agent_id] = action_onehot
                avail_actions1 = [[0,1,2],[0,1,2],[0,1,2],[0,1,2]]
            print("actions22222222222222222222222", actions)
            actions, vehicle_listtotal = self.safety_supervisor(actions, conflit_array, vehicle_listtotal,
                                                                avail_actions1)
            print("actions3333333333333333",actions)
            agent_id = 0
            for ab in actions:
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[ab] = 1
                actions_onehot.append(action_onehot)
                last_action[agent_id] = action_onehot
                agent_id = agent_id + 1
            s_, reward1, terminated, info_n = self.env.step(tuple(actions))  ##4:slow  ##3:speed 2:speed 0:jia 1:jia

            agents_speeds = (eval(str(info_n))).get("speed")
            agents_dones = (eval(str(info_n))).get("agents_dones")
            crasheds = (eval(str(info_n))).get("crashed")
            arrived = (eval(str(info_n))).get("arrived")
            crashed = False
            for elem in crasheds:
                if elem:
                    crashed = True

            scaled_speed = []
            totalspeed = 0
            for speed in agents_speeds:
                totalspeed += speed
                # scaled_speed.append(lmap(speed, low_speed_range, [0, 1]))
                scaled_speed.append(speed / 10)
            totalspeed1 = totalspeed / 4
            if not crashed:
                for i in range(4):
                    if scaled_speed[i] > 0.3:
                        reward += low_speed_punishment * scaled_speed[i]
            else:
                for i in range(4):
                    reward += crasheds[i] * collision_punishment

            for i in range(4):
                if arrived[i] and not has_arrived[i]:
                    reward += arrived_reward
                    has_arrived[i] = True
            if all(arrived):
                terminated = True
            print("terminated",terminated)
            win_tag = True if terminated and all(arrived) else False
            if step >= self.episode_limit - 1:
                win_tag = False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            totalspeed2 += totalspeed1
            sorigin = s_

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        obs = np.squeeze(np.reshape(s_, (4, 30)))
        state = np.squeeze(np.reshape(s_, (1, 120)))
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            # avail_action = [0,1,2]
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        return episode, episode_reward, win_tag, step, episode_reward/step,totalspeed2 / step
