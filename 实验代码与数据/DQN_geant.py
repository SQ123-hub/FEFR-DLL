import numpy as np
import tensorflow as tf
import random
import xlrd
import numpy as np
import openpyxl
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
import xlrd
import os
from collections import deque

def number(number, node):  # 源地址与目的地址赋值
    source_number = []
    for i in number:
        for j in node:
            if i == j:
                source_number.append(node.index(j) + 1)
    return source_number

def topNBetweeness(G):
    score = nx.betweenness_centrality(G)
    score = sorted(score.items(), key=lambda item: item[1], reverse=True)
    output = []
    for node in score:
        output.append(node[0])
    print(output)
    fout = open("betweennessSorted.data", 'w')
    for target in output:
        fout.write(str(target) + " ")
    return score

def addEdge(a, b):  # 该函数进行加边操作   构造一个完整的字典形如
    # 上式为演示的该函数处理完的结果
    global edgeLinks
    if a not in edgeLinks:
        edgeLinks[a] = set()
    if b not in edgeLinks:
        edgeLinks[b] = set()
    edgeLinks[a].add(b)
    edgeLinks[b].add(a)

def target_flow_addition(sources, targets, target_flow):  # 计算目的节点流量总量

    flow = []
    target_allflow_sp = []
    tar = []
    target_sets = list(set(targets))  # 源地址分类转化成列表形式
    targets_flow = list(zip(targets, target_flow))
    print(len(set(sources)))  # 部分节点无流量通过，因此事实节点少于50
    print(len(set(targets)))
    i = 0
    j = 0
    allflow = 0
    for target1 in target_sets:
        allflow = 0
        for target2 in targets_flow:
            if target2[0] == target1:  # 匹配节点，相同节点流量相加
                allflow = allflow + target2[1]
                flow.append(target_flow[j])
        tar.append(target1)
        target_allflow_sp.append(allflow)

    target_allflow = list(zip(tar, target_allflow_sp))
    return targets_flow, target_allflow

def allflow_sum(target_allflow):  # 计算总流量

    all = 0
    for flow in target_allflow:
        all = all + flow[1]
    return all

def Hi_pi(target_flow, target_allflow):  # 计算熵值

    Hi1 = []
    Hi2 = []
    Pi = []
    tar = []
    tar1 = []
    hi = 0
    node_sort = []
    max_flow = 0
    min_flow = 0

    for node in target_allflow:
        node_sort.append(node[0])

    for i in target_allflow:
        for j in target_flow:
            if i[0] == j[0]:
                pi = j[1] / i[1]
                Pi.append(pi)
                tar.append(j[0])

    target_pi = list(zip(tar, Pi))

    for node in node_sort:
        hi = 0
        for key in target_pi:
            if node == key[0]:
                p = (-1) * (key[1]) * math.log(key[1])
                hi = hi + p
        tar1.append(node)
        Hi1.append(hi)
    for key in Hi1:
        if max_flow < key:
            max_flow = key
        if min_flow > key:
            min_flow = key
    for key in Hi1:  # 标准化公式
        p = (key - min_flow) / (max_flow - min_flow)

        Hi2.append(p)
    target_hi = list(zip(tar1, Hi2))
    return target_pi, target_hi

def GNT_computer(h1, h2, p1, p2):  # 计算流量差值

    h = []
    p = []
    GNT = []
    tar = []
    tar2 = []

    for k1 in h1:
        for k2 in h2:
            if k1[0] == k2[0]:
                h.append(k2[1] - k1[1])
        tar.append(k1[0])
    # print(h1）
    h = list(zip(tar, h))

    for k1 in p1:
        for k2 in p2:
            if k1[0] == k2[0]:
                p.append(k2[1] - k1[1])
        tar2.append(k1[0])
    p = list(zip(tar2, p))

    for k1 in h:
        for k2 in p:
            if k1[0] == k2[0]:
                if k1[1] == 0:
                    GNT.append(k2[1])
                else:
                    GNT.append(k2[1] / k1[1])
    GNT = list(zip(tar2, GNT))
    return p, h, GNT

def topo_standard(betweenness_centrality):
    p = 0
    target = []
    To1 = []
    To2 = []
    all_topo = 0
    min_topo = betweenness_centrality[0][1]
    max_topo = betweenness_centrality[0][1]

    for key in betweenness_centrality:
        all_topo = all_topo + key[1]

    for key in betweenness_centrality:
        target.append(key[0])
        p = ((key[1] / all_topo) + 1) * math.log((key[1] / all_topo) + 1)  # 加1，避免有节点介数中心值为0
        if min_topo > p:
            min_topo = p
        if max_topo < p:
            max_topo = p
        To1.append(p)
    for key in To1:  # 标准化公式
        p = (key - min_topo) / (max_topo - min_topo)
        To2.append(p)
    topo_dict = list(zip(target, To2))
    return topo_dict

def degree_sum(flow_entropy, GNT, topo_entropy):
    # 指标的权重
    a = 0.3 #可调
    b = 0.7
    k = 0
    tar = []
    degree = []
    note_degree = []
    flow = flow_entropy.copy()
    GNT1 = GNT.copy()
    topo1 = topo_entropy.copy()
    len1 = len(topo1)
    for k in range(len1):
        if flow[k][0] != topo1[k][0]:
            flow.insert(k, (topo1[k][0], 0))  # 无流量经过节点填0
    for k in range(len1):
        if GNT1[k][0] != topo1[k][0]:
            GNT1.insert(k, (topo1[k][0], 0))
    for k in range(len1):
        if flow[k][0] == topo1[k][0] and topo1[k][0] == GNT1[k][0]:
            p = a * topo1[k][1] + b * GNT1[k][1]
        tar.append(topo1[k][0])
        degree.append(p)

    node_degree = dict(zip(tar, degree))
    #print("node_degree", node_degree)

    q = 0.2  # 节点分配比例
    p = 0.6
    o = 0.2
    len2 = len(node_degree)
    j = 1
    node_degree = sorted(node_degree.items(), key=lambda item: item[1])
    tar1 = []
    degree1 = []
    for node in node_degree:
        if j <= o * len2:
            tar1.append(node[0])
            degree1.append(1)
        elif j > o * len2 and j <= (o + p) * len2:
            tar1.append(node[0])
            degree1.append(2)
        else:
            tar1.append(node[0])
            degree1.append(3)
        j = j + 1

    node_degree1 = dict(zip(tar1, degree1))

    return node_degree1

def link_sum(link, GNT, topo_entropy):
    weight = []
    w = 0
    for key in link:
        w = GNT[key[0] - 1][1] + topo_entropy[key[0] - 1][1] + GNT[key[1] - 1][1] + topo_entropy[key[1] - 1][1]
        weight.append(w)
    return list(zip(link, weight))

def link_class(edge_links, node_degree):

    k = 0
    i = 0
    ol = []
    ol1 = []
    link_degree = []

    k = 0
    weight = 0
    for node in edge_links:
        i = node[0]
        j = node[1]
        k1 = node_degree[i]
        k2 = node_degree[j]
        k = k1 + k2
        if k < 4:
            weight = 1
            link_degree.append(weight)
        elif k >= 4 and k < 6:
            weight = 2
            link_degree.append(weight)
        elif k == 6:
            weight = 3
            link_degree.append(weight)

    link_degree = list(zip(edge_links, link_degree))

    return link_degree

def standard(data):  # 标准化公式
    min_flow = data[0][1]
    max_flow = data[0][1]
    tar = []
    target_allflow_standard = []

    for key in data:
        if key[1] < min_flow:
            min_flow = key[1]
        if key[1] > max_flow:
            max_flow = key[1]
    for key in data:
        p = (key[1] - min_flow) / (max_flow - min_flow)
        tar.append(key[0])
        target_allflow_standard.append(p)
    target_allflow_standard = list(zip(tar, target_allflow_standard))

    return target_allflow_standard

def link_degree_order(link_degree):
    order = link_degree.copy()
    list_len = len(link_degree)
    a = 0.3
    b = 0.5
    i = 0
    tar = []
    degree = []
    order.sort(key=lambda ele: ele[1], reverse=False)
    for key in order:
        if i < int(a * list_len):
            tar.append(key[0])
            degree.append(1)
        elif i < int((a + b) * list_len):
            tar.append(key[0])
            degree.append(2)
        else:
            tar.append(key[0])
            degree.append(3)
        i = i + 1
        order = list(zip(tar, degree))
    return order

def Dijkstra(network, s, d):  # 迪杰斯特拉算法算s-d的最短路径，并返回该路径和代价
    # print("Start Dijstra Path……")
    path = []  # s-d的最短路径
    n = len(network)  # 邻接矩阵维度，即节点个数
    fmax = 9999999
    w = [[0 for i in range(n)] for j in range(n)]  # 邻接矩阵转化成维度矩阵，即0→max
    book = [0 for i in range(n)]  # 是否已经是最小的标记列表
    dis = [fmax for i in range(n)]  # s到其他节点的最小距离
    book[s - 1] = 1  # 节点编号从1开始，列表序号从0开始
    midpath = [-1 for i in range(n)]  # 上一跳列表
    u = s - 1
    for i in range(n):
        for j in range(n):
            if network[i][j] != 0:
                w[i][j] = network[i][j]  # 0→max
            else:
                w[i][j] = fmax
            if i == s - 1 and network[i][j] != 0:  # 直连的节点最小距离就是network[i][j]
                dis[j] = network[i][j]
    for i in range(n - 1):  # n-1次遍历，除了s节点
        min = fmax
        for j in range(n):
            if book[j] == 0 and dis[j] < min:  # 如果未遍历且距离最小
                min = dis[j]
                u = j
        book[u] = 1
        for v in range(n):  # u直连的节点遍历一遍
            if dis[v] > dis[u] + w[u][v]:
                dis[v] = dis[u] + w[u][v]
                midpath[v] = u + 1  # 上一跳更新
    j = d - 1  # j是序号
    path.append(d)  # 因为存储的是上一跳，所以先加入目的节点d，最后倒置
    while (midpath[j] != -1):
        path.append(midpath[j])
        j = midpath[j] - 1
    path.append(s)
    path.reverse()  # 倒置列表
    return path

def return_path_sum(network, path):
    result = 0
    for i in range(len(path) - 1):
        result += network[path[i] - 1][path[i + 1] - 1]
    return result

def add_limit(path, s):  # path=[[[1,3,4,6],5],[[1,3,5,6],7],[[1,2,4,6],8]
    result = []
    for item in path:
        if s in item[0]:
            result.append([s, item[0][item[0].index(s) + 1]])
    result = [list(r) for r in list(set([tuple(t) for t in result]))]  # 去重
    return result

def return_shortest_path_with_limit(network, s, d, limit_segment, choice):  # limit_segment=[[3,5],[3,4]]
    mid_net = copy.deepcopy(network)
    for item in limit_segment:
        mid_net[item[0] - 1][item[1] - 1] = mid_net[item[1] - 1][item[0] - 1] = 0
    s_index = choice.index(s)
    for point in choice[:s_index]:  # s前面的点是禁用点
        for i in range(len(mid_net)):
            mid_net[point - 1][i] = mid_net[i][point - 1] = 0
    mid_path = Dijkstra(mid_net, s, d)
    return mid_path

def judge_path_legal(network, path):
    for i in range(len(path) - 1):
        if network[path[i] - 1][path[i + 1] - 1] == 0:
            return False
    return True

def k_shortest_path(network, s, d, k):
    k_path = []  # 结果列表
    alter_path = []  # 备选列表
    kk = Dijkstra(network, s, d)
    k_path.append([kk, return_path_sum(network, kk)])
    while (True):
        if len(k_path) == k: break
        choice = k_path[-1][0]
        for i in range(len(choice) - 1):
            limit_path = [[choice[i], choice[i + 1]]]  # 限制选择的路径
            if len(k_path) != 1:
                limit_path.extend(add_limit(k_path[:-1], choice[i]))
            mid_path = choice[:i]
            mid_res = return_shortest_path_with_limit(network, choice[i], d, limit_path, choice)
            if judge_path_legal(network, mid_res):
                mid_path.extend(mid_res)
            else:
                continue
            mid_item = [mid_path, return_path_sum(network, mid_path)]
            if mid_item not in k_path and mid_item not in alter_path:
                alter_path.append(mid_item)
        if len(alter_path) == 0:
            print("总共只有{}条最短路径！".format(len(k_path)))
            return k_path
        alter_path.sort(key=lambda x: x[-1])
        x = alter_path[0][-1]
        y = len(alter_path[0][0])
        u = 0
        for i in range(len(alter_path)):
            if alter_path[i][-1] != x:
                break
            if len(alter_path[i][0]) < y:
                y = len(alter_path[i][0])
                u = i
        k_path.append(alter_path[u])
        alter_path.pop(u)
    # for item in k_path:
    #   print(item)
    return k_path

def find_disjoint_shortest_paths(weighted_adj_matrix, source, target, num_paths=7):
    G = nx.Graph()
    num_nodes = len(weighted_adj_matrix)

    # 添加节点和边到图中
    for i in range(num_nodes):
        for j in range(num_nodes):
            if weighted_adj_matrix[i][j] != 0:
                G.add_edge(i, j, weight=weighted_adj_matrix[i][j])

    # 找到源节点到目标节点的所有最短路径
    all_shortest_paths = list(k_shortest_path(weighted_adj_matrix, source, target, 7))

    back_path = []
    disjoint_shortest_paths = []
    for key in all_shortest_paths:
        if len(key[0]) <= 2:
            disjoint_shortest_paths.append(key[0])
        elif len(key[0]) > 2:
            back_path.append(key[0])

    # 过滤出互不相交的路径并取前k条
    set1 = back_path[0][1:len(back_path[0]) - 1]
    #print("set1", set1)
    #print("back_path", back_path)
    for path in back_path:
        set1 = path[1:len(path) - 1]

        if not any(set(set1) & set(existing_path) for existing_path in disjoint_shortest_paths):
            disjoint_shortest_paths.append(path)
            if len(disjoint_shortest_paths) == num_paths:
                break

    #print("disjoint_shortest_paths", disjoint_shortest_paths)
    # 计算每条路径的权重和
    path_weights = []
    for path in disjoint_shortest_paths:
        weight_sum = sum(weighted_adj_matrix[path[i] - 1][path[i+1] - 1] for i in range(len(path)-1))
        path_weights.append(weight_sum)

    return [[path, weight] for path, weight in zip(disjoint_shortest_paths, path_weights)]

class DQN:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_steps

        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()

        self.replay_buffer = deque(maxlen=2000)

    def build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = np.array(state).flatten().reshape(1, -1)
            q_values = self.q_network.predict(np.array(state))
            return np.argmax(q_values)

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def train(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
        if len(self.replay_buffer) < 32:
            return

        batch = random.sample(self.replay_buffer, 32)
        states, actions, rewards, next_states = zip(*batch)

        states = np.array(states).reshape(32, -1)
        next_states = np.array(next_states).reshape(32, -1)
        rewards = np.array(rewards)

        target_q_values = rewards + self.gamma * np.max(self.target_network.predict(next_states), axis=1)
        q_values = self.q_network.predict(states)

        for i, action in enumerate(actions):
            q_values[i][action] = target_q_values[i]

        self.q_network.fit(states, q_values, epochs=1, verbose=0)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_rate
        else:
            self.epsilon = self.epsilon_end

# 实例化环境对象
class Environment:
    def __init__(self, link_weights_set, link_loads_set, backup_demand, num_links=22, link_state_dim=22):
        self.num_links = num_links
        self.link_state_dim = link_state_dim

        self.link_weights_set = link_weights_set
        self.link_loads_set = link_loads_set
        self.backup_demand = backup_demand
        #self.states = self._initialize_states()


    #def _initialize_states(self):
        # 初始化所有时间步的状态空间

        #return self.states

    def get_state(self, time):
        # 获取当前时间步的状态信息
        states = []
        self.current_time_step = time
        link_weights = self.link_weights_set[self.current_time_step]
        link_loads = self.link_loads_set[self.current_time_step]
        backup_demand = self.backup_demand[self.current_time_step]  # 0或1，表示是否需要备份路径

        #states.append(link_weights)
        #states.append(link_loads)
        #states.append(backup_demand)
        link_weights = np.array(link_weights)
        link_loads = np.array(link_loads)
        backup_demand = np.array(backup_demand)
        print("link_weights.shape", link_weights.shape)
        print("link_loads.shape", link_loads.shape)
        print("backup_demand.shape", backup_demand.shape)
        states = np.concatenate([link_weights.flatten(), link_loads.flatten(), backup_demand.flatten()])

        return states

    def execute_action(self, action, link, backup_path, link_load, backup_demand, link_bw):
        # 执行动作，并返回执行动作后的下一个状态信息和奖励
        # 在这里根据链路权重为3的链路确定备份路径的数量，并计算奖励

        # 备份路径
        demand = backup_demand[link[0] - 1][link[1] - 1] / link_bw
        backup_path = backup_path[0:action]

        # 对于每条链路权重为3的链路，计算其备份路径的数量
        x = 0
        y = 0
        z = 0
        for key1 in backup_path:
            x = x + len(key1[0])
            y = y + key1[1]
            r = 0
            for key2 in range(len(key1[0]) - 1):
                l_link = key1[0][key2]
                r_link = key1[0][key2 + 1]
                value = link_load[l_link - 1][r_link - 1]
                r = r + (1 - (value + demand / action))   #原有链路流量所占带宽，demand迁移带来的流量带宽
            r = r / len(key1[0])
            z = z + r

        # 根据选择的动作计算奖励
        if action == 0:
            reward = 0
        else:
            z = z / action
            reward = np.exp(-(x + y)) + z  #x + 1 为整数，取值范围为(0,1]

        return reward




    # 在这里编写环境和经验回放的逻辑，然后使用 DQN 进行训练
if __name__ == "__main__":

    # 网络拓扑
    book1 = xlrd.open_workbook(r'.\topology.xlsx')
    sheet3 = book1.sheets()[0]
    source_topo = sheet3.col_values(10)[23:59]
    target_topo = sheet3.col_values(11)[23:59]

    #状态空间
    link_weights_set = []
    link_loads_set = []
    backup_demand = []
    forpath_link_back_set = []

    '''
    文件循环
    '''
    folder_path = r".\xlsx"
    files = os.listdir(folder_path)

    for i in range(len(files) - 1):

        file1_path = os.path.join(folder_path, files[i])
        file2_path = os.path.join(folder_path, files[i + 1])
        print("i", i)

        # 读取xlsx文件
        df1 = xlrd.open_workbook(file1_path)
        df2 = xlrd.open_workbook(file2_path)

        sheet1 = df1.sheets()[0]
        sheet2 = df2.sheets()[0]

        '''
        计算介数中心性的拓扑熵
        '''

        nrows = sheet2.nrows
        ncols = sheet2.ncols
        node_topo = sheet2.col_values(6)[1:23]
        temp = target_topo.copy()
        target_topo.append(source_topo)
        source_topo.append(target_topo)
        source_number = number(source_topo, node_topo)
        target_number = number(target_topo, node_topo)
        link_topo = list(zip(source_number, target_number))
        temp = source_number[0:36]

        for i in target_number:
            source_number.append(i)
        for i in temp:
            target_number.append(i)

        link = list(zip(source_number, target_number))
        node_number = []
        j = 0
        for i in range(len(node_topo)):
            j = j + 1
            node_number.append(j)

        edge_links = []
        j = 0
        for i in range(len(link_topo)):
            j = j + 1
            edge_links.append(link_topo[i])

        # 构建网络拓扑
        G = nx.Graph()
        G.add_nodes_from(node_number)
        G.add_edges_from(edge_links)
        G_array = np.array(nx.adjacency_matrix(G).todense())

        # 计算节点介数中心性
        betweenness_centrality = []
        betweenness_centrality = topNBetweeness(G)
        To = topo_standard(betweenness_centrality)

        '''   
        计算信息熵与GNT部分
        '''

        target_allflow = []  # 各目的地节点流量
        allflow = 0  # 总流量值
        target1 = 0
        allflow2 = 0

        source = sheet1.col_values(11)[23:]
        target = sheet1.col_values(12)[23:]
        target_flow = sheet1.col_values(13)[23:]

        source2 = sheet2.col_values(11)[23:]
        target2 = sheet2.col_values(12)[23:]
        target_flow2 = sheet2.col_values(13)[23:]

        source1_number = number(source, node_topo)
        target1_number = number(target, node_topo)
        source2_number = number(source2, node_topo)
        target2_number = number(target2, node_topo)

        targets_flow, target_allflow = target_flow_addition(source1_number, target1_number, target_flow)
        allflow = allflow_sum(target_allflow)
        targets_flow2, target_allflow2 = target_flow_addition(source2_number, target2_number, target_flow2)
        allflow2 = allflow_sum(target_allflow2)

        target_allflow_standard = standard(target_allflow)
        target_allflow2_standard = standard(target_allflow2)

        Hi, H = Hi_pi(targets_flow, target_allflow)
        Hi2, H2 = Hi_pi(targets_flow2, target_allflow2)  # Hi为信息熵
        h, p, GNT_U = GNT_computer(H, H2, target_allflow_standard, target_allflow2_standard)
        GNT = standard(GNT_U)

        '''
        计算节点重要度以及边赋权值
        '''
        flow_entropy = H2.copy()
        topo_entropy = To.copy()
        flow_entropy = standard(flow_entropy)
        flow_entropy.sort(key=lambda ele: ele[0], reverse=False)
        topo_entropy.sort(key=lambda ele: ele[0], reverse=False)
        GNT.sort(key=lambda ele: ele[0], reverse=False)
        node_degree = degree_sum(flow_entropy, GNT, topo_entropy)

        link_degree = []
        link_degree = link_class(link, node_degree)

        l = []
        degree = []
        for key in link_degree:
            l.append(key[0])
            degree.append(key[1])
        link_degree_dict = dict(zip(l, degree))

        m1 = [[0 for i in range(22)] for i in range(22)]
        for key, value in link_degree_dict.items():
            i = key[0] - 1  # 不相连 0 一类相连 1 二类相连 2 三类相连 3
            j = key[1] - 1
            m1[i][j] = 1
            m1[j][i] = 1

        m2 = [[0 for i in range(22)] for i in range(22)]
        for key, value in link_degree_dict.items():
            i = key[0] - 1  # 不相连 0 一类相连 1 二类相连 2 三类相连 3
            j = key[1] - 1
            if value == 1:
                m2[i][j] = 1
                m2[j][i] = 1
            elif value == 2:
                m2[i][j] = 2
                m2[j][i] = 2
            elif value == 3:
                m2[i][j] = 3
                m2[j][i] = 3

        link_weights_set.append(m2)

        od = list(zip(source2_number, target2_number))
        od_flow = list(zip(od, target_flow2))

        network1 = m1

        flow = []  # 链路流量
        for key in link:
            flow.append(0)
        link_flow = dict(zip(link, flow))

        ratio = []
        for key in link:
            ratio.append(0)
        link_load = dict(zip(link, ratio))
        forwarding_path_all = []
        link_bw = 1500

        for key in od_flow[0:30]:
            forwarding_path = find_disjoint_shortest_paths(network1, key[0][0], key[0][1])
            forwarding_path_len = len(forwarding_path)

            for key5 in forwarding_path:
                for key2 in range(len(key5[0]) - 1):
                    ac_link1 = key5[0][key2]
                    ac_link2 = key5[0][key2 + 1]
                    k = 0
                    # 利用率检测
                    for key6, value3 in link_load.items():
                        if (key6[0] == ac_link1 and key6[1] == ac_link2):
                            if value3 > 0.7:
                                k = 1
                        elif (key6[0] == ac_link2 and key6[1] == ac_link1):
                            if value3 > 0.7:
                                k = 1
                    if k == 1:
                        break
                    for key4, value2 in link_flow.items():
                        if (key4[0] == ac_link1 and key4[1] == ac_link2):
                            key3 = (ac_link1, ac_link2)
                            link_flow[key3] = link_flow[key3] + key[1]
                            link_load[key3] = link_flow[key3] / link_bw
                        elif (key4[0] == ac_link2 and key4[1] == ac_link1):
                            key3 = (ac_link2, ac_link1)
                            link_flow[key3] = link_flow[key3] + key[1]
                            link_load[key3] = link_flow[key3] / link_bw
            forwarding_path_all.append(forwarding_path)


        # link_loads_set
        m3 = [[0 for i in range(22)] for i in range(22)]
        for key, value in link_load.items():
            i = key[0] - 1
            j = key[1] - 1
            m3[i][j] = value
            m3[j][i] = value
        link_loads_set.append(m3)

        #demand
        m4 = [[0 for i in range(22)] for i in range(22)]
        for key1, value1 in link_load.items():
            for key2, value2 in link_degree_dict:
                if (key1 == key2) and (value2 == 3):
                    i = key1[0] - 1
                    j = key1[1] - 1
                    m4[i][j] = value1
                    m4[j][i] = value1
        backup_demand.append(m4)

        # 链路备份
        #print("forwarding_path_all", forwarding_path_all)
        #print("link_flow", link_flow)
        a_obj = 0.4
        b_obj = 0.6
        back_path_set = []
        network2 = m2
        select_back_path_all = []
        forpath_link = []
        forpath_link_back = []

        for key1 in forwarding_path_all:
            for key2 in key1:
                for key15 in range(len(key2[0]) - 1):
                    for key14, value2 in link_flow.items():
                        if (key14[0] == key2[0][key15] and key14[1] == key2[0][key15 + 1]):
                            re_link = (key2[0][key15], key2[0][key15 + 1])  # 需要备份的链路
                            k = link_degree_dict[re_link]
                        elif (key14[1] == key2[0][key15] and key14[0] == key2[0][key15 + 1]):
                             re_link = (key2[0][key15], key2[0][key15 + 1])  # 需要备份的链路
                             k = link_degree_dict[re_link]

                    select_back_path = []
                    path_w = []

                    if k == 3:
                        print("key2[0]", key2[0])
                        print("re_link", re_link)
                        w = network2[re_link[0] - 1][re_link[1] - 1]
                        network2[re_link[0] - 1][re_link[1] - 1] = 0  # 将故障节点中图中去除
                        network2[re_link[1] - 1][re_link[0] - 1] = 0
                        back_path_set = find_disjoint_shortest_paths(network2, re_link[0], re_link[1])
                        print("back_path_set", back_path_set)
                        # 回环检测与流量控制
                        for key3 in back_path_set:
                            k1 = 1
                            k2 = 1
                            for key4 in range(len(key3[0]) - 1):
                                ac_link1 = key3[0][key4]
                                ac_link2 = key3[0][key4 + 1]
                                for key8 in key2[0][key15 + 2:]:
                                    if key8 == key3[0][key4 + 1]:  # 避免回环链路产生
                                        k1 = 0
                                # MLU控制，避免选择有高负载的链路的备份路径
                                for key13, value2 in link_load.items():
                                    if (key13[0] == ac_link1 and key13[1] == ac_link2):
                                        if value2 >= 0.7:
                                            k2 = 0
                                    elif (key13[0] == ac_link2 and key13[1] == ac_link1):
                                        if value2 >= 0.7:
                                            k2 = 0
                            if k1 == 1 and k2 == 1:
                                c = a_obj * key3[1] + b_obj * (len(key3[0]) - 1)
                                path_w.append(c)
                                select_back_path.append(key3)  # 存入可用的备份路径
                                # print("len(select_back_path1)", len(select_back_path))
                                # 若备份路径数量不足，计算故障尚有节点到目的节点的备份路径
                                if len(select_back_path) <= 1:
                                    k1 = 1
                                    k2 = 1
                                    back_path_set = find_disjoint_shortest_paths(network2, re_link[0],
                                                                                 key2[0][len(key2[0]) - 1])
                                    # print("len(back_path_set1)", len(back_path_set))
                                    # MLU控制，避免选择有高负载的链路的备份路径
                                    for key9 in back_path_set:
                                        for key10 in range(len(key9[0]) - 1):
                                            ac_link1 = key9[0][key10]
                                            ac_link2 = key9[0][key10 + 1]
                                            # 避免备份路径中包含故障节点
                                            if (ac_link1 == key14[0] and ac_link2 == key14[1]) or (
                                                    ac_link2 == key14[0] and ac_link1 == key14[1]):
                                                k1 = 0
                                            for key13, value2 in link_load.items():
                                                if (key13[0] == ac_link1 and key13[1] == ac_link2):
                                                    if value2 >= 0.7:
                                                        k2 = 0
                                                elif (key13[0] == ac_link2 and key13[1] == ac_link1):
                                                    if value2 >= 0.7:
                                                        k2 = 0
                                        if k1 == 1 and k2 == 1:
                                            c = a_obj * key9[1] + b_obj * (len(key9[0]) - 1)
                                            path_w.append(c)
                                            select_back_path.append(key9)
                        if len(select_back_path) <= 3:
                            select_back_path_all.append(select_back_path)
                        else:
                            path_wed = sorted(path_w)
                            select_back_path_sort = []
                            for key16 in range(len(select_back_path) - 1):
                                if path_w[key16] <= path_wed[2]:
                                    select_back_path_sort.append(select_back_path[key16])
                            select_back_path_all.append(select_back_path_sort)
                        forpath_link.append((key2[0], re_link))
                        network2[re_link[0] - 1][re_link[1] - 1] = w  # 将故障节点中图中去除
                        network2[re_link[1] - 1][re_link[0] - 1] = w

        forpath_link_back = list(zip(forpath_link, select_back_path_all))
        #print("forpath_link_back", forpath_link_back)
        forpath_link_back_set.append(forpath_link_back)
    print("len(link_weights_set)", len(link_weights_set))
    print("link_loads_set", len(link_loads_set))
    print("backup_demand", len(backup_demand))
    print("len(forpath_link_back_set)", len(forpath_link_back_set))


    '''
    DQN训练
    '''

    env = Environment(link_weights_set, link_loads_set, backup_demand)
    state_dim = 22 * 22 * 3  # 这里的状态维度为链路权重矩阵、链路负载矩阵、备份路径需求矩阵的组合
    action_dim = 4  # 动作空间为 [1, 2, 3, 0]
    dqn_agent = DQN(state_dim, action_dim)

    # 训练 DQN 智能体
    link_bw = 1500
    T = len(link_weights_set)
    num_episodes = 100
    for episode in range(num_episodes):
        episode_reward = 0
        for time in range(T - 1):
            state = env.get_state(time)
            for forpath_link_back in forpath_link_back_set[time]:
                link = forpath_link_back[0][1]
                backup_path = forpath_link_back[1]
                #link_load = state[1]
                #backup_demand = state[2]
                link_load = state[1 * 22 * 22: 2 * 22 * 22]
                link_load = np.array(link_load).reshape(22, 22)
                backup_demand = state[2 * 22 * 22:]
                backup_demand = np.array(backup_demand).reshape(22, 22)
                action = dqn_agent.select_action(state)
                reward = env.execute_action(action, link, backup_path, link_load, backup_demand, link_bw)
                next_state = env.get_state(time)
                dqn_agent.train(state, action, reward, next_state)
                episode_reward += reward
                state = next_state
        dqn_agent.update_epsilon()
        dqn_agent.update_target_network()
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")
