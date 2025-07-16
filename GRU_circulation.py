
import pandas as pd
import xlrd
import os
import numpy as np
import openpyxl
from openpyxl import load_workbook
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
import random
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense,GRU
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape

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
    # print("betweenness_centrality: ", score)
    # print(type(score))
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
    # target_index = []
    target_sets = []
    tar = []
    target_sets = list(set(targets))  # 源地址分类转化成列表形式
    targets_flow = list(zip(targets, target_flow))
    #print(len(set(sources)))  # 部分节点无流量通过，因此事实节点少于50
    #print(len(set(targets)))
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

    flow = 0
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
                # p = (key[1] + 1)*math.log(key[1] + 1)
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


def topo1_entropy(betweenness_centrality):
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
    a = 0.0  # 指标的权重
    b = 0.3
    c = 0.7
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
            p = a * flow[k][1] + b * topo1[k][1] + c * GNT1[k][1]
        tar.append(topo1[k][0])
        degree.append(p)

    node_degree = dict(zip(tar, degree))
    #print("node_degree")
    #print(node_degree)

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

    node_degree1 = list(zip(tar1, degree1))
    return node_degree1


def link_class(edge_links, node_degree):
    k = 0
    i = 0
    ol = []
    ol1 = []
    link_degree = []
    link_sort1 = edge_links

    for k in range(int(len(edge_links))):
        i = i + 1
        for j in range(i, int(len(edge_links))):
            if link_sort1[k][0] == link_sort1[j][1] and link_sort1[k][1] == link_sort1[j][0]:
                l = [link_sort1[k], link_sort1[j]]
                ol.append(l)
                ol1.append(link_sort1[k])

    k = 0
    weight = 0
    #print("edge_links")
    #print(ol1)
    for node in edge_links:
        i = node[0]
        j = node[1]
        for key in node_degree:
            if i == key[0]:
                k1 = key[1]
        for key in node_degree:
            if j == key[0]:
                k2 = key[1]

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

def standard(target_allflow):  # 标准化公式
    min_flow = target_allflow[0][1]
    max_flow = target_allflow[0][1]
    tar = []
    target_allflow_standard = []

    for key in target_allflow:
        if key[1] < min_flow:
            min_flow = key[1]
        if key[1] > max_flow:
            max_flow = key[1]
    for key in target_allflow:
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
    i = 0.2
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
    # print(path)
    # print(midpath)
    # print(dis)
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


# 构建加权无向图
def build_graph(matrix):
    similarity_matrix = 1 / (1 + pairwise_distances(matrix, metric='euclidean'))
    return similarity_matrix

# 对图进行谱聚类
def spectral_clustering(matrix, k):
    adjacency_matrix = build_graph(matrix)
    spectral = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
    spectral.fit(adjacency_matrix)
    labels = spectral.labels_
    return labels
def find_disjoint_shortest_paths(weighted_adj_matrix, source, target, num_paths=10):
    G = nx.Graph()
    num_nodes = len(weighted_adj_matrix)

    # 添加节点和边到图中
    for i in range(num_nodes):
        for j in range(num_nodes):
            if weighted_adj_matrix[i][j] != 0:
                G.add_edge(i, j, weight=weighted_adj_matrix[i][j])

    # 找到源节点到目标节点的所有最短路径
    all_shortest_paths = list(k_shortest_path(weighted_adj_matrix, source, target, 10))

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

if __name__ == "__main__":

    # 网络拓扑
    book1 = xlrd.open_workbook(r'.\topology.xlsx')
    sheet3 = book1.sheets()[0]
    source_topo = sheet3.col_values(10)[23:59]
    target_topo = sheet3.col_values(11)[23:59]

    '''
    文件循环
    '''
    folder_path = r".\xlsx"
    files = os.listdir(folder_path)

    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    X3 = []
    Y3 = []

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
        To = topo1_entropy(betweenness_centrality)


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

        m = [[0 for i in range(50)] for i in range(50)]
        for key, value in link_degree_dict.items():
            i = key[0] - 1  # 不相连 0 一类相连 1 二类相连 2 三类相连 3
            j = key[1] - 1
            m[i][j] = 1
            m[j][i] = 1

        od = list(zip(source2_number, target2_number))
        od_flow = list(zip(od, target_flow2))

        network = m

        flow = []  # 链路流量
        for key in link:
            flow.append(0)
        link_flow = dict(zip(link, flow))

        ratio = []
        for key in link:
            ratio.append(0)
        link_load = dict(zip(link, ratio))

        link_bw = 1500

        for key in od_flow:
            forwarding_path = find_disjoint_shortest_paths(network, key[0][0], key[0][1])
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
                    if k == 0:
                        c = 0
                        for key4, value2 in link_flow.items():
                            if (key4[0] == ac_link1 and key4[1] == ac_link2):
                                key3 = (ac_link1, ac_link2)
                                c = (link_flow[key3] + key[1]) / link_bw
                                if c > 0.7:
                                    break
                                else:
                                    link_flow[key3] = link_flow[key3] + key[1]
                                    link_load[key3] = link_flow[key3] / link_bw
                            elif (key4[0] == ac_link2 and key4[1] == ac_link1):
                                 key3 = (ac_link2, ac_link1)
                                 c = (link_flow[key3] + key[1]) / link_bw
                                 if c > 0.7:
                                     break
                                 else:
                                    link_flow[key3] = link_flow[key3] + key[1]
                                    link_load[key3] = link_flow[key3] / link_bw


        #print("link_flow", link_flow)
        #print("link_load", link_load)
        m1_flow = [[0 for i in range(22)] for i in range(22)]
        m2_flow = [[0 for i in range(22)] for i in range(22)]
        m3_flow = [[0 for i in range(22)] for i in range(22)]
        for key, value in link_flow.items():
           i = key[0] - 1
           j = key[1] - 1
           if link_degree_dict[key] == 1:
              m1_flow[i][j] = link_load[key]
              m1_flow[j][i] = link_load[key]
           elif link_degree_dict[key] == 2:
                m2_flow[i][j] = link_load[key]
                m2_flow[j][i] = link_load[key]
           elif link_degree_dict[key] == 3:
                m3_flow[i][j] = link_load[key]
                m3_flow[j][i] = link_load[key]

        X1.append(m1_flow)
        X2.append(m2_flow)
        X3.append(m3_flow)


print("len(X1)", len(X1))
print("len(X2)", len(X2))
print("len(X3)", len(X3))

# 生成随机示例数据
X_train = X1
Y_train = Y1

# 定义函数以准备数据
def prepare_data(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


    # 构建模型
def build_model(seq_length, matrix_shape):
    model = Sequential([
        GRU(256, input_shape=(seq_length, np.prod(matrix_shape))),
        Dense(np.prod(matrix_shape)),  # 输出层的维度与目标值的形状相同
        Reshape((matrix_shape[0], matrix_shape[1]))  # 将输出形状调整为 (22, 22)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


    # 创建模拟数据
    # 假设每个链路负载矩阵是一个22x22的矩阵
    # 假设有72个链路负载矩阵
num_samples = 72
matrix_shape = (22, 22)
seq_length = 12

# 准备数据
X, y = prepare_data(X1, seq_length)
print("X.shape", X.shape)
print("y.shape", y.shape)

# 构建和训练模型
model = build_model(seq_length, matrix_shape)

# 将输入数据调整为模型所期望的维度
X = X.reshape(-1, seq_length, np.prod(matrix_shape))
print("输入数据形状：", X.shape)

model.fit(X, y, epochs=700, batch_size=1)

# 进行验证
X_test, Y_test = prepare_data(X1[56:], seq_length)
X_test = X_test.reshape(-1, seq_length, np.prod(matrix_shape))
print("输入数据形状：", X_test.shape)

predicted = model.predict(X_test)
print("输入数据形状：", predicted.shape)

# 将实际值和预测值展平为一维数组
y_val_flat = Y_test.flatten()
y_val_pred_flat = predicted.flatten()

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_val_flat, y_val_pred_flat))

# 计算 MAE
mae = mean_absolute_error(y_val_flat, y_val_pred_flat)

# 计算 R²
r2 = r2_score(y_val_flat, y_val_pred_flat)

print("Validation RMSE:", rmse)
print("Validation MAE:", mae)
print("Validation R²:", r2)

# 进行预测
# 假设我们有最后6个链路负载矩阵，要预测下一个
X1 = np.array(X1)
last_matrices = X1[-seq_length:].reshape(1, seq_length, -1)


predicted_matrix = model.predict(last_matrices)

#print("Predicted Matrix:")
#print(predicted_matrix)


