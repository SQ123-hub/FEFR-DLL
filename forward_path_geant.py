# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:33:09 2022

@author: Lenovo2
"""

import pandas as pd
import xlrd
# import xlwt
import numpy as np
import openpyxl
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
import random


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


def GNT(h1, h2, p1, p2):  # 计算流量差值

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
    a = 0.3
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

    q = 0.09  # 节点分配比例
    p = 0.6
    o = 0.31
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


def find_disjoint_shortest_paths(weighted_adj_matrix, source, target, num_paths=5):
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

if __name__ == "__main__":

    '''
    计算介数中心性部分
    '''
    book = xlrd.open_workbook(r'.\xlsx\01.xlsx')
    book4 = xlrd.open_workbook(r'.\topology.xlsx')

    sheet1 = book.sheets()[0]
    sheet2 = book4.sheets()[0]

    node_topo = []
    source_topo = []
    target_topo = []

    node_topo = sheet1.col_values(6)[1:23]
    source_topo = sheet2.col_values(10)[23:59]
    target_topo = sheet2.col_values(11)[23:59]

    source_number = number(source_topo, node_topo)
    target_number = number(target_topo, node_topo)

    link = list(zip(source_number, target_number))
    link_sort = list(set(link))
    link_sort_len = len(set(link_sort))

    node_number = []
    j = 0
    for i in range(len(node_topo)):
        j = j + 1
        node_number.append(j)

    edge_links = []
    j = 0
    for i in range(len(link)):
        j = j + 1
        edge_links.append(link[i])


    # 构建网络拓扑

    G = nx.Graph()
    G.add_nodes_from(node_number)
    G.add_edges_from(edge_links)
    G_array = np.array(nx.adjacency_matrix(G).todense())

    '''
    计算介数中心性的拓扑熵
    '''

    betweenness_centrality = []
    betweenness_centrality = topNBetweeness(G)
    To = topo_standard(betweenness_centrality)


    '''   
    计算信息熵与GNT部分
    '''

    book5 = xlrd.open_workbook(r'.\xlsx\70.xlsx')
    book6 = xlrd.open_workbook(r'.\xlsx\72.xlsx')

    sheet3 = book5.sheets()[0]
    sheet4 = book6.sheets()[0]

    nrows = sheet3.nrows
    ncols = sheet3.ncols

    target_allflow = []  # 各目的地节点流量
    allflow = 0  # 总流量值
    target1 = 0
    allflow2 = 0

    source = sheet3.col_values(11)[23:]
    target = sheet3.col_values(12)[23:]
    target_flow = sheet3.col_values(13)[23:]

    source2 = sheet4.col_values(11)[23:]
    target2 = sheet4.col_values(12)[23:]
    target_flow2 = sheet4.col_values(13)[23:]

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

    #计算机GNT
    Hi, H = Hi_pi(targets_flow, target_allflow)
    Hi2, H2 = Hi_pi(targets_flow2, target_allflow2)  # Hi为信息熵
    h, p, GNT = GNT(H, H2, target_allflow_standard, target_allflow2_standard)
    GNT = standard(GNT)

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
    link_weight = link_sum(link, GNT, topo_entropy)
    link_degree = link_class(link, node_degree)  #链路分类

    print("link_degree", link_degree)

    G1 = nx.Graph()
    G1.add_nodes_from(node_number)
    for key in link_degree:
        G1.add_edge(key[0][0], key[0][1], weight=key[1])

    fig = plt.figure(figsize=(16, 9))
    pos = nx.spring_layout(G)

    # 重新获取权重序列
    weights = nx.get_edge_attributes(G1, "weight")

    # 发送流量

    # 计算流量转发路径

    link1 = []  # 筛选重复链路
    for key in link:
        for key1, value1 in weights.items():
            if key1[0] == key[0] and key1[1] == key[1]:
                link1.append(key)

    print("len(link1)", len(link1))
    m1 = [[0 for i in range(50)] for i in range(50)]
    for key in link_degree:
        i = key[0][0] - 1  # 不相连 0 一类相连 1 二类相连 2 三类相连 3
        j = key[0][1] - 1
        m1[i][j] = 1
        m1[j][i] = 1

    m2 = [[0 for i in range(50)] for i in range(50)]
    for key in link_weight:
        i = key[0][0] - 1  # 不相连 0 一类相连 1 二类相连 2 三类相连 3
        j = key[0][1] - 1
        m2[i][j] = round(key[1], 3)
        m2[j][i] = round(key[1], 3)

    i = 0
    k = 0
    od_flow_packet = []
    forwarding_path_all = []
    flow = []  # 链路流量
    for key in link1:
        flow.append(0)
    link_flow = dict(zip(link1, flow))

    random_numbers = [3, 7, 13]
    print("random_numbers", random_numbers)
    od_flow_packet = [((18, 6), 32), ((4, 15), 199), ((1, 10), 172), ((21, 20), 176), ((18, 16), 95), ((9, 1), 228), ((16, 14), 141), ((15, 20), 132), ((6, 13), 179), ((4, 8), 77), ((3, 9), 13), ((13, 14), 266), ((16, 9), 245), ((20, 17), 209), ((17, 19), 47), ((19, 14), 239), ((4, 2), 114), ((5, 8), 55), ((5, 7), 208), ((9, 1), 228), ((6, 2), 135), ((2, 17), 110), ((2, 14), 15), ((8, 15), 290), ((10, 4), 111), ((4, 2), 114), ((15, 13), 272), ((16, 18), 236), ((3, 17), 70), ((10, 1), 49), ((16, 8), 234), ((15, 14), 159), ((22, 16), 188), ((3, 16), 159), ((3, 20), 145), ((1, 20), 295), ((19, 20), 117), ((5, 11), 49), ((13, 19), 294), ((18, 15), 38), ((17, 13), 296), ((8, 16), 206), ((9, 15), 1), ((12, 19), 147), ((1, 6), 42), ((21, 16), 169), ((17, 12), 41), ((7, 1), 33), ((13, 16), 46), ((18, 21), 120), ((11, 6), 45), ((8, 12), 157), ((5, 12), 219), ((10, 6), 172), ((10, 5), 32), ((6, 8), 280), ((6, 20), 129), ((5, 20), 204), ((13, 5), 271), ((11, 9), 137), ((17, 19), 47), ((3, 5), 140), ((16, 13), 74), ((14, 17), 217), ((3, 10), 70), ((20, 13), 275), ((13, 16), 46), ((9, 13), 5), ((9, 1), 228), ((17, 6), 268), ((7, 9), 211), ((10, 17), 109), ((11, 16), 42), ((19, 2), 158), ((6, 2), 135), ((8, 20), 283), ((22, 21), 98), ((1, 13), 262), ((20, 17), 209), ((4, 18), 85), ((4, 10), 234), ((3, 1), 178), ((3, 20), 145), ((8, 16), 206), ((21, 3), 190), ((5, 17), 288), ((12, 4), 70), ((20, 21), 211), ((9, 4), 151), ((12, 3), 94), ((14, 8), 19), ((4, 10), 234), ((18, 3), 21), ((17, 14), 237), ((17, 5), 53), ((10, 21), 245), ((1, 6), 42), ((16, 10), 249), ((9, 8), 35), ((5, 4), 110)]
    print("od_flow_packet", od_flow_packet)
    od_flow_packet =  od_flow_packet[0:5]
    link_bw = 1500 #预设链路带宽
    link_load = [] #链路负载集合
    ratio = []
    network = m2
    for key in link:
        ratio.append(0)
    link_load = dict(zip(link, ratio))
    network = m1

    for key in od_flow_packet:
            forwarding_path = find_disjoint_shortest_paths(network, key[0][0], key[0][1])
            forwarding_path_len = len(forwarding_path)

            for key5 in forwarding_path:
                for key2 in range(len(key5[0]) - 1):
                    ac_link1 = key5[0][key2]
                    ac_link2 = key5[0][key2 + 1]
                    k = 0
                    #利用率检测
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


    print("len(forwarding_path_all)", len(forwarding_path_all))
    print("link_load",  link_load)

    # LSNC
    l = []
    degree = []
    for key in link_degree:
        degree.append(key[1])
    link_degree_dict = dict(zip(link1, degree))
    print("len(link_degree_dict)", len(link_degree_dict))

    # 工作路径流表项统计
    flow_table = []
    for key1 in forwarding_path_all:
        for key2 in range(len(key1[0]) - 1):
            flow_entry = (key1[0][key2], key1[0][key2 + 1])
            flow_table.append(flow_entry)

    unique_flow_table = []
    [unique_flow_table.append(x) for x in flow_table if x not in unique_flow_table]
    print("unique_flow_table", len(unique_flow_table))

    # 统计经过链路的工作路径
    account = []
    for i in link1:
        account.append(0)
    link_work = dict(zip(link1, account))

    for key1 in forwarding_path_all:
        for key2 in range(len(key1[0]) - 1):
            ac_link1 = key1[0][key2]
            ac_link2 = key1[0][key2 + 1]
            for key3, value3 in link_work.items():
                if (key3[0] == ac_link1 and key3[1] == ac_link2) or (key3[0] == ac_link2 and key3[1] == ac_link1):
                    link_work[key3] = link_work[key3] + 1

    m3 = [[0 for i in range(50)] for i in range(50)]
    for key, value in link_load.items():
        i = key[0] - 1  # 不相连 0 一类相连 1 二类相连 2 三类相连 3
        j = key[1] - 1
        m3[i][j] = value
        m3[j][i] = value
    network1 = m3

    linkset = []
    back_up_allset = []
    k = 0
    a_obj = 0.4
    b_obj = 0.6
    print(forwarding_path_all)
    back_path_set = []
    pa1 = []
    pa2 = []
    # 链路备份
    for key1 in forwarding_path_all:
        for key2 in range(len(key1[0]) - 1):
            for key14, value2 in link_flow.items():
                if (key14[0] == key1[0][key2] and key14[1] == key1[0][key2 + 1]):
                     re_link = (key14[0], key14[1])  # 需要备份的链路
                     k = link_degree_dict[re_link]
                elif (key14[1] == key1[0][key2] and key14[0] == key1[0][key2 + 1]):
                     re_link = (key14[0], key14[1])  # 需要备份的链路
                     k = link_degree_dict[re_link]
            if k == 3:
                back_path_set = find_disjoint_shortest_paths(network, key1[0][key2], key1[0][key2 + 1])
            elif k == 2:
                back_path_set = k_shortest_path(network, key1[0][key2], key1[0][key2 + 1], 5)
            select_back_path = []
            if k == 3 and len(back_path_set) <= 1:
                pb = k_shortest_path(network, key1[0][key2], key1[0][key2 + 1], 6)[1:]
                for key15 in pb:
                    back_path_set.append(key15)
            c = 100
            c2 = 100
            max_ratio = 0
            for key3 in back_path_set[1:]:
                k1 = 1
                k2 = 1
                for key4 in range(len(key3[0]) - 1):
                    ac_link1 = key3[0][key4]
                    ac_link2 = key3[0][key4 + 1]
                    for key8 in key1[0][key2 + 2:]:
                        if key8 == key3[0][key4 + 1]:  #避免回环链路产生
                            k1 = 0
                    #MLU控制，避免选择有高负载的链路的备份路径
                    for key13, value2 in link_load.items():
                        if (key13[0] == ac_link1 and key13[1] == ac_link2):
                            if value2 >= 0.6:
                                k2 = 0
                        elif (key13[0] == ac_link2 and key13[1] == ac_link1):
                             if value2 >= 0.6:
                                k2 = 0


                c1 = a_obj * key3[1] + b_obj * (len(key3[0]) - 1)

                if (c1 < c or  c1 < c2 ) and k1 == 1 and k2 == 2:  # 选择目标函数最小的备份路径
                    if c1 < c and c2 <= c:
                        c = c1
                        pa1 = key3
                    elif c1 < c2 and c <= c2:
                         c2 = c1
                         pa2 = key3

                      # 按c从大到小排序的备份路径
            select_back_path.append(pa1)
            select_back_path.append(pa2)
            #print("len(select_back_path)", select_back_path)
            back_path_all = []
            if k == 3:  # 提供2条备份路径

                if len(select_back_path) == 0:  #select_back_path中的备份路径均会产生回路
                    #print("len(select_back_path)", len(select_back_path))
                    #print("key1[0][0]", key1[0][key2])
                    back_path1 = k_shortest_path(network1, key1[0][0], key1[0][len(key1[0]) - 1], 5)[2]
                    back_path2 = k_shortest_path(network1, key1[0][key2], key1[0][key2 + 2], 5)[3]
                    back_path_all.append(back_path1)
                    back_path_all.append(back_path2)
                    #print("back_path_all", back_path_all)
                elif len(select_back_path) == 1:
                     #print("len(select_back_path)", len(select_back_path))
                     back_path1 = select_back_path[len(select_back_path) - 1]
                     back_path2 = k_shortest_path(network1, key1[0][0], key1[0][len(key1[0]) - 1], 5)[3]
                     back_path_all.append(back_path1)
                     back_path_all.append(back_path2)
                     #print("back_path_all", back_path_all)
                else:
                     #print("len(select_back_path)", len(select_back_path))
                     back_path1 = select_back_path[len(select_back_path) - 1]
                     back_path2 = select_back_path[len(select_back_path) - 2]
                     back_path_all.append(back_path1)
                     back_path_all.append(back_path2)
                     #print("back_path_all", back_path_all)

                #print("back_path_all", back_path_all)

                for key6 in range(len(back_path1[0]) - 1):
                    key7 = (back_path1[0][key6], back_path1[0][key6 + 1])
                    flow_table.append(key7)
                for key6 in range(len(back_path2[0]) - 1):
                    key7 = (back_path2[0][key6], back_path2[0][key6 + 1])
                    flow_table.append(key7)
                linkset.append(re_link)
                back_up_allset.append(back_path_all)

            elif k == 2:  # 提供1条备份路径
                if len(select_back_path) == 0:
                    back_path1 = k_shortest_path(network1, key1[0][0], key1[0][len(key1[0]) - 1], 5)[1]
                elif len(select_back_path) > 0:
                    back_path1 = select_back_path[len(select_back_path) - 1]
                for key6 in range(len(back_path1[0]) - 1):
                    key7 = (back_path1[0][key6], back_path1[0][key6 + 1])
                    flow_table.append(key7)
                linkset.append(re_link)
                back_up_allset.append(back_path1)

    print("len(resulte_flow_table)")
    #print("back_up_allset", back_up_allset)
    allset = list(zip(linkset, back_up_allset))
    set = []
    [set.append(x) for x in allset if x not in set]

    resulte_flow_table = []
    [resulte_flow_table.append(x) for x in flow_table if x not in resulte_flow_table]
    print("len(resulte_flow_table)", len(resulte_flow_table))

    link_select = []
    for key1, value1 in link_flow.items():
        link_select.append(key1)

    link_failure = []
    for key in random_numbers:
        link_failure.append(link_select[key])
    max_ratio = []

    print("link_flow", link_flow)
    flow_docker = []
    for key5, value in link_flow.items():
        flow_docker.append(value)

    result_flow = []
    for key1 in link_failure:
        flow_docker1 = []
        link_flow1 = link_flow.copy()
        for key2 in set:
            if key1 == key2[0]:
                for key4 in range(len(key2[1][0]) - 1):
                    ac_link1 = key2[1][0][key4]
                    ac_link2 = key2[1][0][key4 + 1]
                    for key8, value2 in link_flow.items():
                        if (key8[0] == ac_link1 and key8[1] == ac_link2):
                            key6 = (key8[0], key8[1])
                            link_flow1[key6] = link_flow1[key6] + link_flow1[key1] / 2
                        elif (key8[0] == ac_link2 and key8[1] == ac_link1):
                            key6 = (key8[0], key8[1])
                            link_flow1[key6] = link_flow1[key6] + link_flow1[key1] / 2
                if len(key2[1]) == 4:
                    for key4 in range(len(key2[1][2]) - 1):
                        ac_link1 = key2[1][2][key4]
                        ac_link2 = key2[1][2][key4 + 1]
                        for key8, value2 in link_flow.items():
                            if (key8[0] == ac_link1 and key8[1] == ac_link2):
                                key6 = (key8[0], key8[1])
                                link_flow1[key6] = link_flow1[key6] + link_flow1[key1]
        for key5, value in link_flow1.items():
            flow_docker1.append(value)
        max_ratio.append(max(flow_docker1))
        result_flow.append(flow_docker1)



    print("max_ratio", max_ratio)
    print("result_flow", result_flow)
    l1 = len(resulte_flow_table) - len(unique_flow_table)
    print("flow_table", len(resulte_flow_table))
    print("len(flow_docker1)", len(flow_docker1))

    acc_distriutionall = []
    for key1 in result_flow:
        accont1 = 0
        accont2 = 0
        accont3 = 0
        accont4 = 0
        accont5 = 0
        acc_distriution = []
        for key2 in key1:
            if 0 <= key2 <= 200:
                accont1 = accont1 + 1
            elif 200 < key2 <= 500:
                accont2 = accont2 + 1
            elif 500 < key2 <= 700:
                accont3 = accont3 + 1
            elif 700 < key2 <= 1000:
                accont4 = accont4 + 1
            else:
                accont5 = accont5 + 1
        acc_distriution.append(accont1)
        acc_distriution.append(accont2)
        acc_distriution.append(accont3)
        acc_distriution.append(accont4)
        acc_distriution.append(accont5)
        acc_distriutionall.append(acc_distriution)

    print("acc_distriutionall", acc_distriutionall)
