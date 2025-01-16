import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import networkx as nx
from networkx.algorithms import community
import matplotlib.cm as cm
import matplotlib.dates as mdates
from sklearn.metrics import f1_score


def custom_sign(tensor):
    return torch.where(tensor <= 0, torch.tensor(-1, dtype=tensor.dtype), torch.tensor(1, dtype=tensor.dtype))

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
def just_calculate_infection_ratio(A, true_y):
    num_users, num_days = true_y.shape
    negative_infection_ratio_list = []
    positive_infection_ratio_list = []

    for day in range(num_days):  # 从第一天开始计算
        # 在每天开始时重置 already_infected_users
        already_infected_users = set()

        # 找到当天观点为负的用户
        negative_users = torch.where(true_y[:, day] == -1)[0]
        # 找到当天观点为正的用户
        positive_users = torch.where(true_y[:, day] == 1)[0]

        # 找到与这些观点为负的用户相连的所有其他用户
        possible_negative_infected_users = torch.where(torch.sum(A[negative_users, :], dim=0) > 0)[0]
        # 找到与这些观点为正的用户相连的所有其他用户
        possible_positive_infected_users = torch.where(torch.sum(A[positive_users, :], dim=0) > 0)[0]

        # 初始化 real_negative_infected_users 和 real_positive_infected_users
        real_negative_infected_users = torch.tensor([], dtype=torch.long)
        real_positive_infected_users = torch.tensor([], dtype=torch.long)

        # 在可能负感染者中找到那些在前一天观点为0在当天转变为-1的用户
        if day > 0:
            neutral_users_previous_day = torch.where(true_y[:, day - 1] != -1)[0]
            possible_negative_infected_users = torch.tensor(list(
                set(neutral_users_previous_day.tolist()).intersection(set(possible_negative_infected_users.tolist()))))
            real_negative_infected_users = torch.tensor(list(
                set(possible_negative_infected_users.tolist()).intersection(
                    set(torch.where(true_y[:, day] == -1)[0].tolist()))))
            real_negative_infected_users = torch.tensor(
                list(set(real_negative_infected_users.tolist()).difference(already_infected_users)))  # 排除已经被感染的用户
            already_infected_users.update(real_negative_infected_users.tolist())  # 更新已经被感染的用户列表

        # 在可能正感染者中找到那些在前一天观点为0在当天转变为1的用户
        if day > 0:
            neutral_users_previous_day = torch.where(true_y[:, day - 1] != 1)[0]
            possible_positive_infected_users = torch.tensor(list(
                set(neutral_users_previous_day.tolist()).intersection(set(possible_positive_infected_users.tolist()))))
            real_positive_infected_users = torch.tensor(list(
                set(possible_positive_infected_users.tolist()).intersection(
                    set(torch.where(true_y[:, day] == 1)[0].tolist()))))
            real_positive_infected_users = torch.tensor(
                list(set(real_positive_infected_users.tolist()).difference(already_infected_users)))  # 排除已经被感染的用户
            already_infected_users.update(real_positive_infected_users.tolist())  # 更新已经被感染的用户列表

        # 计算每天的‘真实负感染者’数量和‘真实正感染者’数量
        if (len(possible_negative_infected_users) != 0):
            negative_infection_ratio = len(real_negative_infected_users) / len(possible_negative_infected_users)
        else:
            negative_infection_ratio = 0

        if (len(possible_positive_infected_users) != 0):
            positive_infection_ratio = len(real_positive_infected_users) / len(possible_positive_infected_users)
        else:
            positive_infection_ratio = 0

        negative_infection_ratio_list.append(negative_infection_ratio)
        positive_infection_ratio_list.append(positive_infection_ratio)

    print("Negative Infection Ratio List:", negative_infection_ratio_list)
    print("Positive Infection Ratio List:", positive_infection_ratio_list)

    return negative_infection_ratio_list, positive_infection_ratio_list
def caculate_combine_Pbar(A,B,true_y,pred_y):
    t_neg, t_pos = just_calculate_infection_ratio(A, true_y)
    p_neg, p_pos = just_calculate_infection_ratio(B, pred_y)

    labels = ['NEG True', 'POS True', 'NEG Pred', 'POS Pred']
    plot_mean_attitude_changes_and_p_value(
        t_neg, t_pos, p_neg, p_pos, labels,
        xlabel='Group', ylabel='Average Number of People', color_scheme=2, show=True
    )
# Updated function to only plot the bar chart with the average values and P-value


import seaborn as sns


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def plot_mean_attitude_changes_and_p_value(t_neg, t_pos, p_neg, p_pos, labels, xlabel, ylabel, color_scheme, show):
    # 计算每个数组的平均值
    # 排除每个数组的第一个元素
    t_neg = np.array(t_neg[2:])
    t_pos = np.array(t_pos[2:])
    p_neg = np.array(p_neg[2:])
    p_pos = np.array(p_pos[2:])

    # 计算每个数组的平均值
    t_neg_mean = np.mean(t_neg)
    t_pos_mean = np.mean(t_pos)
    p_neg_mean = np.mean(p_neg)
    p_pos_mean = np.mean(p_pos)

    means = [t_neg_mean, t_pos_mean, p_neg_mean,p_pos_mean]

    # 使用元素级除法计算比率数组，并避免除以零
    ture_ratio = t_pos / (t_neg + 1e-10)
    pred_ratio = p_pos / (p_neg + 1e-10)

    # 计算两个比率数组之间的P值
    _, p_value = ttest_ind(ture_ratio, pred_ratio, equal_var=False)

    # 绘制柱状图
    colors = ['green', 'orange', 'green', 'orange'] if color_scheme == 2 else ['gray'] * 4
    plt.figure(figsize=(10, 6))  # 设置图的尺寸
    plt.bar(labels, means, color=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Average Number of People')

    # 在图上标注P值
    plt.text(0.5, max(means) * 0.95, f'P-value: {p_value:.2e}', horizontalalignment='center', fontsize=12)

    if show:
        plt.show()

    return ture_ratio, pred_ratio, p_value


def plot_attitude_changes_no_date(dates, attitude_changes_true, attitude_changes_pred, title = 'Attitude Changes Over Time', show = False, group_num = 0):
    line1, = plt.plot(dates, attitude_changes_true, marker='o', linestyle='-', color='b')
    line2, = plt.plot(dates, attitude_changes_pred, marker='o', linestyle='-', color='r')

    plt.xlabel('Time')
    plt.ylabel('Number of Users')
    plt.title(title)
    plt.grid(True)
    if group_num != 0:
        plt.text(0.05, 0.95, f"Group: {group_num}", transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white"))
    if show:
        plt.show()
    return [line1, line2]

def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


def draw_loss_and_save(pred_y, true_y, id_train, criterion, filename, group_num = 0):
    num_x = pred_y.shape[1]
    losses = []

    for i in range(num_x):
        pred = pred_y[:,i]
        true = true_y[:,i]
        mask = (true != 0.0)
        loss = (criterion(pred, true, reduction='none') * mask).sum().float()/mask.sum().float()
        losses.append(loss.item())

    plt.figure()

    # Extract the losses corresponding to id_train and plot them with a blue line and triangles
    train_losses = [losses[i] for i in id_train]
    plt.plot(id_train, train_losses, color='blue', marker='^', linestyle='-')

    # Extract the losses NOT corresponding to id_train and plot them with a gray line and squares
    non_train_ids = [i for i in range(num_x) if i not in id_train]
    non_train_losses = [losses[i] for i in non_train_ids]
    plt.plot(non_train_ids, non_train_losses, color='gray', marker='s', linestyle='--')

    # Scatter plot for all points; triangles for id_train and squares for others
    for i, loss in enumerate(losses):
        if i in id_train:
            plt.scatter(i, loss, marker='^', color='blue')  # Train points in blue
        else:
            plt.scatter(i, loss, marker='s', color='gray')  # Non-train points in gray

    if group_num != 0:
        plt.text(0.05, 0.95, f"Group: {group_num}", transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white"))
    
    plt.xlabel('id_train')
    plt.ylabel('loss')
    plt.ylim([0, 1.5])
    plt.title('loss')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def zipf_smoothing(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    A_prime = A + np.eye(A.shape[0])
    out_degree = np.array(A_prime.sum(1), dtype=np.float32)
    int_degree = np.array(A_prime.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A_prime @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_laplacian(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 * ( D - A ) * D^-1/2 = I - D^-1/2 * ( A ) * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float64)
    int_degree = np.array(A.sum(0), dtype=np.float64)
    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0.0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0.0))
    mx_operator = np.eye(A.shape[0]) - np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator


def normalized_adj(A):
    """
    Input A: np.ndarray
    :return:  np.ndarray  D^-1/2 *  A   * D^-1/2
    """
    out_degree = np.array(A.sum(1), dtype=np.float32)
    int_degree = np.array(A.sum(0), dtype=np.float32)

    out_degree_sqrt_inv = np.power(out_degree, -0.5, where=(out_degree != 0))
    int_degree_sqrt_inv = np.power(int_degree, -0.5, where=(int_degree != 0))
    mx_operator = np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)
    return mx_operator



def torch_sensor_to_torch_sparse_tensor(mx):
    """ Convert a torch.tensor to a torch sparse tensor.
    :param torch tensor mx
    :return: torch.sparse
    """
    index = mx.nonzero().t()
    value = mx.masked_select(mx != 0)
    shape = mx.shape
    return torch.sparse.FloatTensor(index, value, shape)




def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total {:d} Trainable {:d}'.format(total_num, trainable_num))
    return {'Total': total_num, 'Trainable': trainable_num}

