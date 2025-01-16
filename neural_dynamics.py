import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq as ode
import random


class ODEFunc(nn.Module):  # A kind of ODECell in the view of RNN
    def __init__(self, hidden_size, A, dropout=0.0, no_graph=False, no_control=False):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A  # N_node * N_node
        # self.nfe = 0
        self.wt = nn.Linear(hidden_size, hidden_size)
        self.no_graph = no_graph
        self.no_control = no_control

    def forward(self, t, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        # self.nfe += 1
        if not self.no_graph:
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                x = torch.sparse.mm(self.A, x)
            else:
                x = torch.mm(self.A, x)
        if not self.no_control:
            x = self.wt(x)
        x = self.dropout_layer(x)
        # x = torch.tanh(x)
        x = F.relu(x)
        # !!!!! Not use relu seems doesn't  matter!!!!!! in theory. Converge faster !!! Better than tanh??
        # x = torch.sigmoid(x)
        return x

class ODEFunc1(nn.Module):  # A kind of ODECell in the view of RNN
    def __init__(self, hidden_size, A, dropout=0.0, no_graph=False, no_control=False):
        super(ODEFunc1, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.A = A  # N_node * N_node
        # self.nfe = 0
        self.wt1 = nn.Linear(hidden_size//2, hidden_size//2)
        self.wt2 = nn.Linear(hidden_size//2, hidden_size//2)
        self.wt3 = nn.Linear(hidden_size//2, hidden_size//2)
        self.wt4 = nn.Linear(hidden_size//2, hidden_size//2)
        self.no_graph = no_graph
        self.no_control = no_control

    def forward(self, t, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        y1 = x[:,:(self.hidden_size//2)]
        y2 = x[:,(self.hidden_size//2):]
        # self.nfe += 1
        if not self.no_graph:
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                y1 = torch.sparse.mm(self.A, y1)
                y2 = torch.sparse.mm(self.A, y2)
            else:
                y1 = torch.mm(self.A, y1)
                y2 = torch.mm(self.A, y2)
        if not self.no_control:
            y1_1 = self.wt1(y1)
            y2_1 = self.wt2(y2)
            y1_2 = self.wt3(y1)
            y2_2 = self.wt4(y2)
        y1 = y1_1 + y2_1
        y2 = y1_2 + y2_2
        # x = torch.tanh(x)
        x = torch.cat((y1, y2), dim = 1)
        x = self.dropout_layer(x)
        x = F.relu(x)
        # !!!!! Not use relu seems doesn't  matter!!!!!! in theory. Converge faster !!! Better than tanh??
        # x = torch.sigmoid(x)
        return x

    
class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False): #vt,         :param vt:
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # self.integration_time_vector = vt  # time vector
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        vt = vt.to(x.device)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out  # 100 * 400 * 10




class BDG_ODE(nn.Module):  # myModel
    def __init__(self, input_size, hidden_size, A, num_classes, n, dropout=0.0,
                 no_embed=False, no_graph=False, no_control=False,
                 rtol=.01, atol=.001, method='dopri5'):
        super(BDG_ODE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.A = A  # N_node * N_node
        self.num_classes = num_classes
        self.n = n

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.no_embed = no_embed
        self.no_graph = no_graph
        self.no_control = no_control

        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.preprocess_layer1 = nn.Sequential(nn.Linear(input_size, input_size, bias=True), nn.Sigmoid())
        self.preprocess_layer2 = nn.Sequential(nn.Linear(input_size, input_size, bias=True), nn.Sigmoid())
        self.input_layer1 = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.input_layer2 = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                               nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(
                ODEFunc1(hidden_size*2, A, dropout=dropout, no_graph=no_graph, no_control=no_control),# OM
                rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer1 = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Sigmoid())
        self.output_layer2 = nn.Sequential(nn.Linear(hidden_size, num_classes, bias=True), nn.Sigmoid())
        # 在 __init__ 方法内部
        total_elements = n * input_size  # n是节点数，input_size是每个节点的维度
        num_minus_ones = int(total_elements  / 2)  # -1的数量，按照3:2的比例
        num_ones = total_elements - num_minus_ones    # 1的数量

        # 创建一个包含正确数量的-1和1的数组
        values = [-1] * num_minus_ones + [1] * num_ones
        random.shuffle(values)  # 随机排列

        # 将一维数组转化为二维形状 (n, input_size) 的tensor
        initial_values = torch.tensor(values).view(n, input_size).float()

        self.y1 = nn.Parameter(torch.relu(initial_values))
        self.y2 = nn.Parameter(torch.relu(-initial_values))
        # 将initial_values赋值为self.x0的初值
    def forward(self, vt):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        y1 = self.preprocess_layer1(self.y1)
        y2 = self.preprocess_layer2(self.y2)
        x0 = y1-y2
        if not self.no_embed:
            y1 = self.input_layer1(y1)
            y2 = self.input_layer2(y2)
        x = torch.cat((y1, y2), dim = 1)
        hvx = self.neural_dynamic_layer(vt, x)
        output1 = self.output_layer1(hvx[:,:, :self.hidden_size])
        output2 = self.output_layer2(hvx[:,:, self.hidden_size:])
        return output1, output2, x0


