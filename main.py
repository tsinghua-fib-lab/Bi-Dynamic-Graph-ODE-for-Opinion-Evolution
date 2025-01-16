import argparse
import time
import torch.optim as optim
import datetime
from utils_in_learn_dynamics import *
from neural_dynamics import *
import functools
from PIL import Image
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser('BDG—ODE')
parser.add_argument('--method', type=str,
                    choices=['dopri5', 'adams', 'explicit_adams', 'fixed_adams','tsit5', 'euler', 'midpoint', 'rk4'],
                    default='euler')  # dopri5
parser.add_argument('--rtol', type=float, default=0.01,
                    help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
parser.add_argument('--atol', type=float, default=0.001,
                    help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')
parser.add_argument('--lr', type=float, default=0.003,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=5,
                    help='Number of hidden units.')
parser.add_argument('--sampled_time', type=str,
                    choices=['irregular', 'equal'], default='irregular')

parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')

parser.add_argument('--n', type=int, default=9484, help='Number of nodes')
parser.add_argument('--sparse', action='store_true')

parser.add_argument('--layout', type=str, choices=['community', 'degree'], default='community')
parser.add_argument('--seed', type=int, default=0, help='Random Seed')
parser.add_argument('--T', type=float, default=5., help='Terminal Time')
parser.add_argument('--operator', type=str,
                    choices=['lap', 'norm_lap', 'kipf', 'norm_adj' ], default='norm_lap')

parser.add_argument('--baseline', type=str,
                    choices=['BDG'],
                    default='BDG')
parser.add_argument('--dataset', type=str, choices=['f1', 'f2', 'f3'], default='f3',
                    help='Specify which dataset to use (f1, f2, f3)')
parser.add_argument('--dump', action='store_true', help='Save Results')

args = parser.parse_args()
if args.gpu >= 0:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

if args.dump:
    results_dir = r'results/cognitive/'
    makedirs(results_dir)


#从数据集中读入数据
dataset = args.dataset
path = f'{dataset}/{dataset}/'
recorded = np.load(path+'recorded.npy')
pos_node = []
neg_node = []
p_n_node = []
true_y = []
for i, r in enumerate(recorded):
    node = np.load(path+'node_run{:d}.npy'.format(int(r[0])))
    pos_node.append(np.sum(node>0))
    neg_node.append(np.sum(node<0))
    true_y.append(node)
    if i > 0:
        p_n = np.sum((np.array(node)<0)&(np.array(pre_node)>0))
        p_n_node.append(p_n)
        
    pre_node = node.copy()

true_y = np.concatenate(true_y, axis=1)
true_y = torch.from_numpy(true_y).float().to(device)

true_y = true_y.unsqueeze(1)

# 使用avg_pool1d对第二维进行平均池化
# kernel_size为10，表示每10个元素进行一次平均
# stride为10，表示滑动窗口的步长为10
average_y = F.avg_pool1d(true_y, kernel_size=10, stride=10)

# 将average_y的形状从[10000, 1, 50]变为[10000, 50]
average_y = average_y.squeeze(1)

true_y1 = average_y
true_y1_binary = torch.sign(true_y1)
true_y = torch.sign(average_y[:,1:])
print('true_y.shape: ', true_y.shape)

n = true_y.shape[0]
dates = list(range(true_y1.shape[1]))


#建立邻接矩阵A
file_path = f'{dataset}/{dataset}\\edge_run50000.npy'  # 构建 file_path
print(f'Using dataset: {dataset}')
print(f'File path is set to: {file_path}')
edges = np.load(file_path, allow_pickle=True)
A = np.zeros((n, n), dtype=int)

for edge in edges:
    source, target = edge
    A[source, target] = 1

A = torch.from_numpy(A).float()
D = torch.diag(A.sum(1))
L = (D - A)

def calculate_infection_ratio(A, true_y):
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
            neutral_users_previous_day = torch.where(true_y[:, day-1] != -1)[0]
            possible_negative_infected_users = torch.tensor(list(set(neutral_users_previous_day.tolist()).intersection(set(possible_negative_infected_users.tolist()))))
            real_negative_infected_users = torch.tensor(list(set(possible_negative_infected_users.tolist()).intersection(set(torch.where(true_y[:, day] == -1)[0].tolist()))))
            real_negative_infected_users = torch.tensor(list(set(real_negative_infected_users.tolist()).difference(already_infected_users)))  # 排除已经被感染的用户
            already_infected_users.update(real_negative_infected_users.tolist())  # 更新已经被感染的用户列表

        # 在可能正感染者中找到那些在前一天观点为0在当天转变为1的用户
        if day > 0:
            neutral_users_previous_day = torch.where(true_y[:, day-1] != 1)[0]
            possible_positive_infected_users = torch.tensor(list(set(neutral_users_previous_day.tolist()).intersection(set(possible_positive_infected_users.tolist()))))
            real_positive_infected_users = torch.tensor(list(set(possible_positive_infected_users.tolist()).intersection(set(torch.where(true_y[:, day] == 1)[0].tolist()))))
            real_positive_infected_users = torch.tensor(list(set(real_positive_infected_users.tolist()).difference(already_infected_users)))  # 排除已经被感染的用户
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




timetick = int(true_y.shape[1]//1.2)
print('Time tick: ', timetick)

if args.sampled_time == 'irregular':
    id_test = list(range(timetick, int(true_y.shape[1])))  # last 20 beyond 100 for test (extrapolation)
    '''
    注意一下id_test2的选取,分别对应两个不同的实验
    '''
    id_test2 = list(range(25, timetick))
    id_valid = list(range(20, 25))
    id_train = list(set(range(timetick)) - set(id_test2) - set(id_valid))  # first 80 in 100 for train
    id_train.sort()
    t_train = torch.tensor(id_train)
    t_valid = torch.tensor(id_valid)
    t_test = torch.tensor(id_test)
    t_test2 = torch.tensor(id_test2)
    t = torch.cat((t_train, t_valid, t_test, t_test2), dim=0)
    t, _ = t.sort()
    t.to(device)
    print('Train time: ', id_train)
    print('Valid time: ', id_valid)
    print('Test time: ', id_test)
    print('Test time2: ', id_test2)
    print(t)
if args.operator == 'lap':
    print('Graph Operator: Laplacian')
    OM = L
elif args.operator == 'kipf':
    print('Graph Operator: Kipf')
    OM = torch.FloatTensor(zipf_smoothing(A.numpy()))
elif args.operator == 'norm_adj':
    print('Graph Operator: Normalized Adjacency')
    OM = torch.FloatTensor(normalized_adj(A.numpy()))
else:
    print('Graph Operator[Default]: Normalized Laplacian')
    OM = torch.FloatTensor(normalized_laplacian(A.numpy()))  # L # normalized_adj



if args.sparse:
    # For small network, dense matrix is faster
    # For large network, sparse matrix cause less memory
    L = torch_sensor_to_torch_sparse_tensor(L)
    A = torch_sensor_to_torch_sparse_tensor(A)
    OM = torch_sensor_to_torch_sparse_tensor(OM)

x0 = torch.zeros(n, 1)  # 9484 * 1

now = datetime.datetime.now()
appendix = now.strftime("%m%d-%H%M%S")

  # Skip the first time point
true_y0 = x0.float().to(device)  # 9484 * 1
true_y_train = true_y[:, id_train].float().to(device)  # 9484*80  for train
true_y_valid = true_y[:, id_valid].float().to(device)  # 9484*20  for validation
true_y_test = true_y[:, id_test].float().to(device)  # 9484*20  for extrapolation prediction
if args.sampled_time == 'irregular':
    true_y_test2 = true_y[:, id_test2].to(device)  # 9484*20  for interpolation prediction
L = L.to(device)  # 9484 * 9484
OM = OM.to(device)  # 9484 * 9484
A = A.to(device)

changed_users_true = []
pos_to_neg_users_true = []
neg_to_pos_users_true = []
pos_users_true = []
neg_users_true = []
for ii in range(len(t)+1):
    if ii > 0:
        changed_users = ((true_y1_binary[:,ii] != true_y1_binary[:, ii-1]) & (true_y1_binary[:, ii] != 0) & (true_y1_binary[:, ii-1] != 0)).int().sum()
        pos_to_neg_users =  ((true_y1_binary[:,ii] == -1) & (true_y1_binary[:, ii-1] == 1)).int().sum()
        neg_to_pos_users = ((true_y1_binary[:, ii] == 1) & (true_y1_binary[:, ii - 1] == -1)).int().sum()
        pos_to_neg_users_true.append(pos_to_neg_users.item())
        neg_to_pos_users_true.append(neg_to_pos_users.item())
        changed_users_true.append(changed_users.item())
    pos_users = (true_y1_binary[:, ii] == 1).int().sum()
    neg_users = (true_y1_binary[:, ii] == -1).int().sum()
    # print('Time: ', ii, 'Positive Users: ', pos_users.item(), 'Negative Users: ', neg_users.item())
    pos_users_true.append(pos_users.item())
    neg_users_true.append(neg_users.item())


change_users_per_day = torch.tensor(changed_users_true).to(device)
change_users_per_day = change_users_per_day[id_train].float()
change_users_per_day = torch.log10(change_users_per_day + 1.1)

# Build model
input_size = true_y0.shape[1]   # y0: 9484*1 ,  input_size:1
hidden_size = args.hidden  # args.hidden  # 20 default  # [9484 * 1 ] * [1 * 20] = 9484 * 20
dropout = args.dropout  # 0 default, not stochastic ODE
num_classes = 1  # 1 for regression
# Params for discrete models
input_n_graph= true_y0.shape[0]
hidden_size_gnn = 5
hidden_size_rnn = 10


flag_model_type = ""  # "continuous" "discrete"  input, model, output format are little different
# Continuous time network dynamic models
if args.baseline == 'BDG':
    print('Choose model:' + args.baseline)
    flag_model_type = "continuous"
    p_hidden = 10
    prep_hidden = 10
    model = BDG_ODE(input_size=input_size, hidden_size=hidden_size, A=OM, num_classes=num_classes, n=n,
                 dropout=dropout, no_embed=False, no_graph=False, no_control=False,
                 rtol=args.rtol, atol=args.atol, method=args.method)


model = model.to(device)
# model = nn.Sequential(*embedding_layer, *neural_dynamic_layer, *semantic_layer).to(device)

num_paras = get_parameter_number(model)

if __name__ == '__main__':
    # Initialize lists to store loss data
    abs_errors = []
    rel_errors = []
    f1_1_values = []
    f1_2_values = []
    iterations = []
    best_train_accuracy = 0
    best_validation_accuracy = 0
    best_model_state = None
    patience_counter = 0
    PATIENCE_LIMIT = 40  # Set this to the number of epochs/iterations without improvement before stopping
    BEST_MODEL_PATH = results_dir + r'/result_best_model' + appendix + '.' + args.baseline
    os.makedirs("temp_frames", exist_ok=True)
    frames1 = []
    frames2 = []
    frames3 = []
    frames4 = []
    frames5 = []
    frame_files_combined = []
    fig, ax = plt.subplots()
    t_start = time.time()
    params = model.parameters()
    optimizer = optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = F.mse_loss  # F.mse_loss(pred_y, true_y)
    # time_meter = RunningAverageMeter(0.97)
    # loss_meter = RunningAverageMeter(0.97)
    if args.dump:
        results_dict = {
            'args': args.__dict__,
            'v_iter': [],
            'abs_error': [],
            'rel_error': [],
            'true_y': [true_y.t()],
            'predict_y': [],
            'abs_error2': [],
            'rel_error2': [],
            'predict_y2': [],
            'model_state_dict': [],
            'total_time': []}

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        if flag_model_type == "continuous":
            pred_y1, pred_y2, x0 = model(t)  # 21 * 9484 * 1
            pred_y1 = pred_y1.squeeze().t() # 9484*21
            pred_y2 = pred_y2.squeeze().t() # 9484*21
            pred_y = pred_y1 - pred_y2
            # loss_mse = (criterion(pred_y, true_y_train, reduction='none')*changed[:,id_train]).sum().float()/changed[:,id_train].sum().float() # true_y)  # 9484 * 20 (time_tick)
            # result = true_y_train * changed[:, id_train]
            # non_zero_values = result[result != 0]
            # print(non_zero_values)

            pred_y_train = pred_y[:, id_train]
            mask = (true_y_train != 0.0)
            '''
            计算MSE损失
            364-366行代码对应一种算法, 367行代码对应另一种算法
            '''
            # true_y_non_zero = true_y_train[mask]
            # pred_y_non_zero = pred_y_train[mask]
            # loss_mse = criterion(pred_y_non_zero, true_y_non_zero)
            loss_mse = torch.matmul((criterion(pred_y_train, true_y_train, reduction='none') * mask).float(),change_users_per_day).sum().float()/mask.sum().float()
            # 获取pred_y的前一列
            prev_column = torch.cat((x0, pred_y[:, :-1]), dim=1)
            # 获取pred_y除了第一列的所有列
            current_column = pred_y
            # prev_column = pred_y[:, :-1]
            # current_column = pred_y[:, 1:]
            difference = current_column - prev_column
            # 计算正则化项

            gradient_input = torch.autograd.grad(outputs=loss_mse, inputs=pred_y_train, grad_outputs=torch.ones_like(loss_mse), create_graph=True)[0]
            gradient_absolute = torch.abs(gradient_input)
            loss_derivative = gradient_absolute.pow(2).sum()

            '''
            训练参数！
            '''
            alpha = 1.0e-5
            beta = 2.5e-5
            lambda1 = 1.0e-5
            delta = 1.0

            regularization_term = alpha* loss_derivative +beta * difference.pow(2).sum()#/(n*difference.shape[1])
            
            #约束pred_y1和pred_y2不能同时有值
            loss_add = lambda1 * (pred_y1 * pred_y2).pow(2).sum()

            loss_x0 = delta * criterion(x0.squeeze(),true_y1[:,0])

            loss_train = loss_mse + regularization_term + loss_add + loss_x0
            # torch.mean(torch.abs(pred_y - batch_y))
            relative_loss_train = criterion(pred_y_train, true_y_train) / true_y_train.mean()

        else:
            print("flag_model_type NOT DEFINED!")
            exit(-1)

        loss_train.backward()
        optimizer.step()

        # time_meter.update(time.time() - t_start)
        # loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                if flag_model_type == "continuous":
                    # pred_y = model(true_y0).squeeze().t() # odeint(model, true_y0, t)
                    # loss = criterion(pred_y, true_y)
                    # relative_loss = criterion(pred_y, true_y) / true_y.mean()
                    pred_y1, pred_y2, x0 = model(t)  # odeint(model, true_y0, t)
                    pred_y1 = pred_y1.squeeze().t() # 9484*21
                    pred_y2 = pred_y2.squeeze().t()
                    pred_y = pred_y1 - pred_y2
                    loss = criterion(pred_y[:,id_test], true_y_test)
                    relative_loss = criterion(pred_y[:, id_test], true_y_test)
                    vaild_loss = criterion(pred_y[:,id_valid], true_y_valid) 
                    if args.sampled_time == 'irregular': # for interpolation results
                        loss2 = criterion(pred_y[:,id_test2], true_y_test2)
                        relative_loss2 = criterion(pred_y[:, id_test2], true_y_test2)

                if args.dump:
                    # Info to dump
                    results_dict['v_iter'].append(itr)
                    results_dict['abs_error'].append(loss.item())    # {'abs_error': [], 'rel_error': [], 'X_t': []}
                    results_dict['rel_error'].append(relative_loss.item())
                    results_dict['predict_y'].append(pred_y[:, id_test])
                    results_dict['model_state_dict'].append(model.state_dict())
                    if args.sampled_time == 'irregular':  # for interpolation results
                        results_dict['abs_error2'].append(loss2.item())  # {'abs_error': [], 'rel_error': [], 'X_t': []}
                        results_dict['rel_error2'].append(relative_loss2.item())
                        results_dict['predict_y2'].append(pred_y[:, id_test2])
                    # now = datetime.datetime.now()
                    # appendix = now.strftime("%m%d-%H%M%S")
                    # results_dict_path = results_dir + r'/result_' + appendix + '.' + args.dump_appendix
                    # torch.save(results_dict, results_dict_path)
                    # print('Dump results as: ' + results_dict_path)
                if args.sampled_time == 'irregular':
                    print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                          '| Test Loss {:.6f}({:.6f} Relative) '
                          '| Test Loss2 {:.6f}({:.6f} Relative) '
                          '| Valid Loss {:.6f}({:.6f} Relative)'
                          '| Time {:.4f}'
                          .format(itr, loss_train.item(), relative_loss_train.item(),
                                  loss.item(), relative_loss.item(),
                                  loss2.item(), relative_loss2.item(),
                                  vaild_loss.item(), relative_loss.item(),
                                  time.time() - t_start))
                else:
                    print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                          '| Test Loss {:.6f}({:.6f} Relative) '
                          '| Time {:.4f}'
                          .format(itr, loss_train.item(), relative_loss_train.item(),
                                  loss.item(), relative_loss.item(),
                                  time.time() - t_start))
                    
                pred_binary = custom_sign(pred_y[:,id_train])  # 将预测值转换为二分类
                pred_binary2 = custom_sign(pred_y[:, id_valid])  # 将预测值转换为二分类
                pred_binary1 = custom_sign(pred_y[:, id_test])  # 将预测值转换为二分类
                true_y_binary = true_y_train.int()  # 将真实值转换为二分类
                true_y_binary2 = true_y_valid.int()
                true_y_binary1 = true_y_test.int()  # 将真实值转换为二分类

                mask = (true_y_binary != 0)
                mask1 = (true_y_binary1 != 0)
                mask2 = (true_y_binary2 != 0)

                masked_pred_binary = pred_binary[mask]
                masked_true_y_binary = true_y_binary[mask]
                masked_pred_binary1 = pred_binary1[mask1]
                masked_true_y_binary1 = true_y_binary1[mask1]
                masked_pred_binary2 = pred_binary2[mask2]
                masked_true_y_binary2 = true_y_binary2[mask2]

                accuracy0 = (masked_pred_binary == masked_true_y_binary).float().mean().item()
                accuracy1 = (masked_pred_binary1 == masked_true_y_binary1).float().mean().item()
                accuracy2 = (masked_pred_binary2 == masked_true_y_binary2).float().mean().item()

                if args.sampled_time == 'irregular':
                    pred_binary2 = custom_sign(pred_y[:, id_test2])  # 将预测值转换为二分类
                    true_y_binary2 = true_y_test2.int()
                    mask2 = (true_y_binary2 != 0)
                    masked_pred_binary2 = pred_binary2[mask2]
                    masked_true_y_binary2 = true_y_binary2[mask2]
                    # 1. 拼接预测张量
                    pred = torch.cat((masked_pred_binary1, masked_pred_binary2), dim=0)

                    # 2. 拼接真实标签张量
                    true_y_combine = torch.cat((masked_true_y_binary1, masked_true_y_binary2), dim=0)

                    # 3. 计算准确率
                    accuracy3 = (pred == true_y_combine).float().mean().item()
                    # 4. 计算f1
                    f1_out = compute_f1(pred, true_y_combine)
                    print('Model Accuracy: {:.6f} | Model f1 : {:.6f} | Train Accuracy : {:.6f}| Vaild Accuracy : {:.6f}'.format(accuracy3, f1_out,accuracy0, accuracy2))
                else:
                    print('Model Accuracy: {:.6f} | Train Accuracy : {:.6f}'.format(accuracy3, accuracy0))

                # 计算F1值

                f_1_1 = compute_f1(masked_pred_binary1, masked_true_y_binary1)
                f_1_2 = compute_f1(masked_pred_binary2, masked_true_y_binary2)
                f1_1_values.append(f_1_1)
                f1_2_values.append(f_1_2)
                iterations.append(itr)

                # 早停
                if accuracy2 >= best_validation_accuracy and accuracy0 >= best_train_accuracy:
                    best_train_accuracy = accuracy0
                    best_validation_accuracy = accuracy2
                    best_model_state = model.state_dict()
                    torch.save(best_model_state, BEST_MODEL_PATH)
                    print('Best Model Saved at Epoch: ', itr)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE_LIMIT:
                        print('Early stopping at epoch: ', itr)
                        break

        #绘制动图
        if itr % 5 == 0:
            with torch.no_grad():
                pred_y1, pred_y2, x0 = model(t)
                pred_y1 = pred_y1.squeeze().t() # 9484*21
                pred_y2 = pred_y2.squeeze().t()
                pred_y = pred_y1 - pred_y2
                pred_data = torch.sign(torch.cat((x0, pred_y), dim=1)).int()
                changed_users_pred = []
                pos_to_neg_users_pred = []
                neg_to_pos_users_pred = []
                pos_users_pred = []
                neg_users_pred = []
                pos_users_ob_pred = []
                neg_users_ob_pred = []
                for ii in range(pred_data.shape[1]):
                    if ii > 0:
                        changed_users = ((pred_data[:,ii] != pred_data[:, ii-1]) & (pred_data[:, ii] != 0) & (pred_data[:, ii-1] != 0) & (true_y1[:,ii]!=0)).int().sum()
                        pos_to_neg_users = ((pred_data[:, ii] == -1) & (pred_data[:, ii - 1] == 1) & (true_y1[:,ii]!=0)).int().sum()
                        neg_to_pos_users = ((pred_data[:, ii] == 1) & (pred_data[:, ii - 1] == -1) & (true_y1[:,ii]!=0)).int().sum()
                        pos_to_neg_users_pred.append(pos_to_neg_users.item())
                        neg_to_pos_users_pred.append(neg_to_pos_users.item())
                        changed_users_pred.append(changed_users.item())    
                    pos_users = ((pred_data[:, ii] == 1) & (true_y1[:, ii] != 0.0)).int().sum()
                    neg_users = ((pred_data[:, ii] == -1) & (true_y1[:, ii] != 0.0)).int().sum()
                    pos_users_pred.append(pos_users.item())
                    neg_users_pred.append(neg_users.item())

                plt.figure()
                plot_attitude_changes_no_date(dates[1:int(true_y.shape[1])+1], neg_to_pos_users_true, neg_to_pos_users_pred, 'Attitude n-p Changes Over Time', group_num= itr, show=False)
                frame_name_attitude = f"temp_frames/frame_n-p_{itr}.png"
                plt.savefig(frame_name_attitude)
                frames1.append(frame_name_attitude)
                plt.close()
                plt.figure()
                frame_name_loss = f"temp_frames/frame_p-n_{itr}.png"
                plot_attitude_changes_no_date(dates[1:int(true_y.shape[1])+1], pos_to_neg_users_true, pos_to_neg_users_pred, 'Attitude p-n Changes Over Time', group_num= itr, show=False)
                plt.savefig(frame_name_loss)
                frames2.append(frame_name_loss)
                plt.close()

                plt.figure()
                frame_name_loss1 = f"temp_frames/frame_p_{itr}.png"
                plot_attitude_changes_no_date(dates[0:int(true_y.shape[1])+1], pos_users_true,pos_users_pred,'pos users', group_num= itr, show=False)
                plt.savefig(frame_name_loss1)
                frames3.append(frame_name_loss1)
                plt.close()

                plt.figure()
                frame_name_loss2 = f"temp_frames/frame_n_{itr}.png"
                plot_attitude_changes_no_date(dates[0:int(true_y.shape[1])+1], neg_users_true, neg_users_pred, 'neg users', group_num= itr, show=False)
                plt.savefig(frame_name_loss2)
                frames4.append(frame_name_loss2)
                plt.close()

                plt.figure()
                frame_name_loss2 = f"temp_frames/frame_c_{itr}.png"
                plot_attitude_changes_no_date(dates[1:int(true_y.shape[1])+1], changed_users_true, changed_users_pred, 'change users', group_num= itr, show=False)
                plt.savefig(frame_name_loss2)
                frames5.append(frame_name_loss2)
                plt.close()

                frame_name_combined = f"temp_frames/frame_combined_{itr}.png"
                draw_loss_and_save(pred_y, true_y, id_train, criterion, frame_name_combined, itr)
                frame_files_combined.append(frame_name_combined)


    # 使用生成的图像创建GIF
    with Image.open(frames1[0]) as img:
        img.save('animated_attitude_n-p.gif', save_all=True, append_images=[Image.open(f) for f in frames1[1:]], duration=300, loop=0)

    with Image.open(frames2[0]) as img:
        img.save('animated_attitude_p-n.gif', save_all=True, append_images=[Image.open(f) for f in frames2[1:]], duration=300, loop=0)

    with Image.open(frame_files_combined[0]) as img:
        img.save('animated_loss.gif', save_all=True, append_images=[Image.open(f) for f in frame_files_combined[1:]], duration=300, loop=0)

    with Image.open(frames3[0]) as img:
        img.save('animated_attitude_p.gif', save_all=True, append_images=[Image.open(f) for f in frames3[1:]], duration=300, loop=0)

    with Image.open(frames4[0]) as img:
        img.save('animated_attitude_n.gif', save_all=True, append_images=[Image.open(f) for f in frames4[1:]], duration=300, loop=0)

    with Image.open(frames5[0]) as img:
        img.save('animated_combined.gif', save_all=True, append_images=[Image.open(f) for f in frames5[1:]], duration=300, loop=0)

    #删除临时图像
    for frame_file in frames1:
        if os.path.exists(frame_file):
            os.remove(frame_file)
    for frame_file in frames2:
        if os.path.exists(frame_file):
            os.remove(frame_file)
    for frame_file in frame_files_combined:
        if os.path.exists(frame_file):
            os.remove(frame_file)
    for frame_file in frames3:
        if os.path.exists(frame_file):
            os.remove(frame_file)
    for frame_file in frames4:
        if os.path.exists(frame_file):
            os.remove(frame_file)
    for frame_file in frames5:
        if os.path.exists(frame_file):
            os.remove(frame_file)

# 取出保存的最佳模型并计算准确率
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    with torch.no_grad():
        if flag_model_type == "continuous":
            pred_y1, pred_y2, x0 = model(t)  # odeint(model, true_y0, t)
            pred_y1 = pred_y1.squeeze().t() # 9484*21
            pred_y2 = pred_y2.squeeze().t()
            pred_y = pred_y1 - pred_y2
            loss = criterion(pred_y[:,id_test], true_y_test)
            relative_loss = criterion(pred_y[:, id_test], true_y_test)
            vaild_loss = criterion(pred_y[:,id_valid], true_y_valid)
            if args.sampled_time == 'irregular': # for interpolation results
                loss2 = criterion(pred_y[:,id_test2], true_y_test2)
                relative_loss2 = criterion(pred_y[:, id_test2], true_y_test2)

        pred_binary = torch.sign(torch.cat((x0, pred_y), dim=1)).int()
        changed_users_pred = []
        pos_to_neg_users_pred = []
        neg_to_pos_users_pred = []
        pos_users_pred = []
        neg_users_pred = []

        for ii in range(pred_binary.shape[1]):
            pos_users = ((pred_binary[:, ii] == 1) & (true_y1[:, ii] != 0.0)).int().sum()
            neg_users = ((pred_binary[:, ii] == -1) & (true_y1[:, ii] != 0.0)).int().sum()
            pos_users_pred.append(pos_users.item())
            neg_users_pred.append(neg_users.item())

        plt.figure()
        plot_attitude_changes_no_date(dates[0:int(true_y.shape[1])+1], pos_users_true,pos_users_pred,'pos users', show=False)
        plt.figure()
        plot_attitude_changes_no_date(dates[0:int(true_y.shape[1])+1], neg_users_true,neg_users_pred,'neg users', show=False)

        now = datetime.datetime.now()
        appendix = now.strftime("%m%d-%H%M%S")



        if args.dump:
            # Info to dump
            results_dict['v_iter'].append(itr)
            results_dict['abs_error'].append(loss.item())    # {'abs_error': [], 'rel_error': [], 'X_t': []}
            results_dict['rel_error'].append(relative_loss.item())
            results_dict['predict_y'].append(pred_y[:, id_test])
            results_dict['model_state_dict'].append(model.state_dict())
            if args.sampled_time == 'irregular':  # for interpolation results
                results_dict['abs_error2'].append(loss2.item())  # {'abs_error': [], 'rel_error': [], 'X_t': []}
                results_dict['rel_error2'].append(relative_loss2.item())
                results_dict['predict_y2'].append(pred_y[:, id_test2])

        if args.sampled_time == 'irregular':
            print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                    '| Test Loss {:.6f}({:.6f} Relative) '
                    '| Test Loss2 {:.6f}({:.6f} Relative) '
                    '| Valid Loss {:.6f}({:.6f} Relative)'
                    '| Time {:.4f}'
                    .format(itr, loss_train.item(), relative_loss_train.item(),
                            loss.item(), relative_loss.item(),
                            loss2.item(), relative_loss2.item(),
                            vaild_loss.item(), relative_loss.item(),
                            time.time() - t_start))
        else:
            print('Iter {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                    '| Test Loss {:.6f}({:.6f} Relative) '
                    '| Time {:.4f}'
                    .format(itr, loss_train.item(), relative_loss_train.item(),
                            loss.item(), relative_loss.item(),
                            time.time() - t_start))
            
        pred_binary = custom_sign(pred_y[:,id_train])  # 将预测值转换为二分类
        pred_binary2 = custom_sign(pred_y[:, id_valid])  # 将预测值转换为二分类
        pred_binary1 = custom_sign(pred_y[:, id_test])  # 将预测值转换为二分类
        true_y_binary = true_y_train.int()  # 将真实值转换为二分类
        true_y_binary2 = true_y_valid.int()
        true_y_binary1 = true_y_test.int()  # 将真实值转换为二分类

        mask = (true_y_binary != 0)
        mask1 = (true_y_binary1 != 0)
        mask2 = (true_y_binary2 != 0)

        masked_pred_binary = pred_binary[mask]
        masked_true_y_binary = true_y_binary[mask]
        masked_pred_binary1 = pred_binary1[mask1]
        masked_true_y_binary1 = true_y_binary1[mask1]
        masked_pred_binary2 = pred_binary2[mask2]
        masked_true_y_binary2 = true_y_binary2[mask2]

        accuracy0 = (masked_pred_binary == masked_true_y_binary).float().mean().item()
        accuracy1 = (masked_pred_binary1 == masked_true_y_binary1).float().mean().item()
        accuracy2 = (masked_pred_binary2 == masked_true_y_binary2).float().mean().item()

        if args.sampled_time == 'irregular':
            pred_binary2 = custom_sign(pred_y[:, id_test2])  # 将预测值转换为二分类
            true_y_binary2 = true_y_test2.int()
            mask2 = (true_y_binary2 != 0)
            masked_pred_binary2 = pred_binary2[mask2]
            masked_true_y_binary2 = true_y_binary2[mask2]
            # 1. 拼接预测张量
            pred = torch.cat((masked_pred_binary1, masked_pred_binary2), dim=0)

            # 2. 拼接真实标签张量
            true_y_combine = torch.cat((masked_true_y_binary1, masked_true_y_binary2), dim=0)

            # 3. 计算准确率
            accuracy3 = (pred == true_y_combine).float().mean().item()
            # 4. 计算f1
            f1_out = compute_f1(pred, true_y_combine)
            print('Model Accuracy: {:.6f} | Model f1 : {:.6f} | Train Accuracy : {:.6f}| Vaild Accuracy : {:.6f}'.format(accuracy3, f1_out,accuracy0, accuracy2))
        else:
            print('Model Accuracy: {:.6f} | Train Accuracy : {:.6f}'.format(accuracy3, accuracy0))



        t_total = time.time() - t_start
        print('Total Time {:.4f}'.format(t_total))
        num_paras = get_parameter_number(model)
        if args.dump:
            results_dict['total_time'] = t_total
            results_dict_path = results_dir + r'/result_' + appendix + '.' + args.baseline  #args.dump_appendix
            torch.save(results_dict, results_dict_path)
            print('Dump results as: ' + results_dict_path)

            # Test dumped results:
            rr = torch.load(results_dict_path)
            fig, ax = plt.subplots()
            ax.plot(rr['v_iter'], rr['abs_error'], '-', label='Absolute Error')
            ax.plot(rr['v_iter'], rr['rel_error'], '--', label='Relative Error')
            legend = ax.legend( fontsize='x-large') # loc='upper right', shadow=True,
            # legend.get_frame().set_facecolor('C0')
            fig.savefig(results_dict_path + ".png", transparent=True)
            fig.savefig(results_dict_path + ".pdf", transparent=True)
            plt.show()
            plt.pause(0.001)
            plt.close(fig)

            caculate_combine_Pbar(A,A,true_y,pred_y)

# python main.py --T 5 --sampled_time irregular --baseline BDG --gpu 1 --weight_decay 1e-3 --lr 0.02 --sparse --niters 1500 --dump --dataset f3