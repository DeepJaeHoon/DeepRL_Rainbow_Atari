#-----------------Callback Enviroment Module--------------------#
import gymnasium as gym
from gymnasium.wrappers import FrameStack
#---------------------------------------------------------------#


#------------------Deep Learning Framework Module---------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
#---------------------------------------------------------------#


#-----------------Python Module-----------------#
import numpy as np
import collections
import random
import cv2
import math
import time
#-----------------------------------------------#


#----------Image_Preprocessin_Parameter-----------#
crop_start = 30 # Image의 y좌표 30부터
crop_end = 200 # 200까지만 남기도 나머지는 버린다.
resize_w = 84 # Image의 크기를 가로 84
resize_h = 84 # 세로 84로 제한한다.
max_remember_frame = 4 # 최대 프레임 저장 길이
#-------------------------------------------------#


#---------Train_Parameter-------------#
episode = 100000000 # 최대 에피소드 길이
memory_size = 300000 # PER 메모리 크기
learning_rate = 0.0000625# 학습률
target_update_interval = 8 # Target Network를 Hard update하기위한 step 주기
train_start = 50000 # 학습 시작하는 step 수
train_interval = 4 # 학습을 시작하기위한 간격
batch = 32 # 배치 사이즈
gamma = 0.99 # 감가율, Breakout은 보상이 희소해서 큰 값이 좋음
T_step = 3 # N -step Learning
atom_size = 51 # 확룰 분포의 원자 갯수
v_min = -9.9 # 확률 분포 x축의 최솟 값
v_max = 10.1 # 확률 분포 x축의 최댓 값
std_variation = 0.45 # 편차, Noisy Layer에 사용
support = torch.from_numpy(np.linspace(v_min, v_max, atom_size, True)) # 확률 분포 x축
tau = 0.005 # Soft Update 비중
#-------------------------------------#


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # CUDA를 사용한 병렬 연산이 가능한가?
print(device)
summary = SummaryWriter("C:/Users/dlwog/Desktop/RL/tensorboard/rainbow/last_carnival_parmeter")


class Agent(nn.Module):

    def __init__(self, atom_num, vmin, vmax, spp, std):
        super(Agent, self).__init__()

        self.atom_size = atom_num
        self.V_min = vmin
        self.V_max = vmax
        standard_variation = std
        self.support = spp

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.batch_1 = nn.BatchNorm2d(32)
        self.batch_2 = nn.BatchNorm2d(64)
        self.batch_3 = nn.BatchNorm2d(64)

        self.common_layer = nn.Linear(3136, 1024)

        self.fc_value1 = NoisyLinear(1024, 512, std_init = standard_variation)
        self.fc_value2= NoisyLinear(512, self.atom_size, std_init = standard_variation)
        self.fc_advantage1 = NoisyLinear(1024, 512, std_init = standard_variation)
        self.fc_advantage2= NoisyLinear(512, 3*self.atom_size, std_init = standard_variation)

        self.active = nn.GELU()
        self.flatten = nn.Flatten()

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.constant_(self.batch_1.weight, 1)
        nn.init.constant_(self.batch_1.bias, 0)
        nn.init.constant_(self.batch_2.weight, 1)
        nn.init.constant_(self.batch_2.bias, 0)
        nn.init.constant_(self.batch_3.weight, 1)
        nn.init.constant_(self.batch_3.bias, 0)
        nn.init.kaiming_uniform_(self.common_layer.weight)

    def forward(self, x):

        x = x.to(device)
        x = -1 + 2*x/255
        x = self.active(self.batch_1(self.conv1(x)))
        x = self.active(self.batch_2(self.conv2(x)))
        x = self.active(self.batch_3(self.conv3(x)))
        x = self.flatten(x)
        common = self.active(self.common_layer(x))
        noisy_value = self.fc_value2(self.active(self.fc_value1(common)))
        noisy_action = self.fc_advantage2(self.active(self.fc_advantage1(common)))

        vv = noisy_value.view(-1, 1, atom_size)
        advadv = noisy_action.view(-1, 3, atom_size)

        advAvantage = torch.mean(advadv, dim=1, keepdim=True) 
        Q = vv + (advadv - advAvantage)
        dist = F.softmax(Q, dim=-1)
        dist = dist.clamp(min=1e-3)
        before_sum = dist * self.support.to(device)
        avg = torch.sum(before_sum, dim=2)

        return avg.to("cpu"), dist.to("cpu")
    
    def reset_noise(self):
        self.fc_value1.reset_noise()
        self.fc_value2.reset_noise()
        self.fc_advantage1.reset_noise()
        self.fc_advantage2.reset_noise()

class NoisyLinear(nn.Module):
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class Prioritized_Experience_Replay: # 기존의 Replay MEmory보다 상위 호환인 PER

    data_pointer = 0

    def __init__(self, capacity):

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

        self.PER_e = 0.01
        self.PER_a = 0.70
        self.PER_b = 0.30
        self.PER_b_increment_per_sampling = 0.00001

    def add(self, priority, data):

        tree_index = self.data_pointer + self.capacity - 1 # 현재 인덱스를 구해서 
        self.data[self.data_pointer] = data # 상태 천이 값 집어넣기?
        self.update(tree_index, priority) # leaf(priority) 업데이트, TD ERROR 계산 -> PER에서 우선 순위 계산 -> SUMTREE에 반영
        self.data_pointer += 1  # pointer를 1 증가시킴

        if self.data_pointer >= self.capacity:  # capacity를 넘었다면 첫번째 index로 돌아감
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # leaf priority score 업데이트
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        # parent가 0이면 중단. root node에 도달했기 때문
        if parent != 0:
            self._propagate(parent, change)

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    # 이진 트리 구조를 사용하여 특정 조건을 만족하는 노드를 찾는 재귀 함수(recursive function)
    # 주어진 값 s에 대해 특정 조건을 만족하는 노드의 인덱스를 찾아라.
    def _retrieve(self, idx, s):
        left_child_index = 2 * idx + 1
        right_child_index = left_child_index + 1
        # 현재 노드가 leaf node(자식이 없는 node)인 경우를 검
        if left_child_index >= len(self.tree):
            return idx
        # s가 왼쪽 자식 노드에 저장된 값보다 작거나 같으면, 왼쪽 자식으로 재귀적으로 이동
        if s <= self.tree[left_child_index]:
            return self._retrieve(left_child_index, s)
        else:
            return self._retrieve(right_child_index, s - self.tree[left_child_index])

    def get_leaf(self, s):
        leaf_index = self._retrieve(0, s)
        data_index = leaf_index - self.capacity + 1
        return (leaf_index, self.tree[leaf_index], self.data[data_index])

    # 루트 노드를 반환
    def total_priority(self):
        return self.tree[0]
    
    def _getPriority(self, error):
        return (error + self.PER_e) ** self.PER_a

    def store(self, error, sample):
        max_priority = self._getPriority(error)
        self.add(max_priority, sample)

    # Update the priorities on the tree
    def batch_update(self, idx, error):
        p = self._getPriority(error)
        self.update(idx, p)

    def sample(self, n):
        
        minibatch = []
        idxs = []
        priority_segment = self.total_priority() / n
        priorities = []
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        for i in range(n):
            # A value is uniformly sample from each range
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            (idx, p, data) = self.get_leaf(value)
            priorities.append(p)
            minibatch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.total_priority()
        is_weight = np.power(self.n_entries * sampling_probabilities, -self.PER_b)
        is_weight /= is_weight.max()

        return minibatch, idxs, is_weight
    
    def size(self,):
        return self.n_entries
    
class N_step_buffer:
    def __init__(self, T_step):
        self.buffer = collections.deque(maxlen=T_step)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def put_per(self, q_network, target_network, atom_size, v_min, v_max, gamma, batch_size, per, spp): # 배치 1로 집어넣음

        state_buffer = collections.deque(maxlen=T_step)
        action_buffer = collections.deque(maxlen=T_step)
        reward_buffer = collections.deque(maxlen=T_step)
        state_prime_buffer = collections.deque(maxlen=T_step)
        done_buffer = collections.deque(maxlen=T_step)

        if self.size() % T_step == 0 and self.size() != 0:
            transition = list(self.buffer)

            G = 0

            for i in reversed(transition): # 거꾸로임

                s, a, r, s_, d = i
                G = gamma*G + r[0]
                state_buffer.append(s)
                action_buffer.append(a)
                state_prime_buffer.append(s_)
                reward_buffer.append(G)
                done_buffer.append(d)

        else:
            return 
        

        if done_buffer[0][0] == 0 and len(done_buffer) == 3: # Terminal State
            for i in range(T_step):
                n_step_gamma = math.pow(gamma, T_step + 1 - i) # 의심해
                n_step_td_loss = calcurate_cross_entropy_loss(q_network, target_network, torch.from_numpy(np.array(state_buffer[T_step -1 -i])).type(torch.float),
                                                               torch.from_numpy(np.array([action_buffer[T_step -1 -i]])).to(torch.int64),
                                                                 torch.from_numpy(np.array(state_prime_buffer[0])).type(torch.float),
                                                                   atom_size, v_min, v_max, gamma,
                                                                     torch.from_numpy(np.array([reward_buffer[T_step -1 -i]])),
                                                                       torch.from_numpy(np.array([done_buffer[0]])).to(torch.int64), batch_size, spp)
                error = n_step_td_loss.item()
                data = (state_buffer[T_step -1 -i], [action_buffer[T_step -1 -i]], [reward_buffer[T_step -1 -i]], state_prime_buffer[0], done_buffer[0])

                per.store(error, data)

            state_buffer.clear()
            action_buffer.clear()
            state_prime_buffer.clear()
            reward_buffer.clear()
            done_buffer.clear()

        elif (done_buffer[0][0] == 1 and done_buffer[1][0] == 1 and done_buffer[2][0] == 1)and len(done_buffer) % T_step == 0: # Non Terminal State
            n_step_gamma = math.pow(gamma, T_step)

            n_step_td_loss = calcurate_cross_entropy_loss(q_network, target_network, torch.from_numpy(np.array(state_buffer[-1])).type(torch.float),
                                                           torch.from_numpy(np.array([action_buffer[-1]])).to(torch.int64),
                                                             torch.from_numpy(np.array(state_prime_buffer[0])).type(torch.float),
                                                               atom_size, v_min, v_max, n_step_gamma,
                                                                 torch.from_numpy(np.array([reward_buffer[-1]])),
                                                                   torch.from_numpy(np.array([done_buffer[0]])).to(torch.int64),
                                                                     batch_size, spp)
            error = n_step_td_loss.item()
            data = (state_buffer[-1], [action_buffer[-1]], [reward_buffer[-1]], state_prime_buffer[0], done_buffer[0])
            per.store(error, data)

        elif len(done_buffer) % T_step != 0:
            return #0, 0
    
    def size(self):
        return len(self.buffer)
    
def preprocess_frame(frame): # 기존 Image 전처리

    empty_frame = np.zeros((max_remember_frame, 84, 84), dtype=np.uint8)

    for i in range(max_remember_frame):
        # 이미지를 y축으로 30부터 200까지 자르기
        cropped_frame = frame[i][30:200, :, :]
        
        # 이미지 크기 조정
        resized_frame = cv2.resize(cropped_frame, (84, 84))
        
        # 그레이스케일로 변환
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
        
        empty_frame[i] = gray_frame

    return empty_frame

def calcurate_cross_entropy_loss(q_network, target_network, s, a, s_, atom_size, v_min, v_max, gamma, reward, done, batch_size, spp):

    #start = time.time()

    if s.shape == (4, 84, 84):
        _, q_dist = q_network(s.unsqueeze(0))

    else:
        _, q_dist = q_network(s)

    q_dist_a = torch.gather(q_dist, 1, a.view(batch_size, 1, 1).expand(batch_size, 1, 51)) # state에서 내가 행동한 action


    if s.shape == (4, 84, 84):
        q_prime_value, _ = q_network(s_.unsqueeze(0))

    else:
        q_prime_value, _ = q_network(s_)


    max_action_of_q_prime = torch.argmax(q_prime_value, dim = 1) # torch.Size([batch])

    if s_.shape == (4, 84, 84):
        _, q_prime_dist_target = target_network(s_.unsqueeze(0))

    else:
        _, q_prime_dist_target = target_network(s_)


    target_distribution = torch.gather(q_prime_dist_target, dim=1, index=max_action_of_q_prime.unsqueeze(1).unsqueeze(2).expand(-1, -1, q_prime_dist_target.size(2))) # torch.Size([batch, 1, 51])

    projection_target_distribution = projection_distribution(atom_size, v_min, v_max, gamma, reward, target_distribution, done, batch_size, spp)

    log_q_dist = torch.log(q_dist_a)

    product = projection_target_distribution*log_q_dist

    product_sum = torch.sum(product, dim= 2)

    loss = -product_sum

    return loss

def projection_distribution(atom_size, v_min, v_max, gamma, reward, distribution, done, batch_size, support):
    
    prob = distribution.reshape(batch_size, atom_size)
    shift_support = torch.clamp(reward + gamma*done*support, v_min, v_max)
    delta =  (v_max - v_min)/(atom_size-1)

    b = (shift_support - v_min)/delta #torch.Size([batch_size, 51])

    u =  b.ceil().to(torch.int64) # torch.Size([batch_size, 51])
    l =  b.floor().to(torch.int64) # torch.Size([batch_size, 51])

    ml = prob*(u - b) #torch.Size([batch_size, 51])
    mu = prob*(b - l) #torch.Size([batch_size, 51])

    out = torch.zeros(batch_size, atom_size)

    for i in range(batch_size):
        if done[i] == 0 and ml[i].sum() == 0 and mu[i].sum() == 0:
            index = ((reward[i] - v_min) / delta).round().to(torch.int64).clamp(0, atom_size - 1).item()
            out[i, index] = 1
        else:
            for j in range(atom_size):
                l_idx = l[i, j].clamp(0, atom_size - 1).item()
                u_idx = u[i, j].clamp(0, atom_size - 1).item()

                out[i, l_idx] += ml[i, j].item()
                if u_idx != l_idx:
                    out[i, u_idx] += mu[i, j].item()


    return out.reshape(batch_size, 1, atom_size)

def training_model(q_network, target_network, atom_size, v_min, v_max, gamma, per, optimizer, step, spp):

    n_step_gamma = math.pow(gamma, T_step) 

    loss_lst = []

    for _ in range(20):

        try:
            transition, index, weight = per.sample(batch)
            s, a, r, s_prime, done_mask = zip(*transition)
        except TypeError as e:
            print(f"Error in unpacking transition: {e}")
            return
        
        s = torch.from_numpy(np.array(s)).type(torch.float)
        s_prime = torch.from_numpy(np.array(s_prime)).type(torch.float)
        a = torch.from_numpy(np.array(a)).to(torch.int64)
        r = torch.from_numpy(np.array(r))
        done_mask = torch.from_numpy(np.array(done_mask)).to(torch.int64)


        calcurate_loss = calcurate_cross_entropy_loss(q_network, target_network, s, a, s_prime, atom_size, v_min, v_max, n_step_gamma, r, done_mask, 32, spp)

        error = torch.abs(calcurate_loss).data.numpy()

        for i in range(batch):
            tree_idx = index[i]
            per.batch_update(tree_idx, error[i])

        loss = (torch.FloatTensor(weight) * calcurate_loss).mean().to(device)

        loss_lst.append(loss.detach().item())

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm = 10)
        optimizer.step()

        q_network.reset_noise()
        target_network.reset_noise()

        soft_update(q_network, target_network)

    #soft_update(q_network, target_network)

    summary.add_scalar('Cross Entropy Loss', max(loss_lst), step)

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    print("SOFT UPDATE")

def Hard_update(net, net_target):

    net_target.load_state_dict(net.state_dict()) # 타겟 네트워크 하드 업데이트
    print("Hard UPDATE")
    return

def save_when_max_score(model):

    full_path = "C:/Users/dlwog/Desktop/RL/history/"
    with torch.no_grad():

        torch.save(model,  full_path + "last_carnival_rainbow_max.pth")
        torch.save(model,  full_path + "last_carnival_rainbow_max.pt")

def periodical_save(model):


    full_path = "C:/Users/dlwog/Desktop/RL/history/"

    with torch.no_grad():

        torch.save(model,  full_path + "last_carnival_rainbow_period.pth")
        torch.save(model,  full_path + "last_carnival_rainbow_period.pt")

def score_100_save(model):

    full_path = "C:/Users/dlwog/Desktop/RL/history/"

    with torch.no_grad():

        torch.save(model,  full_path + "last_carnival_rainbow_100.pth")
        torch.save(model,  full_path + "last_carnival_rainbow_100.pt")

def score_200_save(model):

    full_path = "C:/Users/dlwog/Desktop/RL/history/"

    with torch.no_grad():

        torch.save(model,  full_path + "last_carnival_rainbow_200.pth")
        torch.save(model,  full_path + "last_carnival_rainbow_200.pt")

def score_300_save(model):

    full_path = "C:/Users/dlwog/Desktop/RL/history/"

    with torch.no_grad():

        torch.save(model,  full_path + "last_carnival_rainbow_300.pth")
        torch.save(model,  full_path + "last_carnival_rainbow_300.pt")

def score_400_save(model):

    full_path = "C:/Users/dlwog/Desktop/RL/history/"
    with torch.no_grad():

        torch.save(model,  full_path + "last_carnival_rainbow_400.pth")
        torch.save(model,  full_path + "last_carnival_rainbow_400.pt")

def score_500_save(model):

    full_path = "C:/Users/dlwog/Desktop/RL/history/"

    with torch.no_grad():

        torch.save(model,  full_path + "last_carnival_rainbow_500.pth")
        torch.save(model,  full_path + "last_carnival_rainbow_500.pt")

def main():

    env = gym.make("ALE/Breakout-v5")#, render_mode = "human") # 강화학습의 ENV 생성
    env = FrameStack(env, max_remember_frame) # Markov한 Input 만들어주기 위해서 Frame을 지정한 횟수만큼 쌓기

    #rainbow = Agent(atom_size, v_min, v_max, support, std_variation).to(device) # 강화학습 Agent 생성
    rainbow = torch.load("C:/Users/dlwog/Desktop/RL/history/" + "last_dance_rainbow_period.pt", map_location=device)
    target_rainbow = Agent(atom_size, v_min, v_max, support, std_variation).to(device) # 타겟 네트워크 생성
    target_rainbow.load_state_dict(rainbow .state_dict()) # 타겟 네트워크 복사
    per = Prioritized_Experience_Replay(memory_size) # PER 생성
    n_step_buffer = N_step_buffer(T_step) # N step 버퍼 생성

    step = 0 # step 초기화
    max_score = 0 # 최대 리턴 저장
    past_step = 0 
    #sstteepp = 0

    optimizer = optim.Adam(rainbow.parameters(), lr = learning_rate, eps=1.5e-4) # 옵티마이저 Adam 선언 

    for n_epi in range(0, episode): # 에피소드 진행

        state, data = env.reset() # 환경 첫 시작 세팅
        for _ in range(random.randint(1, 12)): # 시작하자마자 특정 방향으로만 가는 것을 막기 위해서, random 값만큼 멈추기

            state, _, _, _, _ = env.step(1)

        done = False # 게임이 아직 안죽음

        state = preprocess_frame(state) # State 전처리하기

        score = 0 # Return 값 초기화

        initial_lives = data['lives'] # 게임 첫 시작 목숨 초기화 = 5개의 목숨

        #if n_epi % target_update_interval == 0 and n_epi !=0 : # 타겟 네트워크 업데이트 주기가 된다면
            #print("Hard step =", n_epi)
            #Hard_update(rainbow, target_rainbow) # 타겟 네트워크 하드 업데이트

        while not done: # 5번의 목숨이 다 죽지 않는다면

            for _ in range(T_step):

                Q_avg, _ = rainbow.forward(torch.from_numpy(state).float().unsqueeze(0)) # Q의 가중평균과, Q의 확률 분포 받기
                action = torch.argmax(Q_avg, dim = 1).detach().item() # Action 선택
                state_prime, reward, terminated, _, info = env.step(action + 1) # Action에 해당하는 State prime, Reward, Done 관측
                state_prime = preprocess_frame(state_prime)
                current_lives = info['lives'] # Action을 취하고나서 죽었는지, 살았는지 확인하기

                if initial_lives > current_lives: # 만약 죽었다면

                    done_mask = 0 # TD value = R
                    initial_lives = current_lives #현재 목숨을 다음 목숨으로 가져가기
                    env.step(1) # 게임 시작하기 위해 강제 액션

                else: # 아직 살아있다면
                    done_mask = 1 # TD value = Double DQN Target Value 계산
                    initial_lives = current_lives # 현재 목숨으로 계속 진행

                n_step_buffer.put((state, [action], [reward/10], state_prime, [done_mask])) # N Step Buffer에 Transition 저장
                n_step_buffer.put_per(rainbow, target_rainbow, atom_size, v_min, v_max, gamma, 1, per, support) # PER에 Transition 저장

                state = state_prime # 상태 천이

                score += reward # 리턴 계산

                #trigger = False

                # if step % target_update_interval == 0 and step !=0 : # 타겟 네트워크 업데이트 주기가 된다면
                #     print("Hard step =", step)
                #     Hard_update(rainbow, target_rainbow) # 타겟 네트워크 하드 업데이트
                #     step += 1
                #     trigger = True

                if terminated: # 게임이 끝났다면
                    done = True
                    break

                # if trigger:
                    # step += 0

                # else:
                step += 1  # 1 action당 1 step 증가
            survive_time = step - past_step
            if survive_time >= 6000 and not terminated:
                print("게임 오류")
                env.step(1)

            elif survive_time >= 8000 and not terminated:
                past_step = step
                break

            print("For문 끝났다= {}, Done = {}, lives = {}, score = {}".format(action, terminated, current_lives, score))

            if done:
                print("탈출이다")
                past_step = step
                break

        #print("sstteepp =", sstteepp)

        if step >= train_start:# and step % train_interval == 0: # 학습 주기라면
            print("Training =", step)
            training_model(rainbow, target_rainbow, atom_size, v_min, v_max, gamma, per, optimizer, step, support) # 학습 시작하기

        if score >= 500:
            print("Save Score 500 weights")
            score_500_save(rainbow)

        elif 400 <= score < 500:
            print("Save Score 400 weights")
            score_400_save(rainbow)

        elif 300 <= score < 400:
            print("Save Score 300 weights")
            score_300_save(rainbow) 

        elif 200 <= score < 300:
            print("Save Score 200 weights")
            score_200_save(rainbow)

        elif 100 <= score < 200:
            print("Save Score 100 weights")
            score_100_save(rainbow)

        if n_epi % 5 == 0 and n_epi != 0:
            print("Periodical Save")
            periodical_save(rainbow)

        if max_score < score:
            max_score = score
            print("Save Max Score weights")
            save_when_max_score(rainbow)

        print("episode = {}, return = {}, step = {}, max_score = {}".format(n_epi + 1, score, step, max_score))
        summary.add_scalar('BREAK_OUT_SCORE',score, n_epi)

    summary.close()
    env.close()

main()