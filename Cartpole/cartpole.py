import os
import gym
import math
import random
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from itertools import count
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICE"] = "8"

env = gym.make('CartPole-v0').unwrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


plt.ion()


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {'state': [],
                       'action': [],
                       'next_state': [],
                       'reward': []}
        self.position = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory['state'].append(state)
            self.memory['action'].append(action)
            self.memory['next_state'].append(next_state)
            self.memory['reward'].append(reward)
        else:
            self.position = (self.position + 1) % self.capacity
            self.memory['state'][self.position] = state
            self.memory['action'][self.position] = action
            self.memory['next_state'][self.position] = next_state
            self.memory['reward'][self.position] = reward

    def sample(self, batch_size):
        state = random.sample(self.memory['state'], batch_size)
        action = random.sample(self.memory['action'], batch_size)
        next_state = random.sample(self.memory['next_state'], batch_size)
        reward = random.sample(self.memory['reward'], batch_size)
        return state, action, next_state, reward

    def __len__(self):
        return len(self.memory['state'])  # 데이터 갯수


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        layers = []
        layers += self.conv_layer(3, 16)           # 64 * 128 일때
        layers += self.conv_layer(16, 32)
        layers += self.conv_layer(32, 64)
        layers += self.conv_layer(64, 128)
        self.model = nn.Sequential(*layers)
        self.fc1 = nn.Linear(4*8*128, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def conv_layer(self, in_channel, out_channel, kernel_size=5, stride=2, padding=2):
        return [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()]

    def forward(self, x):
        x = self.model(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


'''
screen_width = 600
world_width = env.x_threshold * 2
scale = screen_width / world_width
env.reset()



screen = env.render(mode='rgb_array')
plt.imshow(screen)
print(env.state[0])
'''
screen_width = 600
screen_height = 400

resize = T.Compose([T.ToPILImage(), T.Scale(64, interpolation=Image.CUBIC), T.ToTensor()])


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # middle of the cart


def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # transpose into torch order (DHW)
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(screen_width - view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

    # Strip off the edges so that we have a square image centered on the cart.
    screen = screen[:, :, slice_range]

    # Convert to float, rescale, convert to PyTorch tensor (this doesn't require a copy).
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # Resize and add a batch dimension (BDHW)
    return resize(screen).unsqueeze(0).type(Tensor)


env.reset()

BATCH_SIZE = 5
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()
# test = model(torch.rand(5, 3, 64, 128))

if use_cuda:
    model.cuda()

steps_done = 0


def select_action(model, state, random_select=True):
    global steps_done
    sample = random.random()
    steps_done += 1
    if steps_done < EPS_DECAY:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * (1 - steps_done / EPS_DECAY)
    else:
        eps_threshold = EPS_END

    if sample > eps_threshold or not random_select:
        pred = model(state)
        _, index = torch.max(pred, 1)
        return index.view(-1, 1).float()
    else:
        return torch.randint(1, (state.size()[0], 1))


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    state, action, next_state, reward = memory.sample(BATCH_SIZE)

    batch_state = torch.cat(state, 0)
    batch_action = torch.cat(action, 0)
    batch_reward = torch.cat(reward, 0)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in next_state
                                       if s is not None])
    state_action_values = model(batch_state).gather(1, batch_action.long())
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + batch_reward

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


optimizer = optim.RMSprop(model.parameters())
episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


num_episodes = 100
memory = ReplayMemory(10000)


for i_episode in range(num_episodes):
    # Initialize the environment and state.
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action.
        action = select_action(model, state)
        a = int(action.item())
        _, reward, done, _ = env.step(a)
        reward = Tensor([reward])

        # Observe new state
        last_screen = get_screen()
        current_screen = get_screen()

        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break


print('Complete.')

