import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-1):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, batch_size):
        # state = state.reshape([batch_size, self.num_inputs])
        # action = action.reshape([batch_size, self.num_actions])

        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x

class LSTMValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-1):
        super(ValueNetwork, self).__init__()
        self.series_extractor = nn.LSTM(input_size=num_inputs - 1, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.non_series_extractor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.state_extractor = nn.Linear(hidden_size * 2, hidden_size)
        self.value_extractor = nn.Sequential(
            nn.Linear(hidden_size + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action, batch_size):
        train_series, num_samples = state[:, :, :-1], state[:, :, -1]
        loss_emd, _ = self.series_extractor(train_series)
        extracted_loss = loss_emd[:, :, -1]
        extracted_num_samples = self.non_series_extractor(num_samples)

        state_vec = torch.cat([extracted_loss, extracted_num_samples], dim=-1)
        state_vec = self.state_extractor(state_vec)

        state_action_vec = torch.cat([state_vec, action], dim=-1)
        value = self.value_extractor(state_action_vec)

        return value

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-1):
        super(PolicyNetwork, self).__init__()

        # self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)

        # self.linear3e = nn.Linear(hidden_size, hidden_size)
        # self.linear4e = nn.Linear(hidden_size, num_actions)

        # # Free learning epochs
        # freeze_layer(self.linear3e)
        # freeze_layer(self.linear4e)

        # self.linear3i = nn.Linear(hidden_size, hidden_size)
        # self.linear4i = nn.Linear(hidden_size, num_actions)

        # self.linear3n = nn.Linear(hidden_size, num_actions)

        # self.activation = nn.Sigmoid()

        # self.linear4i.weight.data.uniform_(-init_w, init_w)
        # self.linear4i.bias.data.uniform_(-init_w, init_w)

        # self.linear4e.weight.data.uniform_(-init_w, init_w)
        # self.linear4e.bias.data.uniform_(-init_w, init_w)

        # self.linear3n.weight.data.uniform_(-init_w, init_w)
        # self.linear3n.bias.data.uniform_(-init_w, init_w)
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.tanh = nn.Tanh()

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        

    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        x = self.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        action = self.forward(state)
        # action = torch.flatten(action)
        return action.detach().cpu()

class LSTMPolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=1e-1):
        super(LSTMPolicyNetwork, self).__init__()

        self.series_extractor = nn.LSTM(input_size=num_inputs - 1, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.non_series_extractor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.action_extractor = nn.Linear(hidden_size * 2, num_actions)
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        train_series, num_samples = state[:, :, :-1], state[:, :, -1]
        loss_emd, _ = self.series_extractor(train_series)
        extracted_loss = loss_emd[:, :, -1]
        extracted_num_samples = self.non_series_extractor(num_samples)

        context_vec = torch.cat([extracted_loss, extracted_num_samples], dim=-1)
        action = self.tanh(self.action_extractor(context_vec))

        return action
    
    def get_action(self, state):
        action = self.forward(state)
        # action = torch.flatten(action)
        return action.detach().cpu()


if __name__ == '__main__':
    model = ValueNetwork(10 * 4, 10 * 3, 256)

    state = torch.ones(size=[4, 40])
    action = torch.ones(size=[12, 10])

    value = model(state, action, 4)
    print(value.shape)