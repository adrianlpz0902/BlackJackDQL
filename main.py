import random
import torch
import numpy as np
import pandas as pd
from collections import deque
from Game import BlackjackGame

MEMORY_SIZE = 100_000
BATCH_SIZE = 64


class Deep_Q_Network(torch.nn.Module):
    def __init__(self, lr, input_size, hidden_size_one, hidden_size_two, actions):
        super(Deep_Q_Network, self).__init__()
        self.linear1 = torch.nn.Linear(*input_size, hidden_size_one)
        self.linear2 = torch.nn.Linear(hidden_size_one, hidden_size_two)
        self.linear3 = torch.nn.Linear(hidden_size_two, actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()

    def forward(self, state):
        x = torch.nn.functional.relu(self.linear1(state))
        x = torch.nn.functional.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions


class Agent():
    def __init__(self, input_size, actions):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decrement = 5e-4
        self.action_space = list(range(actions))
        self.mem_size = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.memory = deque(maxlen=self.mem_size)
        self.q_network = Deep_Q_Network(lr=0.001, actions=actions, input_size=input_size, hidden_size_one=256,
                                        hidden_size_two=256)
        self.target_network = Deep_Q_Network(lr=0.001, actions=actions, input_size=input_size, hidden_size_one=256,
                                             hidden_size_two=256)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, input_state):
        if np.random.random() > self.epsilon:
            state = torch.tensor([input_state], dtype=torch.float32)
            actions = self.q_network.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        sample = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        self.q_network.optimizer.zero_grad()

        states = torch.tensor(states, dtype=torch.float32)
        new_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        actions = torch.tensor(actions, dtype=torch.long)

        q_values = self.q_network.forward(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_actions = torch.argmax(self.q_network.forward(new_states), dim=1, keepdim=True)
        next_q_values = self.target_network.forward(new_states).gather(1, next_actions).squeeze()
        next_q_values[dones] = 0.0
        target_q_values = rewards + self.gamma * next_q_values

        loss = self.q_network.loss(target_q_values, q_values)
        loss.backward()
        self.q_network.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrement
        else:
            self.epsilon = self.epsilon_min

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


def Train_BlackJack_DQL(n_games):
    stats = []
    agent = Agent(input_size=[2], actions=2)
    game = BlackjackGame()

    wins, losses, ties = 0, 0, 0

    for i in range(1, n_games + 1):
        score = 0
        done = False
        state = game.get_game_state()
        print(f"\n=== Starting Game {i} ===")  # 🔍 Debugging Print

        while not done:
            print(f"Current State: {state}")  # 🔍 See the current state
            action = agent.choose_action(state)
            move = "Hit" if action == 1 else "Stand"
            print(f"Agent Chose: {move}")  # 🔍 See what the agent does

            reward, done, tie = game.play_round(move)
            next_state = game.get_game_state()
            print(f"Reward: {reward}, Done: {done}, Tie: {tie}")  # 🔍 Check game result

            score += reward
            agent.store_memory(state, action, reward, next_state, done)
            agent.train()
            state = next_state

        game.start_new_round()
        print(f"Final Score: {score}")  # 🔍 Check final game score

        if i % 10 == 0:
            agent.update_target_network()

        if score == 100 and tie:
            ties += 1
        elif score == 100 and not tie:
            wins += 1
        else:
            losses += 1

        stats.append([i, (wins / i) * 100, (losses / i) * 100, (ties / i) * 100])

        if i % 1000 == 0 and i > 0:
            print(
                f'Game #: {i} Win(%): {wins / i * 100:.2f} Loss(%): {losses / i * 100:.2f} Ties(%): {ties / i * 100:.2f}')

    df = pd.DataFrame(stats, columns=["Game", "Win%", "Loss%", "Tie%"])
    return df


def main():
    n_games = int(input("How many games do you want to simulate: "))
    df = Train_BlackJack_DQL(n_games)
    print(df.tail())  # Show last few results


if __name__ == '__main__':
    main()
