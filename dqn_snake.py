import tkinter as tk
import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim

GAME_WOIDTH = 400
GAME_HEIGHT = 400
SPACE_SIZE = 20
SNAKE_COLOR = "cyan"
FOOD_COLOR = "red"
SPEED = 1

BATCH_SIZE = 256
LR = 0.001
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_DECAY = 0.998
EPSILON_MIN = 0.02


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training accelerated on: {device.type.upper()}") 



class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class ReplayMemory:
    def __init__(self, capacity=50000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

class SnakeGameAI:
    def __init__(self):
        self.direction = "right"
        self.snake_coordinates = [[200, 200]]
        self.food_x = 0
        self.food_y = 0
        self.score = 0
        self.games_played = 0
        self.high_score = 0
        self.epsilon = EPSILON_START

        self.window = tk.Tk()
        self.window.title("Deep Q-Network (DQN) Snake Game")
        self.window.resizable(False, False)

        self.canvas = tk.Canvas(self.window, bg="black", width=GAME_WOIDTH, height=GAME_HEIGHT)
        self.canvas.pack()

        self.score_text = self.canvas.create_text(
            GAME_WOIDTH / 2,
            30,
            text="Initializing Model...",
            fill="white",
            font=("Arial", 11, "bold")
        )

        self.model = Linear_QNet(11, 256, 4).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory()
        self.ACTIONS = ["up", "down", "left", "right"]

        self.spawn_food()
    


    def spawn_food(self):
        self.canvas.delete("food")
        columns = int(GAME_WOIDTH / SPACE_SIZE)
        rows = int(GAME_HEIGHT / SPACE_SIZE)
        self.food_x = random.randint(0, columns - 1) * SPACE_SIZE
        self.food_y = random.randint(0, rows - 1) * SPACE_SIZE
        self.canvas.create_rectangle(
            self.food_x,
            self.food_y,
            self.food_x + SPACE_SIZE,
            self.food_y + SPACE_SIZE,
            fill=FOOD_COLOR,
            tag="food"
        )
    


    def is_collision(self, x, y):
        if x < 0 or x >= GAME_WOIDTH or y < 0 or y >= GAME_HEIGHT:
            return True
        if [x, y] in self.snake_coordinates[:-1]:
            return True
        return False



    def get_state(self):
        head_x, head_y = self.snake_coordinates[0]

        point_l = (head_x - SPACE_SIZE, head_y)
        point_r = (head_x + SPACE_SIZE, head_y)
        point_u = (head_x, head_y - SPACE_SIZE)
        point_d = (head_x, head_y + SPACE_SIZE)

        dir_l = self.direction == "left"
        dir_r = self.direction == "right"
        dir_u = self.direction == "up"
        dir_d = self.direction == "down"

        state = [
            (dir_r and self.is_collision(*point_r)) or
            (dir_l and self.is_collision(*point_l)) or
            (dir_u and self.is_collision(*point_u)) or
            (dir_d and self.is_collision(*point_d)),

            (dir_u and self.is_collision(*point_r)) or
            (dir_d and self.is_collision(*point_l)) or
            (dir_l and self.is_collision(*point_u)) or
            (dir_r and self.is_collision(*point_d)),

            (dir_u and self.is_collision(*point_l)) or
            (dir_d and self.is_collision(*point_r)) or
            (dir_l and self.is_collision(*point_d)) or
            (dir_r and self.is_collision(*point_u)),

            dir_u, dir_d, dir_l, dir_r,
            self.food_y < head_y,
            self.food_y > head_y,
            self.food_x < head_x,
            self.food_x > head_x
        ]

        return tuple(int(s) for s in state)
    


    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state_tensor)
            return torch.argmax(prediction).item()
    

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        min_batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*min_batch)

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).float().to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def reset_game(self):
        self.games_played += 1
        if self.score > self.high_score:
            self.high_score = self.score
            print(f"🔥 NEW BEST! | Game: {self.games_played:<4} | Score: {self.score:<3} | High Score: {self.high_score:<3} | Eps: {self.epsilon:.2f}")


        self.score = 0
        self.direction = "right"
        self.snake_coordinates = [[200, 200]]

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY\

        self.canvas.delete("all")
        self.score_text = self.canvas.create_text(
            GAME_WOIDTH / 2,
            30,
            text=f"Game: {self.games_played} | Score: {self.score} | Best: {self.high_score} | Eps: {self.epsilon:.2f}",
            fill="white",
            font=("Arial", 11, "bold")
        )
        self.spawn_food()
    

    def play_step(self):
        state_old = self.get_state()
        action_index = self.choose_action(state_old)
        self.direction = self.ACTIONS[action_index]

        x, y = self.snake_coordinates[0]
        if self.direction == "up": y -= SPACE_SIZE
        elif self.direction == "down": y += SPACE_SIZE
        elif self.direction == "left": x -= SPACE_SIZE
        elif self.direction == "right": x += SPACE_SIZE

        reward = 0.0
        done = False

        if self.is_collision(x, y):
            reward = -10.0
            done = True
        elif x == self.food_x and y == self.food_y:
            reward = 10.0
            self.score += 1
            self.snake_coordinates.insert(0, [x, y])
            self.spawn_food()
        else:
            head_x, head_y = self.snake_coordinates[0]
            old_dist = abs(head_x - self.food_x) + abs(head_y - self.food_y)
            new_dist = abs(x - self.food_x) + abs(y - self.food_y)
            reward = 0.2 if new_dist < old_dist else -0.4

            self.snake_coordinates.insert(0, [x, y])
            self.snake_coordinates.pop()

        state_new = self.get_state()
        self.memory.push(state_old, action_index, reward, state_new, done)
        self.train_step()

        self.canvas.delete("snake")
        for segment in self.snake_coordinates:
            self.canvas.create_rectangle(
                segment[0],
                segment[1],
                segment[0] + SPACE_SIZE,
                segment[1] + SPACE_SIZE,
                fill=SNAKE_COLOR,
                tag="snake"
            )
        
        self.canvas.itemconfig(
            self.score_text,
            text=f"Game: {self.games_played} | Score: {self.score} | Best: {self.high_score} | Eps: {self.epsilon:.2f}"
        )

        if done:
            self.reset_game()
        
        self.window.after(SPEED, self.play_step)


if __name__ == "__main__":
    game = SnakeGameAI()
    game.play_step()
    game.window.mainloop()
