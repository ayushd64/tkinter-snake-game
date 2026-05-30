import tkinter as tk
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- GAME CONSTANTS ---
GAME_WIDTH = 400
GAME_HEIGHT = 400
SPACE_SIZE = 20
GRID_SIZE = GAME_WIDTH // SPACE_SIZE # 20x20 Grid
SNAKE_COLOR = "magenta"  # New color for our advanced CNN agent
FOOD_COLOR = "red"
SPEED = 1              # Absolute maximum training speed

# --- CNN HYPERPARAMETERS ---
BATCH_SIZE = 256
LR = 0.0005           # Slightly lower learning rate for CNN stability
GAMMA = 0.95          # Look further into the future
EPSILON_START = 1.0
EPSILON_DECAY = 0.996 # Slower decay because CNN requires more patterns to learn
EPSILON_MIN = 0.02

# --- CONVOLUTIONAL NEURAL NETWORK (CNN) BRAIN ---
class Convolutional_QNet(nn.Module):
    def __init__(self, grid_size, output_size):
        super().__init__()
        # Input channel is 1 (the 2D grid matrix)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # Extracts edges/walls
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Extracts body curves
        
        # Calculate flattened size after convolutions (64 channels * 20 * 20 grid)
        self.fc1 = nn.Linear(64 * grid_size * grid_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        # x shape expected: (Batch, Channels, Height, Width) -> (B, 1, 20, 20)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten the 2D grid into a 1D vector
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- EXPERIENCE REPLAY MEMORY ---
class ReplayMemory:
    def __init__(self, capacity=50000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- ENVIRONMENT & AGENT MAIN ENGINE ---
class SnakeGameCNN:
    def __init__(self):
        self.direction = "right"
        self.snake_coordinates = [[200, 200]]
        self.food_x = 0
        self.food_y = 0
        self.score = 0
        self.games_played = 0
        self.high_score = 0
        self.epsilon = EPSILON_START

        # Setup GUI Window
        self.window = tk.Tk()
        self.window.title("Convolutional Neural Network (CNN) Snake")
        self.window.resizable(False, False)
        self.canvas = tk.Canvas(self.window, bg="black", width=GAME_WIDTH, height=GAME_HEIGHT)
        self.canvas.pack()
        
        self.score_text = self.canvas.create_text(
            GAME_WIDTH / 2, 30, 
            text="Initializing CNN Engine...", fill="white", font=("Arial", 11, "bold")
        )

        # Device Selection (Your RTX 3070!)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 CNN Engine acceleration locked on: {self.device.type.upper()}")

        # Initialize AI Modules
        self.model = Convolutional_QNet(GRID_SIZE, 4).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory()
        self.ACTIONS = ["up", "down", "left", "right"]

        self.spawn_food()

    def spawn_food(self):
        self.canvas.delete("food")
        self.food_x = random.randint(0, GRID_SIZE - 1) * SPACE_SIZE
        self.food_y = random.randint(0, GRID_SIZE - 1) * SPACE_SIZE
        self.canvas.create_rectangle(self.food_x, self.food_y, self.food_x + SPACE_SIZE, self.food_y + SPACE_SIZE, fill=FOOD_COLOR, tag="food")

    def is_collision(self, x, y):
        if x < 0 or x >= GAME_WIDTH or y < 0 or y >= GAME_HEIGHT:
            return True
        if [x, y] in self.snake_coordinates[:-1]:
            return True
        return False

    def get_grid_state(self):
        """ Generates a full 2D spatial grid map of the game layout """
        # Start with an empty grid representation
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Populate Snake Body segments (value: 0.5)
        for segment in self.snake_coordinates[1:]:
            gx, gy = segment[0] // SPACE_SIZE, segment[1] // SPACE_SIZE
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                grid[gy][gx] = 0.5

        # Populate Snake Head (value: 1.0)
        hx, hy = self.snake_coordinates[0][0] // SPACE_SIZE, self.snake_coordinates[0][1] // SPACE_SIZE
        if 0 <= hx < GRID_SIZE and 0 <= hy < GRID_SIZE:
            grid[hy][hx] = 1.0

        # Populate Food Item (value: -1.0)
        fx, fy = self.food_x // SPACE_SIZE, self.food_y // SPACE_SIZE
        if 0 <= fx < GRID_SIZE and 0 <= fy < GRID_SIZE:
            grid[fy][fx] = -1.0

        return grid # Returns a clean 20x20 matrix map array

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            # Reshape grid array to fit PyTorch CNN input standards: (Batch=1, Channel=1, H=20, W=20)
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prediction = self.model(state_tensor)
            return torch.argmax(prediction).item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        mini_batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Reshape to array formats matching CNN structures (Batch, 1, 20, 20)
        states = torch.tensor(np.array(states), dtype=torch.float).unsqueeze(1).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).float().to(self.device)

        # Deep Reinforcement Q-Value Predictions
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        # Backpropagation optimization cycle
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset_game(self):
        self.games_played += 1
        if self.score > self.high_score:
            self.high_score = self.score
            print(f"👁️ CNN BREAKTHROUGH! | Game: {self.games_played:<4} | High Score: {self.high_score:<3} | Eps: {self.epsilon:.2f}")
            
        self.score = 0
        self.direction = "right"
        self.snake_coordinates = [[200, 200]]
        
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
            
        self.canvas.delete("all")
        self.score_text = self.canvas.create_text(
            GAME_WIDTH / 2, 30, 
            text=f"Game: {self.games_played}  Score: {self.score}  Best: {self.high_score}  Eps: {self.epsilon:.2f}", 
            fill="white", font=("Arial", 11, "bold")
        )
        self.spawn_food()

    def play_step(self):
        state_old = self.get_grid_state()
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
            reward = 12.0 # Higher payout for food captures
            self.score += 1
            self.snake_coordinates.insert(0, [x, y])
            self.spawn_food()
        else:
            # Dynamic alignment reward calculations
            head_x, head_y = self.snake_coordinates[0]
            old_dist = abs(head_x - self.food_x) + abs(head_y - self.food_y)
            new_dist = abs(x - self.food_x) + abs(y - self.food_y)
            reward = 0.2 if new_dist < old_dist else -0.4
            
            self.snake_coordinates.insert(0, [x, y])
            self.snake_coordinates.pop()

        state_new = self.get_grid_state()
        self.memory.push(state_old, action_index, reward, state_new, done)
        self.train_step()

        # Render graphics layout
        self.canvas.delete("snake")
        for segment in self.snake_coordinates:
            self.canvas.create_rectangle(segment[0], segment[1], segment[0] + SPACE_SIZE, segment[1] + SPACE_SIZE, fill=SNAKE_COLOR, tag="snake")

        self.canvas.itemconfig(
            self.score_text, 
            text=f"Game: {self.games_played}  Score: {self.score}  Best: {self.high_score}  Eps: {self.epsilon:.2f}"
        )

        if done:
            self.reset_game()

        self.window.after(SPEED, self.play_step)

# --- RUN COMPUTATIONAL ENGINE ---
if __name__ == "__main__":
    game = SnakeGameCNN()
    game.play_step()
    game.window.mainloop()

