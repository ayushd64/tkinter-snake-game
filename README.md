# 🐍 Snake AI Evolution: From Manual Play to Convolutional Neural Networks

This repository demonstrates the iterative evolution of an AI agent trained to play the classic Snake Game using various Reinforcement Learning (RL) paradigms, accelerated by NVIDIA CUDA.

## 🛠️ Project Evolution Breakdown

### 1. Manual Play Baseline (`snake.py`)
* **Objective:** Built the structural game engine using `tkinter`. 
* **Control:** Human keyboard input. Serves as the control group.

### 2. Tabular/Basic RL (`rl_snake.py`)
* **Objective:** Explored early state-space logic.
* **Limitation:** Struggled with scaling due to the curse of dimensionality.

### 3. Deep Q-Network (DQN) with Line-of-Sight (`dqn_snake.py`)
* **Objective:** Used a deep neural network with an 11-bit vector representing relative food location and immediate obstacles.
* **Results:** Trained rapidly on CPU/GPU. Hit a hard architectural bottleneck at a high score of **53**. 
* **Limitation:** The agent was "near-sighted"—it could only see 1 step ahead and frequently trapped itself in its own coiled body.

### 4. Convolutional Neural Network (CNN) Engine (`cnn_snake.py`)
* **Objective:** Upgraded the input from an 11-bit vector to a full **2D Spatial Grid Matrix (20x20)**. Utilized `nn.Conv2d` layers to interpret the board as a visual environment.
* **Hardware Acceleration:** Leveraged an NVIDIA RTX 3070 Laptop GPU (CUDA) to compute large batch updates (Size: 256) efficiently.
* **Results:** Solved the "near-sightedness" problem, allowing the agent to evaluate complex shapes and manage tail loops for high-scoring runs.

---

## ⚡ Performance Matrix & Findings

| Model Architecture | Input State Space | Hardware Target | Early Convergence Speed | High Score Ceiling |
| :--- | :--- | :--- | :--- | :--- |
| **DQN (Line-of-Sight)** | 11-Bit Vector | CPU / CUDA | Very Fast | 53 (Hard Wall) |
| **CNN (Computer Vision)**| 20x20 Grid Map | CUDA (RTX 3070) | Slow (High Exploration) | Unlimited (Scalable) |

### Key Takeaway
While the 11-bit DQN model learned faster initially due to its simple input structure, it lacked spatial awareness. The CNN model requires more training games to decipher the 400-pixel grid, but it builds structural awareness necessary to map out path loops without colliding into its tail.

---

## 🚀 How to Run

1. Clone the repository.
2. Install dependencies: `pip install torch numpy`
3. Run the advanced visual agent:
   ```bash
   python src/cnn_snake.py
