import tkinter as tk
import random

# --- GAME CONSTANTS ---
GAME_WIDTH = 400
GAME_HEIGHT = 400
SPACE_SIZE = 20 # This means each square (snake part and food) is 20x20 pixels
SNAKE_COLOR = "green"
FOOD_COLOR = "red"


alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.997
min_epsilon = 0.01
q_table = {}
ACTIONS = ["up", "down", "left", "right"]

# --- GAME STATE VARIABLES ---
direction = "right" # Initial direction of the snake
snake_coordinates = [[200, 200]] # List to hold the coordinates of the snake parts
food_x = 0
food_y = 0
score = 0
games_played = 0
high_score = 0


# --- INITIALIZE WINDOW ---
window = tk.Tk()
window.title("Agentic AI Snake Game")
window.resizable(False, False) # Prevent resizing to prevent our grid intact


# Create a canvas widget with a black background
# This act as our game board where the snake and food will be drawn
canvas = tk.Canvas(window, bg="black", width=GAME_WIDTH, height=GAME_HEIGHT)
canvas.pack() # Add the canvas to the window

score_text = canvas.create_text(
    GAME_WIDTH / 2,
    30,
    text=f"Games: {games_played} Score: {score} Best: {high_score}",
    fill="white",
    font=("Arial", 14, "bold")
)



def spawn_food():
    global food_x, food_y
    canvas.delete("food") # Remove the previous food from the canvas
    columns = int(GAME_WIDTH / SPACE_SIZE) # Calculate how many columns we have
    rows = int(GAME_HEIGHT / SPACE_SIZE) # Calculate how many rows we have
    food_x = random.randint(0, columns - 1) * SPACE_SIZE # Random x coordinate for food
    food_y = random.randint(0, rows - 1) * SPACE_SIZE # Random y coordinate for food
    canvas.create_rectangle(
        food_x,
        food_y,
        food_x + SPACE_SIZE,
        food_y + SPACE_SIZE,
        fill=FOOD_COLOR,
        tag="food"
    ) # Draw the food on the canvas at the random coordinates


def is_collision(x, y):
    if x < 0 or x >= GAME_WIDTH or y < 0 or y >= GAME_HEIGHT:
        return True # Collision with wall
    if [x, y] in snake_coordinates[:-1]:
        return True # Collision with itself
    return False


def get_state():
    head_x, head_y = snake_coordinates[0]

    point_l = (head_x - SPACE_SIZE, head_y)
    point_r = (head_x + SPACE_SIZE, head_y)
    point_u = (head_x, head_y - SPACE_SIZE)
    point_d = (head_x, head_y + SPACE_SIZE)

    dir_l = direction == "left"
    dir_r = direction == "right"
    dir_u = direction == "up"
    dir_d = direction == "down"

    state = (
        (dir_r and is_collision(*point_r)) or
        (dir_l and is_collision(*point_l)) or
        (dir_u and is_collision(*point_u)) or
        (dir_d and is_collision(*point_d)),

        (dir_u and is_collision(*point_r)) or
        (dir_d and is_collision(*point_l)) or
        (dir_l and is_collision(*point_u)) or
        (dir_r and is_collision(*point_d)),

        (dir_u and is_collision(*point_l)) or
        (dir_d and is_collision(*point_r)) or
        (dir_l and is_collision(*point_d)) or
        (dir_r and is_collision(*point_u)),

        dir_u, dir_d, dir_l, dir_r,

        food_y < head_y,
        food_y > head_y,
        food_x < head_x,
        food_x > head_x
    )

    return tuple(int(s) for s in state)



def choose_action(state):
    if state not in q_table:
        q_table[state] = [0.0, 0.0, 0.0, 0.0] # Initialize Q-values for new state
    
    if random.random() < epsilon:
        return random.randint(0, 3) # Explore: choose a random action
    else:
        state_scores = q_table[state]
        max_score = max(state_scores)
        best_actions = [i for i, score in enumerate(state_scores) if score == max_score]
        return random.choice(best_actions) # Exploit: choose the best action (randomly among ties)



def reset_game():
    global snake_coordinates, direction, score, games_played, epsilon, high_score

    games_played += 1

    if score > high_score:
        high_score = score

    score = 0
    direction = "right"
    snake_coordinates = [[200, 200]]

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
    
    canvas.delete("all") # Clear the canvas

    global score_text
    score_text = canvas.create_text(
        GAME_WIDTH / 2,
        30,
        text=f"Games: {games_played} Score: {score} Best: {high_score} Exploration: {epsilon:.2f}",
        fill="white",
        font=("Arial", 12, "bold")
    )
    spawn_food()


def game_step():
    global direction, score

    state_old = get_state()

    action_index = choose_action(state_old)
    direction = ACTIONS[action_index]

    x, y = snake_coordinates[0]
    if direction == "up": y -= SPACE_SIZE
    elif direction == "down": y += SPACE_SIZE
    elif direction == "left": x -= SPACE_SIZE
    elif direction == "right": x += SPACE_SIZE

    reward = 0.0
    game_over_round = False

    if is_collision(x, y):
        reward = -10.0
        game_over_round = True
    elif x == food_x and y == food_y:
        reward = 10.0
        score += 1
        snake_coordinates.insert(0, [x, y]) # Add new head to the snake
        spawn_food()
    else:
        head_x, head_y = snake_coordinates[0]
        old_dist = abs(head_x - food_x) + abs(head_y - food_y)
        new_dist = abs(x - food_x) + abs(y - food_y)
        reward = 0.1 if new_dist < old_dist else -0.2

        snake_coordinates.insert(0, [x, y]) # Move snake by adding new head
        del snake_coordinates[-1] # Remove tail
    
    state_new = get_state()
    if state_new not in q_table:
        q_table[state_new] = [0.0, 0.0, 0.0, 0.0] # Initialize Q-values for new state
    
    old_q_value = q_table[state_old][action_index]
    max_future_q = max(q_table[state_new])

    new_q_value = old_q_value + alpha * (reward + gamma * max_future_q - old_q_value)

    q_table[state_old][action_index] = new_q_value


    canvas.delete("snake") # Remove the previous snake from the canvas
    for segment in snake_coordinates:
        canvas.create_rectangle(
            segment[0],
            segment[1],
            segment[0] + SPACE_SIZE,
            segment[1] + SPACE_SIZE,
            fill=SNAKE_COLOR,
            tag="snake"
        ) # Draw the snake on the canvas

    canvas.itemconfigure(score_text, text=f"Games: {games_played} Score: {score} Best: {high_score} Exploration: {epsilon:.2f}") # Update the score display on the canvas

    if game_over_round:
        reset_game()
    
    window.after(5, game_step) # Schedule the next game step after 100 milliseconds


spawn_food()
game_step()
window.mainloop()
