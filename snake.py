import tkinter as tk
import random

# --- GAME CONSTANTS ---
GAME_WIDTH = 600
GAME_HEIGHT = 600
SPACE_SIZE = 20 # This means each square (snake part and food) is 20x20 pixels
SNAKE_COLOR = "green"
FOOD_COLOR = "red"
SPEED = 100 # Game speed in milliseconds (lower number  = faster snake)


# --- GAME STATE VARIABLES ---
direction = "right" # Initial direction of the snake
snake_coordinates = [[0, 0]] # List to hold the coordinates of the snake parts
food_x = 0
food_y = 0
score = 0


# --- INITIALIZE WINDOW ---
window = tk.Tk()
window.title("Snake Game")
window.resizable(False, False) # Prevent resizing to prevent our grid intact


# Create a canvas widget with a black background
# This act as our game board where the snake and food will be drawn
canvas = tk.Canvas(window, bg="black", width=GAME_WIDTH, height=GAME_HEIGHT)
canvas.pack() # Add the canvas to the window


score_text = canvas.create_text(
    GAME_WIDTH / 2,
    30,
    text=f"Score: {score}",
    fill="white",
    font=("Arial", 20, "bold")
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



def change_direction(new_direction):
    global direction

    # Prevent the snake from reversing on itself
    if new_direction == "left" and direction != "right":
        direction = new_direction
    elif new_direction == "right" and direction != "left":
        direction = new_direction
    elif new_direction == "up" and direction != "down":
        direction = new_direction
    elif new_direction == "down" and direction != "up":
        direction = new_direction



def check_collisions():
    x, y = snake_coordinates[0] # Get the head of the snake

    # Check if the snake has collided with the walls
    if x < 0 or x >= GAME_WIDTH:
        return True
    elif y < 0 or y >= GAME_HEIGHT:
        return True

    # Check if the snake has collided with itself
    for body_segment in snake_coordinates[1:]:
        if x == body_segment[0] and y == body_segment[1]:
            return True

    return False # No collision detected



def game_over():
    canvas.delete("all") # Clear the canvas
    canvas.create_text(
        GAME_WIDTH / 2,
        GAME_HEIGHT / 2,
        font=("Arial", 40, "bold"),
        fill="white",
        text="GAME OVER!"
    ) # Display "GAME OVER!" message in the center of the canvas

    canvas.create_text(
        GAME_WIDTH / 2,
        GAME_HEIGHT / 2 + 60,
        font=("Arial", 20, "bold"),
        fill="white",
        text=f"Final Score: {score}"
    ) # Display the final score below the game over message



def next_turn():
    global direction, food_x, food_y, score

    x, y = snake_coordinates[0]

    if direction == "up":
        y -= SPACE_SIZE
    elif direction == "down":
        y += SPACE_SIZE
    elif direction == "left":
        x -= SPACE_SIZE
    elif direction == "right":
        x += SPACE_SIZE

    snake_coordinates.insert(0, (x, y)) # Add new head position to the snake coordinates list
    
    if x == food_x and y == food_y: # Check if the snake has eaten the food
        score += 1 # Increase the score
        canvas.itemconfigure(score_text, text=f"Score: {score}") # Update the score display on the canvas
        spawn_food() # Spawn new food
    else:
        del snake_coordinates[-1]
    
    if check_collisions(): # Check for collisions after moving
        game_over() # End the game if a collision is detected
    else:
        canvas.delete("snake") # Remove the previous snake from the canvas

        for segment in snake_coordinates:
            canvas.create_rectangle(
                segment[0],
                segment[1],
                segment[0] + SPACE_SIZE,
                segment[1] + SPACE_SIZE,
                fill=SNAKE_COLOR,
                tag="snake"
            ) # Draw the snake on the canvas at its new position

        window.after(SPEED, next_turn) # Schedule the next turn after a delay defined by SPEED

window.bind("<Left>", lambda event: change_direction("left")) # Bind left arrow key to change direction to left
window.bind("<Right>", lambda event: change_direction("right")) # Bind right arrow key to change direction to right
window.bind("<Up>", lambda event: change_direction("up")) # Bind up arrow key to change direction to up
window.bind("<Down>", lambda event: change_direction("down")) # Bind down arrow key to change direction to down

spawn_food() # Spawn the first food item on the canvas
next_turn() # Start the game loop by calling the next_turn function for the first time

window.mainloop() # Start the Tkinter event loop to display the window and keep it open



 
