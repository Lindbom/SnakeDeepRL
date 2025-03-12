import numpy as np

class Agent:
    def __init__(self) -> None:
        self.body = np.array(np.array((4,4))).reshape(1, 2)
        self.direction = np.array((0, 1))  # Initial direction: right

        self.action_dict = {
            "forward": 0,  # Continue in current direction
            "left" :  1,  # Turn 90 degrees left
            "right" :  2,  # Turn 90 degrees right
        }

    def get_next_direction(self, action):
        # Convert relative action (forward/left/right) to absolute direction
        if action == 0:  # Forward - keep current direction
            return self.direction
        elif action == 1:  # Left - rotate 90 degrees counter-clockwise
            return np.array((-self.direction[1], self.direction[0]))
        elif action == 2:  # Right - rotate 90 degrees clockwise
            return np.array((self.direction[1], -self.direction[0]))

    def update(self, action):
    
        action = int(action)
        self.direction = self.get_next_direction(action)
        body_temp = self.body.copy()
        self.body[0] = self.body[0] + self.direction
        for index in range(len(self.body)-1):
            self.body[index+1] = body_temp[index]

    def get_features(self, apple, grid):
        # Calculate head position and apple position
        head = self.body[0]
        
        # Calculate distance to apple (Manhattan distance)
        dx = apple[0] - head[0]  # positive if apple is to the right
        dy = apple[1] - head[1]  # positive if apple is below
        
        # Get positions in each relative direction
        right_pos = tuple(head + self.get_next_direction(self.action_dict['right']))
        forward_pos = tuple(head + self.get_next_direction(self.action_dict['forward']))
        left_pos = tuple(head + self.get_next_direction(self.action_dict['left']))
        
        # Check for walls or body in each direction
        right_obstacle = grid[right_pos] == 1 or any(np.array_equal(right_pos, b) for b in self.body[1:])
        forward_obstacle = grid[forward_pos] == 1 or any(np.array_equal(forward_pos, b) for b in self.body[1:])
        left_obstacle = grid[left_pos] == 1 or any(np.array_equal(left_pos, b) for b in self.body[1:])
        
        # Convert apple direction to relative directions
        # First, get the current direction vector
        dir_vector = self.get_next_direction(self.action_dict['forward'])
        
        # Apple is on the right side of the snake
        apple_right = (dir_vector[1] * dx - dir_vector[0] * dy) > 0
        
        # Apple is in front of the snake
        apple_forward = (dir_vector[0] * dx + dir_vector[1] * dy) > 0
        
        # Apple is on the left side of the snake
        apple_left = (dir_vector[1] * dx - dir_vector[0] * dy) < 0
        
        # Normalized distances to apple
        grid_size = len(grid)
        distance_to_apple = (abs(dx) + abs(dy)) / (2 * grid_size)
        
        return [
            right_obstacle,    # Danger on right
            forward_obstacle,  # Danger ahead
            left_obstacle,     # Danger on left
            apple_right,       # Apple is to the right
            apple_forward,     # Apple is ahead
            apple_left,        # Apple is to the left
            distance_to_apple  # Normalized distance to apple
        ]
    def add_body_part(self):
        if len(self.body) > 1:
            dir = self.body[-2] - self.body[-1]
        else:
            dir = self.direction
        last_body_part = self.body[-1].copy().reshape(1, 2)
        last_body_part = last_body_part + dir*(-1)
        self.body = np.concatenate([self.body, last_body_part], axis=0)

class Game:
    # Rewards
    FOUND_APPLE = 50
    HIT_WALL = -20
    HIT_BODY = -20
    NOTHING = -0.5

    def __init__(self, size: int):
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.grid = np.pad(self.grid, pad_width=1, mode="constant", constant_values=1)
        self.agent = Agent()
        self.apple = self.generate_random(self.agent.body)

    def update(self, action: int):
        
        prev_dir = self.agent.direction.copy()
        self.agent.update(action)

        if all(self.agent.body[0] == self.apple):
            self.agent.add_body_part()
            self.apple = self.generate_random(self.agent.body)

            return self.FOUND_APPLE
        elif self.grid[tuple(self.agent.body[0])] == 1:
            return self.HIT_WALL
        elif any(all(self.agent.body[0] == body) for body in self.agent.body[1:]):
            return self.HIT_BODY
        elif np.array_equal(self.agent.direction, -1 * prev_dir) and len(self.agent.body) > 1:
            return self.HIT_BODY
        else:
            return self.NOTHING

    def generate_random(self, invalid_squares):
        while True:
            random_coord = np.random.randint(1, self.grid.shape[1], 2)
            if (not any(all(random_coord == coord) for coord in invalid_squares)) & (not self.grid[tuple(random_coord)] == 1):
                return random_coord

    def draw_game(self):
        game_grid = np.full_like(self.grid, ".", dtype="str")
        game_grid[:, -1] = 'x'
        game_grid[:, 0] = 'x'
        game_grid[0, :] = 'x'
        game_grid[-1, :] = 'x'
        for i, body_part in enumerate(self.agent.body):
            if i == 0:
                game_grid[tuple(body_part)] = "H"
            else:
                game_grid[tuple(body_part)] = "S"
        game_grid[tuple(self.apple)] = "A"
        for row in game_grid:
            
            print("".join(row))
        print("")

# Optional: Main function for manual play
def main():
    game = Game(size=10)
    game.draw_game()
    while True:
        feedback = game.update(input(" 0: Right \n 1: Up \n 2: Left \n 3: Down \n 4: Nothing \n Input: "))
        features = game.agent.get_features(grid=game.grid, apple=game.apple)


        if feedback == Game.HIT_BODY:
            break
        elif feedback == Game.HIT_WALL:
            break
        game.draw_game()

if __name__ == "__main__":
    main()