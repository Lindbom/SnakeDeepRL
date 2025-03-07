import numpy as np

class Agent:
    def __init__(self) -> None:
        self.body = np.array(np.array((4,4))).reshape(1, 2)
        self.action_dict = {
            0 : np.array((0, 1)),   # Right
            1 : np.array((-1, 0)),  # Up
            2 : np.array((0, -1)),  # Left
            3 : np.array((1, 0)),   # Down
            4 : np.array((0, 0)),   # Do nothing
        }
        self.direction = self.action_dict[np.random.randint(0, 4)]
        self.features = [0, 0, 0, 0, 0]

    def update(self, action):
        try:
            action = int(action)
        except:
            action = 4
        self.direction = self.action_dict[action] if action != 4 else self.direction
        body_temp = self.body.copy()
        self.body[0] = self.body[0] + self.direction
        for index in range(len(self.body)-1):
            self.body[index+1] = body_temp[index]

    def get_features(self, apple, grid):
        # Calculate distance to apple in each direction
        dx = apple[0] - self.body[0][0]  # positive if apple is to the right
        dy = apple[1] - self.body[0][1]  # positive if apple is below
        
        # Get grid dimensions for normalization
        grid_height, grid_width = grid.shape
        max_distance = max(grid_height, grid_width)
        
        # Check walls and calculate distance to walls in each direction
        head = self.body[0]
        
        # Wall distances are set to 0 if there's an immediate wall
        # Check for immediate body parts in each direction
        right_pos = tuple(head + self.action_dict[0])
        up_pos = tuple(head + self.action_dict[1])
        left_pos = tuple(head + self.action_dict[2])
        down_pos = tuple(head + self.action_dict[3])

        right_wall = grid[right_pos] == 1 
        up_wall = grid[up_pos] == 1 
        left_wall = grid[left_pos] == 1 
        down_wall = grid[down_pos] == 1 
        
        # Check if body is in each direction and set distances
        body_right = any(np.array_equal(right_pos, b) for b in self.body[1:])
        body_up    =    any(np.array_equal(up_pos, b) for b in self.body[1:]) 
        body_left = any(np.array_equal(left_pos, b) for b in self.body[1:])
        body_down = any(np.array_equal(down_pos, b) for b in self.body[1:])
        
        # Normalize all distances to be between -1 and 1 or 0 and 1
        normalized_dx = dx / max_distance
        normalized_dy = dy / max_distance
        apple_direction = [0, 0, 0, 0]
        if dx > 0:
            apple_direction[0] = 1
        elif dx < 0:
            apple_direction[2] = 1
        if dy > 0:
            apple_direction[3] = 1
        elif dy < 0:
            apple_direction[1] = 1

        return [
            normalized_dx,          # Normalized distance to apple horizontally
            normalized_dy,          # Normalized distance to apple vertically
            right_wall,
            up_wall, 
            left_wall, 
            down_wall,  # Normalized distance to wall below
            body_right,
            body_up,
            body_left,
            body_down,   # Normalized distance to body below
            *apple_direction,
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