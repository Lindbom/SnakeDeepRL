

import numpy as np


class Agent:

    def __init__(self) -> None:
        self.body = np.array(np.array((4,4))).reshape(1, 2)
        self.action_dict = {
            0 : np.array((0, 1)), # Right
            1 : np.array((-1, 0)), # Up
            2 : np.array((0, -1)), # Left
            3 : np.array((1, 0)),  # Down,
            4 : np.array((0, 0)), # Do nothing
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
        
        return [
        apple[0] > self.body[0][0], # apple over snake
        apple[0] < self.body[0][0], # apple under snake
        apple[1] < self.body[0][1], # apple to the left of snake
        apple[1] > self.body[0][1], # apple to the right of snake
        grid[tuple(self.body[0]+ self.action_dict[0])] == 1, # wall to the right
        grid[tuple(self.body[0]+ self.action_dict[1])] == 1, # wall to the up
        grid[tuple(self.body[0]+ self.action_dict[2])] == 1, # wall to the left
        grid[tuple(self.body[0]+ self.action_dict[3])] == 1, # wall to the down
        any(all(self.body[0] + self.action_dict[0] == b) for b in self.body[1:]), # body to the right
        any(all(self.body[0] + self.action_dict[1] == b) for b in self.body[1:]), # body to the up
        any(all(self.body[0] + self.action_dict[2] == b) for b in self.body[1:]), # body to the left
        any(all(self.body[0] + self.action_dict[3] == b) for b in self.body[1:]), # body to the down
        ]

    def add_body_part(self):
        if len(self.body) > 1:
            # (7,8) (6, 8) = 1, 0            
            dir = self.body[-2] - self.body[-1]
        else:
            dir = self.direction
        
        last_body_part = self.body[-1].copy().reshape(1, 2)
        last_body_part = last_body_part+dir*(-1)
        self.body = np.concatenate([self.body, last_body_part], axis=0)



class Game:

    # Rewards
    FOUND_APPLE = 50
    HIT_WaLL = -10
    HIT_BODY = -10
    NOTHING = -0.5

    

    def __init__(self, size: int):
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.grid = np.pad(self.grid, pad_width=1, mode="constant", constant_values=1)
        self.agent = Agent()
        self.apple = self.generate_random(self.agent.body)


    def update(self, action: int):
        self.agent.update(action)
        if all(self.agent.body[0] == self.apple):
            self.agent.add_body_part()
            self.apple = self.generate_random(self.agent.body)
            return self.FOUND_APPLE
        elif self.grid[tuple(self.agent.body[0])] == 1:
            return self.HIT_WaLL
        elif any(all(self.agent.body[0] == body) for body in self.agent.body[1:]):
            return self.HIT_BODY
        else:
            return self.NOTHING

    def generate_random(self, invalid_squares):
        while True:
            random_coord = np.random.randint(1, self.grid.shape[1], 2)
            if (not any(all(random_coord == coord) for coord in invalid_squares)) & (not self.grid[tuple(random_coord)] == 1):
                return random_coord
                

    def draw_game(self):
        game_grid = np.full_like(self.grid, "-", dtype="str")
        for body_part in self.agent.body:
            game_grid[tuple(body_part)] = "S"
        game_grid[tuple(self.apple)] = "A"
        for row in game_grid:
            print("".join(row))


def main():
    
    game = Game(size=10)
    game.draw_game()
    while True:
        feedback = game.update(input(" 0: Right \n 1: Up \n 2: Left \n 3: Down \n 4: Nothing \n Input: "))
        features = game.agent.get_features(grid=game.grid, apple=game.apple)
        
        if feedback == Game.HIT_BODY:
            break
        elif feedback == Game.HIT_WaLL:
            break
        game.draw_game()

if __name__ == "__main__":
    main()

