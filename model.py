import os
import numpy as np
import tensorflow as tf
import random
import time
import pandas as pd


# Import the Game class from the previous script
from Game import Game
from snake_tensorboard import TensorboardCallback

INPUT_SHAPE = 7
GRID= 8


class SnakeModel(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(14, activation='relu', input_shape=(input_shape,))
        self.dense2 = tf.keras.layers.Dense(20, activation='relu')
        self.dense3 = tf.keras.layers.Dense(24, activation='relu')
        self.dense4 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(3, activation="softmax")  # 5 possible actions

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.output_layer(x)



class SnakeAgent:
    def __init__(self, input_shape, learning_rate=0.0001, load_model=""):
        self.model = SnakeModel(input_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.memory = []
        self.gamma = 0.98  # Discount factor
        self.epsilon = 1  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.last_action = 0
        self.input_shape = (1,input_shape)
        self.last_activation = self.get_activation(tf.zeros((1, input_shape)))
        self.last_action_probs = [0]*3
        self.current_q_values = None
        if load_model:
            dummy_input = tf.zeros((1, input_shape))
            self.model(dummy_input)
            self.model.load_weights(load_model)
            
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 2)  # Random action
        state_array = np.array(state, dtype=np.float32)
        state_tensor = tf.convert_to_tensor([state_array], dtype=tf.float32)
        action_probs = self.model(state_tensor)
        self.last_action_probs = action_probs[0].numpy()
        self.last_activation = self.get_activation(state_tensor)
        return tf.argmax(action_probs[0]).numpy()

    def train(self):
        if len(self.memory) < 32:  # Minimum batch size
            return -1

        # Sample a batch from memory
        batch = random.sample(self.memory, 32)
        
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        # Predict Q-values for current and next states
        state_array = np.array(states, dtype=np.float32)

        states_tensor = tf.convert_to_tensor(state_array, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)

        current_q_values = self.model(states_tensor)
        self.current_q_values = current_q_values

        next_q_values = self.model(next_states_tensor)

        # Compute target Q-values
        target_q_values = current_q_values.numpy()
        for i in range(len(batch)):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        with tf.GradientTape() as tape:
            predictions = self.model(states_tensor)
            loss = tf.keras.losses.MSE(target_q_values, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return  tf.reduce_sum(loss)

    def get_activation(self, inputs):
    
        def convert_activations_for_visualization(tensor_activations):
            converted_activations = []
            for tensor in tensor_activations:
                activation_values = tensor.numpy().squeeze()
                converted_activations.append(activation_values)            
            return converted_activations
        # Apply each layer and collect outputs
        x1 = self.model.dense1(inputs)
        x2 = self.model.dense2(x1)
        x3 = self.model.dense3(x2)
        x4 = self.model.dense4(x3)
        outputs = self.model.output_layer(x4)
        # Create a new model with multiple outputs
        return convert_activations_for_visualization([x1, x2, x3, x4, outputs])

def train_snake_ai(episodes=20000, grid_size=GRID, model='', name=""):
    save_visualise = 500
    snake_ai = SnakeAgent(INPUT_SHAPE, load_model=model)
    tb_callback = TensorboardCallback()

    record = 0
    for episode in range(episodes):
        game = Game(size=grid_size)

        state = game.agent.get_features(game.apple, game.grid)
        total_reward = 0
        steps = 0
        steps_since_apple = 0
        foods_eaten = 0
        episode_losses = []
        while True:
            # Choose and apply action
            action = snake_ai.choose_action(state)
            reward = game.update(action)
            total_reward += reward
            if episode%save_visualise == 0:
                print("Action:", action)
                game.draw_game()
                time.sleep(0.1)

            # Determine if episode is done
            done = (reward == game.HIT_WALL or reward == game.HIT_BODY)
            # Break if game is over
            try:
                next_state = game.agent.get_features(game.apple, game.grid)
            except:
                pass
            if steps_since_apple > 1000:
                print("Braking due to no food has been collected.")
                break
            steps_since_apple += 1
            if reward == game.FOUND_APPLE:
                steps_since_apple = 0
                foods_eaten += 1
            # Store experience
            snake_ai.memory.append((state, action, reward, next_state, done))
            # Train
            loss = snake_ai.train()
            episode_losses.append(loss)
            # Update state
            state = next_state
            steps += 1
            if total_reward > record  and episode > 2:
                record = total_reward+50
                snake_ai.model.save_weights(f'models/snake_model_episode_{episode}_{steps}_{total_reward}_{name}.weights.h5')
            if done:
                break
                # Decay exploration rate
        if snake_ai.epsilon > snake_ai.epsilon_min:
            snake_ai.epsilon *= snake_ai.epsilon_decay
        
        avg_loss = np.mean(episode_losses) if episode_losses else -1
        print(f"Episode {episode+1}: Total Reward = {total_reward:5}, Steps = {steps:5}, Epsilon {snake_ai.epsilon:.4f}, AVG loss {avg_loss:.4f}, Foods eaten: {foods_eaten}")
        
        episode_losses = []  # Reset for next episode
        tb_callback.log_episode(
            episode=episode,
            reward=total_reward,
            length=steps,
            food_eaten=foods_eaten,
            epsilon=snake_ai.epsilon,
            avg_loss=avg_loss,
            q_values=snake_ai.current_q_values,
        )
              # Log model weights occasionally
        if episode % save_visualise == 0:
            tb_callback.log_weights(episode, snake_ai.model)
        if (episode%save_visualise == 0 and episode >0):
            snake_ai.model.save_weights(f'models/snake_model_episode_{episode}_{steps}_{total_reward}_{name}.weights.h5')

    snake_ai.model.save_weights('models/snake_model_final.weights.h5')

def run_all_models(path ='/Users/emil.lindbom/Documents/Other/local/Other/Snake_deep_rf/models/', draw_game=False):
        """Run all models in a folder in the terminal """
        files = os.listdir(path)
        files.sort()
        data = []
        for p, dir, files in os.walk(path):
            # print(f"Running Model: {i} - {file}")
            for file in files:
                model = os.path.join(p,file)
                try:
                    steps, reward = run_snake(model, draw_game=draw_game)
                except:
                    print(f"failed {file}")
                    continue
                data.append([p.split('/')[-1], file, steps, reward])
        return pd.DataFrame(data=data, columns=['dir', 'model', 'steps', 'reward'])

def evaluate_all_models(number_of_tests = 1):
    print("Starting experiments")
    experiments = pd.DataFrame()
    for i, experiment in enumerate(range(number_of_tests)):
        print(f"Index: {i}")
        output = run_all_models()   
        output['experiment'] = experiment
        experiments = pd.concat([experiments, output])
        experiments.to_csv('experiments_output.csv', index=False)

def run_snake(model, draw_game=False, runs=1):
            
            """Simple loop to run the game """
            for run in range(runs):
                game = Game(size=GRID)
                snake_ai = SnakeAgent(input_shape=INPUT_SHAPE, load_model=model)
                # When running we don't want to explore set epsilon to negative
                snake_ai.epsilon = -1
                total_reward = 0
                steps = 0
                max_steps = 10000
                step_since_apple = 0
                while True:
                    state = game.agent.get_features(game.apple, game.grid)
                    action = snake_ai.choose_action(state)
                    reward = game.update(action)
                    total_reward += reward
                    if reward == game.FOUND_APPLE:
                        step_since_apple = 0
                    else:
                        step_since_apple += 1
                    done = (reward == game.HIT_WALL or reward == game.HIT_BODY)
                    if draw_game:
                        time.sleep(0.1)
                        print(f"Model: {model}")
                        game.draw_game()
                        print("")
                    if done or step_since_apple > max_steps:
                        
                        break
                    steps += 1
            return  steps, total_reward
                    
if __name__ == "__main__":
    
    
    train_snake_ai()

    # run_snake('/Users/emil.lindbom/Documents/Other/local/Other/Snake_deep_rf/models/snake_model_episode_3600_43_254.weights.h5', draw_game=True, runs=10)

    # run_all_models(path="models", draw_game=True) 
    # evaluate_all_models(number_of_tests=100)