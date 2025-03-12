import io
import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

class TensorboardCallback:
    def __init__(self, log_dir=None):
        """Initialize TensorBoard logger for the Snake DQN agent.
        
        Args:
            log_dir: Directory to save logs. If None, creates a timestamped directory.
        """
        if log_dir is None:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join('logs', f'snake_dqn_{current_time}')
            
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.food_collected = []
        self.max_score = 0
        
        # Create log directories
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"TensorBoard logs will be saved to: {self.log_dir}")
        print(f"To view, run: tensorboard --logdir={os.path.dirname(self.log_dir)}")

    def log_episode(self, episode, reward, length, food_eaten, epsilon, avg_loss=None, q_values=None):
        """Log episode metrics to TensorBoard.
        
        Args:
            episode: Current episode number
            reward: Total reward for the episode
            length: Number of steps in the episode
            food_eaten: Number of food items collected
            epsilon: Current exploration rate
            avg_loss: Average loss for the episode (if available)
            q_values: Q-values from recent prediction (if available)
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if avg_loss is not None:
            self.episode_losses.append(avg_loss)
        self.food_collected.append(food_eaten)
        
        # Update max score
        if food_eaten > self.max_score:
            self.max_score = food_eaten
            
        # Calculate moving averages
        window_size = min(100, len(self.episode_rewards))
        avg_reward = np.mean(self.episode_rewards[-window_size:])
        avg_length = np.mean(self.episode_lengths[-window_size:])
        avg_food = np.mean(self.food_collected[-window_size:])
        
        with self.writer.as_default():
            # Episode metrics
            tf.summary.scalar('Metrics/Reward', reward, step=episode)
            tf.summary.scalar('Metrics/Episode Length', length, step=episode)
            tf.summary.scalar('Metrics/Food Collected', food_eaten, step=episode)
            tf.summary.scalar('Metrics/Max Score', self.max_score, step=episode)
            
            # Moving averages
            tf.summary.scalar('Average/Reward (100 ep)', avg_reward, step=episode)
            tf.summary.scalar('Average/Episode Length (100 ep)', avg_length, step=episode)
            tf.summary.scalar('Average/Food Collected (100 ep)', avg_food, step=episode)
            
            # Training parameters
            tf.summary.scalar('Training/Epsilon', epsilon, step=episode)
            if avg_loss is not None:
                tf.summary.scalar('Training/Loss', avg_loss, step=episode)
                
            # Q-values distribution
            if q_values is not None:
                tf.summary.histogram('Q-values/Distribution', q_values, step=episode)
                tf.summary.scalar('Q-values/Max', np.max(q_values), step=episode)
                tf.summary.scalar('Q-values/Min', np.min(q_values), step=episode)
                tf.summary.scalar('Q-values/Mean', np.mean(q_values), step=episode)
                
            # Every 1000 episodes, log the reward distribution
            if episode % 1000 == 0 and episode > 0:
                fig = plt.figure(figsize=(10, 6))
                plt.hist(self.episode_rewards[-1000:], bins=20)
                plt.title(f'Reward Distribution (Episodes {episode-999}-{episode})')
                plt.xlabel('Reward')
                plt.ylabel('Frequency')
                
                # Convert to image and log
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)
                tf.summary.image('Reward Distribution', image, step=episode)

    def log_weights(self, episode, model):
        """Log model weight histograms.
        """
        with self.writer.as_default():
            for layer in model.layers:
                for weight in layer.weights:
                    tf.summary.histogram(f"weights/{layer.name}/{weight.name}", 
                                         weight, step=episode)
