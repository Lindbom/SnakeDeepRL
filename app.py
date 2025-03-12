import streamlit as st
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
from PIL import Image


from Game import Game, Agent

sys.path.append('.')
from model import SnakeAgent, INPUT_SHAPE, GRID
from PIL import Image, ImageDraw, ImageFont
import io

# Set page configuration
st.set_page_config(
    page_title="Snake Viz",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS styling
st.markdown("""
<style>
    .stApp {
        background-color: #f0f0f8;
    }
    .game-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metrics-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #e0e0e8;
    }
</style>
""", unsafe_allow_html=True)

# Define color schemes for the snake visualization
SNAKE_COLORS = {
    "classic": {"head": "green", "body": "lightgreen", "apple": "red", "wall": "darkgray", "background": "white"},
    "blue": {"head": "navy", "body": "royalblue", "apple": "crimson", "wall": "slategray", "background": "aliceblue"},
    "dark": {"head": "lime", "body": "forestgreen", "apple": "orangered", "wall": "dimgray", "background": "black"},
    "neon": {"head": "magenta", "body": "deeppink", "apple": "yellow", "wall": "purple", "background": "black"},
    "monochrome": {"head": "black", "body": "darkgray", "apple": "black", "wall": "black", "background": "whitesmoke"}
}


if 'running_game' not in st.session_state:
    st.session_state.running_game = False

if 'current_model' not in st.session_state:
    st.session_state.current_model = None

if 'game_speed' not in st.session_state:
    st.session_state.game_speed = 0.05

if 'color_scheme' not in st.session_state:
    st.session_state.color_scheme = "classic"

if 'grid_size' not in st.session_state:
    st.session_state.grid_size = 15

if 'max_episodes' not in st.session_state:
    st.session_state.max_episodes = 100

if 'current_episode' not in st.session_state:
    st.session_state.current_episode = 0

if 'best_reward' not in st.session_state:
    st.session_state.best_reward = 0

if 'best_steps' not in st.session_state:
    st.session_state.best_steps = 0

if 'best_apples' not in st.session_state:
    st.session_state.best_apples = 0

import pygame
import numpy as np


def convert_game_to_image_streamlit(game, color_scheme="classic", show_thinking=True, activations=None, action_probs=None, 
                                   cell_size=20, cache=None):
    """
    Convert the game grid to a PIL Image optimized for Streamlit display,
    optionally showing neural network activations.
    
    Args:
        game: The snake game object
        color_scheme: Color scheme to use ("classic", "dark", etc.)
        show_thinking: Whether to show neural network visualization
        activations: List of activation arrays for each layer
        action_probs: Action probabilities from the output layer
        cell_size: Size of each grid cell in pixels
        cache: Dictionary for caching fonts and other resources
        
    Returns:
        PIL.Image: The rendered game visualization
    """
    # Define colors based on color scheme
    SNAKE_COLORS = {
        "classic": {
            "background": (220, 220, 220),
            "wall": (40, 40, 40),
            "body": (50, 150, 50),
            "head": (50, 50, 50),
            "apple": (200, 0, 0),
            "text": (0, 0, 0),
            "neuron_bg": (255, 255, 255),
            "neuron_active": (255, 0, 0),
            "connections": (200, 200, 200)
        },
        "dark": {
            "background": (30, 30, 30),
            "wall": (70, 70, 70),
            "body": (0, 180, 0),
            "head": (0, 230, 0),
            "apple": (230, 0, 0),
            "text": (220, 220, 220),
            "neuron_bg": (50, 50, 50),
            "neuron_active": (200, 60, 0),
            "connections": (100, 100, 100)
        }
    }
    
    colors = SNAKE_COLORS.get(color_scheme, SNAKE_COLORS["classic"])
    
    # Initialize cache if needed
    if cache is None:
        cache = {}
    
    cache['font'] = ImageFont.load_default()
    
    font = cache['font']
    # Calculate dimensions
    grid_height, grid_width = game.grid.shape
    panel_width = 500 if show_thinking else 0
    
    image_width = grid_width * cell_size + panel_width
    image_height = grid_height * cell_size
    
    # Create image and drawing context
    image = Image.new("RGB", (image_width, image_height), colors["background"])
    draw = ImageDraw.Draw(image)
    
    # Create a grid representation
    grid = np.full_like(game.grid, 0, dtype=np.int8)
    
    # Mark walls
    grid[game.grid == 1] = 1
    
    # Mark snake body
    for body_part in game.agent.body[1:]:
        try:
            grid[tuple(body_part)] = 2
        except IndexError:
            pass  # Skip if body part is out of bounds
    
    # Mark snake head
    try:
        grid[tuple(game.agent.body[0])] = 3
    except IndexError:
        pass  # Skip if head is out of bounds
    
    # Mark apple
    grid[tuple(game.apple)] = 4
    
    # Draw the grid
    for y in range(grid_height):
        for x in range(grid_width):
            cell_value = grid[y, x]
            rect = [x * cell_size, y * cell_size, (x+1) * cell_size, (y+1) * cell_size]
            
            if cell_value == 0:  # Empty
                pass  # Skip drawing to show background
            elif cell_value == 1:  # Wall
                draw.rectangle(rect, fill=colors["wall"])
            elif cell_value == 2:  # Snake body
                draw.rectangle(rect, fill=colors["body"])
            elif cell_value == 3:  # Snake head
                draw.rectangle(rect, fill=colors["head"])
            elif cell_value == 4:  # Apple
                draw.rectangle(rect, fill=colors["apple"])
            
            # Draw cell borders
            draw.rectangle(rect, outline=(0, 0, 0))
    

    # Neural network panel starts after the game grid
    panel_x = grid_width * cell_size
    
    # Draw panel background
    draw.rectangle([panel_x, 0, image_width, image_height], fill=(240, 240, 240))
    
    # Draw title
    draw.text((panel_x + 10, 10), "Neural Network Activations", fill=colors["text"], font=font)
    
    # Visualize neural network layers
    layer_names = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]
    layer_sizes = [7, 20, 24, 24, 3]  # From the model architecture
    
    # Calculate positions for layers
    total_width = panel_width - 60
    layer_spacing = total_width / (len(layer_sizes) - 1)
    
    # Create action probabilities array if not in activations
    latest_activation = activations.copy()
    max_idx = np.argmax(action_probs)
    action = np.zeros_like(action_probs)
    action[max_idx] = 1
    latest_activation[-1] = action
    
    # Draw network
    for i, (layer_name, layer_size) in enumerate(zip(layer_names, layer_sizes)):
        x_pos = panel_x + 20 + i * layer_spacing
        
        text_width = len(layer_name) * 7  # Rough estimate
        draw.text((x_pos - text_width//2, image_height - 30), layer_name, fill=colors["text"], font=font)
        
        # Calculate neuron positions for this layer
        neuron_spacing = min(15, (image_height - 80) / layer_size)
        start_y = 50 + (image_height - 80 - neuron_spacing * layer_size) / 2
        
        # Get activation values for normalization
        if i <= len(latest_activation):
            act_values = latest_activation[i]
            if len(act_values) > 0:
                max_val = max(act_values)
                min_val = min(act_values)
                range_val = max_val - min_val if max_val != min_val else 1
            
            # Draw neurons with activation values
            for j in range(min(layer_size, len(act_values))):
                y_pos = start_y + j * neuron_spacing
                
                # Normalize value for coloring
                val = act_values[j]
                norm_val = (val - min_val) / range_val if range_val != 0 else 0.5
                
                # Interpolate color between background and active
                r = int(colors["neuron_bg"][0] + norm_val * (colors["neuron_active"][0] - colors["neuron_bg"][0]))
                g = int(colors["neuron_bg"][1] + norm_val * (colors["neuron_active"][1] - colors["neuron_bg"][1]))
                b = int(colors["neuron_bg"][2] + norm_val * (colors["neuron_active"][2] - colors["neuron_bg"][2]))
                
                # Draw neuron
                neuron_radius = 5
                draw.ellipse([x_pos-neuron_radius, y_pos-neuron_radius, 
                                x_pos+neuron_radius, y_pos+neuron_radius], 
                                fill=(r, g, b), outline=(0, 0, 0))
        
        # Draw connections to next layer (simplified - only showing main connections)
        if i < len(layer_sizes) - 1:
            next_x = panel_x + 20 + (i + 1) * layer_spacing
            next_size = layer_sizes[i + 1]
            next_spacing = min(15, (image_height - 80) / next_size)
            next_start_y = 50 + (image_height - 80 - next_spacing * next_size) / 2
            
            # Draw only a subset of connections to reduce clutter
            connection_stride = max(1, layer_size // 5)
            # connection_stride =1# max(1, layer_size)
            for j in range(0, layer_size, connection_stride):
                y_pos = start_y + j * neuron_spacing

                next_stride = max(1, next_size // 5)
                # next_stride = 1
                for k in range(0, next_size, next_stride):
                    next_y = next_start_y + k * next_spacing
                    draw.line([(x_pos + neuron_radius, y_pos), 
                                (next_x - neuron_radius, next_y)], 
                                fill=colors["connections"], width=1)

    return image


MODEL_FOLDER = 'best_models'
def main():
    
    tab1,  = st.tabs(["Snake AI"])
    
    with st.sidebar:
        st.header("Configuration")
        
        st.session_state.grid_size = st.slider("Grid Size", min_value=5, max_value=30, value=15, step=1)
        speed_slider = st.slider("Game Speed", min_value=0.01, max_value=0.25, value=st.session_state.game_speed, step=0.01, key="game_speed_slider")
        st.session_state.game_speed = speed_slider
        
        st.header("Model Selection")
        # Scan for models in the models directory
        available_models = []

        if os.path.exists(MODEL_FOLDER):
            available_models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.h5')]
        
        model_options = available_models
        selected_model = st.selectbox("Select Model", model_options, 
                                    on_change=lambda: setattr(st.session_state, 'running_game', True))
        
        if selected_model == "None (Random Actions)":
            st.session_state.current_model = None
        else:
            st.session_state.current_model = os.path.join(MODEL_FOLDER, selected_model)
        
        col1, col2 = st.columns(2)
        
        st.session_state.running_game = True
        with col1:
            if st.button("Start Game"):
                st.session_state.running_game = True
                
        with col2:
            if st.button("Stop Game"):
                st.session_state.running_game = False
        
    # Tab 1: Snake AI Playground
    with tab1:
        st.title("Snake")
        
        # Create a two-column layout for game and stats
        game_col, stats_col = st.columns([3, 1])
        
        with game_col:
            game_display = st.empty()
            
        with stats_col:
            metrics_display = st.empty()
        
        if st.session_state.running_game:
            # Create and setup the game objects
            game = Game(size=st.session_state.grid_size)
            
            if st.session_state.current_model:
                snake_ai = SnakeAgent(input_shape=INPUT_SHAPE, load_model=st.session_state.current_model)
                snake_ai.epsilon = -1  # Small epsilon for some randomness
                
            total_reward = 0
            steps = 0
            apples_eaten = 0
            steps_since_apple = 0

            # Game loop
            while st.session_state.running_game:

                try:
                    state = game.agent.get_features(game.apple, game.grid)
                    action = snake_ai.choose_action(state)
                    reward = game.update(action)
                    total_reward += reward
                    steps += 1
                    
                    if reward == game.FOUND_APPLE:
                        apples_eaten += 1
                        steps_since_apple = 0
                    else:
                        steps_since_apple += 1
                    
                    done = (reward == game.HIT_WALL or reward == game.HIT_BODY)
                    
                    img = convert_game_to_image_streamlit(game, 
                                                st.session_state.color_scheme,
                                                show_thinking=True,
                                                activations=snake_ai.last_activation,
                                                action_probs=snake_ai.last_action_probs)
                    game_display.image(img, use_container_width=False)
                    
                    # Update statistics display
                    metrics_display.markdown(f"""
                    ### Live Game Statistics
                    
                    **Steps:** {steps}
                    
                    **Apples Eaten:** {apples_eaten}
                    
                    **Total Reward:** {total_reward}
                    
                    **Current Action:** {["Right", "Up", "Left", "Down", "None"][action]}
                    
                    **Snake Length:** {len(game.agent.body)}
                    
                    **Game Speed:** {st.session_state.game_speed}s delay
                    """)
                    
                    time.sleep(st.session_state.game_speed)
                    
                    if done:
                        print(f"Breaking due to  {'Hit Wall' if reward == game.HIT_WALL else 'Hit Body'} ")
                        
                        game = Game(size=st.session_state.grid_size)
                        snake_ai = SnakeAgent(input_shape=INPUT_SHAPE, load_model=st.session_state.current_model)
                        snake_ai.epsilon = -1  # Small epsilon for some randomness
                            
                        total_reward = 0
                        steps = 0
                        apples_eaten = 0
                        steps_since_apple = 0
                        time.sleep(2)

                except Exception as e:
                    st.error(f"Error during game: {e}")
                    st.session_state.running_game = False
                    break
 
if __name__ == "__main__":
    main()