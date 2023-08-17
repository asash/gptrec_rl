import io
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


def plot_to_image(func):
  def wrapper(*args, **kwargs):
        figure = func(*args, **kwargs)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image
  return wrapper

@plot_to_image
def plot_rewards_per_pos(rewards):
    rewards = np.array(rewards)
    mean_rewards = np.mean(rewards,axis=0)
    fig =plt.figure(figsize=(4, 2.5))
    ax = fig.add_subplot()
    ax.bar(np.arange(len(mean_rewards))+1, mean_rewards)
    ax.set_xlabel("Position")
    ax.set_ylabel(f"Reward per position")
    ax.set_title(f"Reward sum: {np.sum(mean_rewards):.2f}")
    return fig
    
 
   
@plot_to_image
def plot_tradeoff_trajectory(trajectory, tradeoff_name):
    n = len(trajectory)
    indices = list(range(n))  # Original indices
    
    # Ensure the trajectory has at most 1000 points while preserving the first and last point
    if n > 1000:
        step = n // 1000
        trajectory = [trajectory[i] for i in range(0, n, step)]
        indices = [indices[i] for i in range(0, n, step)]
        if trajectory[-1] != trajectory[n-1]:
            trajectory.append(trajectory[n-1])
            indices.append(n-1)
    
    metric1_values = [x[0] for x in trajectory]
    metric2_values = [x[1] for x in trajectory]
    
    # Create a color map from dark green to red
    cmap = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a scatter plot instead of a line plot
    scatter = ax.scatter(metric1_values, metric2_values, c=indices, cmap=cmap, s=10)

    # Increase size of the last point and mark it in pure red
    ax.scatter(metric1_values[-1], metric2_values[-1], color='red', marker='o', s=100)

    metric1_name = tradeoff_name.split(':::')[0]
    metric2_name = tradeoff_name.split(':::')[1]
    ax.set_xlabel(metric1_name)
    ax.set_ylabel(metric2_name)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Step in trajectory', rotation=270, labelpad=15)
    
    ax.grid()
    return fig