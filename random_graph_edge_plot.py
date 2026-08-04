import copy
import numpy as np
import matplotlib.pyplot as plt
from util import sample_gaussian
from rigidity import MBR_required_Rd
import time
import sys
import os

from environment import Environment
from visualizer import Visualizer


# # DEBUG
def sample_and_plot_gaussian(mean, variance, n, num_samples=1000):
    """
    Samples from a Gaussian distribution and plots the histogram.
    """
    std_dev = np.sqrt(variance)

    # Generate samples
    samples = sample_gaussian(mean, variance, n, num_samples)

    # Plotting
    plt.figure(figsize=(8, 5))
    count, bins, ignored = plt.hist(samples, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')

    # Overlay the theoretical probability density function (PDF)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    plt.plot(x, p, 'r-', linewidth=2, label=f'Theoretical PDF\nMean={mean}, Var={variance}')

    plt.title('Gaussian Distribution Sampling')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if len(sys.argv) < 2:
    print("usage: python3 manual.py [environment_name]")
    quit()

env_name = sys.argv[1]
filepath = "./environments/" + env_name + ".json"

if not os.path.exists(filepath):
    print(f"{filepath} not found")
    quit()

raw_env = Environment()
raw_env.load(filepath)

netw = raw_env.network
n = len(netw.agents)

# DEBUG
domains = [agent.domain for agent in netw.agents]
mean = MBR_required_Rd(n, 2 if (("R^2" in domains) or ("R^2xS^1" in domains)) else 3)
random_edge_count = sample_and_plot_gaussian(
    mean,
    ((n**2-n) - mean)**2/9,
    n
    )
