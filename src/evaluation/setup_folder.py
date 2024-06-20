import os
import config

def setup_folder():
    os.makedirs(config.summary_dir, exist_ok=True)
    os.makedirs(f'{config.plot_dir}deterministic', exist_ok=True)
    os.makedirs(f'{config.plot_dir}stochastic', exist_ok=True)
    os.makedirs(f'{config.plot_dir}prices', exist_ok=True)