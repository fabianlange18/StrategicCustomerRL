from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import CallbackList

import config
from market import Market

from evaluation.evaluator import Evaluator
from evaluation.callbacks.policy_evaluation_callback import PolicyEvaluationCallback
from evaluation.callbacks.seasonal_evaluation_callback import SeasonalEvaluationCallback
from evaluation.callbacks.simulation_plot_callback import SimulationPlotCallback

def train_PPO():

    e = Market()
    ev = Evaluator()
    callbacks = setup_callbacks(ev)

    model = PPO("MlpPolicy", e, config.learning_rate, gamma=config.gamma)

    print(f"RUN {config.run_name}")

    model.learn(config.episode_length * config.n_episodes, callbacks, progress_bar=True)
    model.save(f'{config.summary_dir}/model')


def setup_callbacks(evaluator):

    policy_evaluation_callback = PolicyEvaluationCallback(evaluator, n_steps=config.episode_length * config.peval_cb_n_episodes)
    simulation_plot_callback = SimulationPlotCallback(evaluator, n_steps=config.episode_length * config.sim_cb_n_episodes)

    callbacks = [policy_evaluation_callback, simulation_plot_callback]
    
    if config.customers_types.__contains__("seasonal"):
        seasonal_callback = SeasonalEvaluationCallback(evaluator, n_steps=config.episode_length * config.peval_cb_n_episodes)
        callbacks.append(seasonal_callback)
    
    callback_list = CallbackList(callbacks)

    return callback_list

if __name__ == '__main__':
    train_PPO()