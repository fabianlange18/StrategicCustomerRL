# source: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

import config as config
import numpy as np

from evaluation.calc_optimal_policy import calculate_optimal_policy_seasonal, calculate_expected_reward, calculate_mean_difference

from stable_baselines3.common.callbacks import BaseCallback

class SeasonalEvaluationCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, evaluator, n_steps, verbose=0):
        self.perfect_prices, self.perfect_profits_per_customer = calculate_optimal_policy_seasonal()

        self.evaluator = evaluator
        self.n_steps = n_steps
        self.last_time_trigger = 0

        self.price_diffs = []
        self.profit_diffs = []

        super(SeasonalEvaluationCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:

        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps

            actual_prices = self.predict_prices()
            expected_profits_per_customer = calculate_expected_reward(actual_prices)

            price_diff = calculate_mean_difference(actual_prices, self.perfect_prices)
            profit_diff = calculate_mean_difference(expected_profits_per_customer, self.perfect_profits_per_customer)

            self.price_diffs.append(price_diff)
            self.profit_diffs.append(profit_diff)
            

        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """

        actual_prices = self.predict_prices()
        expected_profits_per_customer = calculate_expected_reward(actual_prices)

        self.evaluator.plot_seasonal_diff(self.price_diffs, 'Prices')
        self.evaluator.plot_seasonal_diff(self.profit_diffs, 'Profits')
        self.evaluator.print_seasonal_policy_stats(actual_prices, expected_profits_per_customer, self.perfect_prices, self.perfect_profits_per_customer)

        pass

    def predict_prices(self):
        try:
            state = np.array(self.locals['obs_tensor'][0])
        except KeyError:
            state = np.array(self.locals['new_obs'][0])            
            
        actual_prices = np.array([self.model.predict(np.array([s, *state[1:]]), deterministic=True)[0][0] for s in range(config.week_length)])
        
        ## DQN ##
        # actual_prices = [self.model.predict(np.array([s, *state[1:]]), deterministic=True)[0] for s in range(config.week_length)]

        return actual_prices
