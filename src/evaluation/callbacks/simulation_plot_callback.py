# source: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
import config as config

from simulation import simulate_policy
from market import Market

from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps

class SimulationPlotCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, evaluator, n_steps, verbose=0):

        self.customers = Market().customers

        self.evaluator = evaluator
        self.n_steps = n_steps
        self.last_time_trigger = 0

        super(SimulationPlotCallback, self).__init__(verbose)
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

            infos_deterministic = simulate_policy(self.model)
            infos_stochastic = simulate_policy(self.model)

            self.evaluator.plot_trajectories(infos_deterministic, save = f"{config.plot_dir}deterministic/step_{self.num_timesteps}")
            self.evaluator.plot_trajectories(infos_stochastic, save = f"{config.plot_dir}stochastic/step_{self.num_timesteps}")
            

        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True
