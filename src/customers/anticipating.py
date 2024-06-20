import numpy as np
from typing import Tuple
from scipy.special import softmax
from collections import deque
from statsmodels.tsa.ar_model import AutoReg

import config
from customers.base import Customer


class Anticipating_Customer(Customer):

    def __init__(self):
        self.name = "anticipating"
        self.ability_to_wait = True
        self.price_storage_length = 2 * (1 + config.competitor) * config.week_length
        self.n_predictions = (1 + config.competitor) * config.n_last_prices_customer
        self.last_prices = deque([], maxlen=self.price_storage_length)
        self.predictions = np.ndarray(self.n_predictions)

    def generate_purchase_probabilities_from_offer(self, state, action) -> Tuple[np.array, int]:

        weights = [config.nothing_preference]

        predicted = self.predict_prices()

        for price in action:
            threshold_price = min(self.predictions) * config.undercut_min
            if predicted and price < min(threshold_price, config.max_buying_price):
                weights.append(10)
            else:
                weights.append(-10)

        self.last_prices.append(min(action))
        
        return softmax(np.array(weights)), config.undercut_min * self.predictions[0] # min(threshold_price, config.max_buying_price)
    

    def predict_prices(self):
        if len(self.last_prices) == self.price_storage_length:
            if config.competitor:
                self.predictions = AutoReg(
                    list(self.last_prices),
                    lags=0,
                    hold_back=self.n_predictions,
                    seasonal=True,
                    trend='n',
                    period=(1 + config.competitor) * config.week_length
                    ).fit().predict(
                        start=len(self.last_prices),
                        end=len(self.last_prices) + self.n_predictions - 1
                    )
            else:
                self.predictions = AutoReg(
                    list(self.last_prices),
                    lags=self.n_predictions,
                    ).fit().predict(
                        start=len(self.last_prices),
                        end=len(self.last_prices) + self.n_predictions - 1
                    )
        return len(self.last_prices) == self.price_storage_length