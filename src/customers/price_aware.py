import numpy as np
from typing import Tuple
from scipy.special import softmax
from collections import deque

import config
from customers.base import Customer


class Price_Aware_Customer(Customer):

    def __init__(self):
        self.name = "price-aware"
        self.ability_to_wait = True
        self.price_storage_length = (1 + config.competitor) * config.n_last_prices_customer
        self.last_prices = deque([0], maxlen=self.price_storage_length)

    def generate_purchase_probabilities_from_offer(self, state, action) -> Tuple[np.array, int]:

        weights = [config.nothing_preference]
        threshold_price = min(self.last_prices) * config.undercut_min

        for price in action:
            if len(self.last_prices) >= self.price_storage_length and price < min(threshold_price, config.max_buying_price):
                weights.append(10)
            else:
                weights.append(-10)

        self.last_prices.append(min(action))
        
        return softmax(np.array(weights)), threshold_price