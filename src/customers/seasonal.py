import numpy as np
from typing import Tuple
from scipy.special import softmax

import config
from customers.base import Customer

class Seasonal_Customer(Customer):

    def __init__(self, name, ability_to_wait):
        self.name = name
        self.ability_to_wait = ability_to_wait

    def generate_purchase_probabilities_from_offer(self, state, action) -> Tuple[np.array, int]:

        weights = [config.nothing_preference]

        beta = config.beta[state[0]]

        for price in action:
            weights.append(self.calculate_weight(price, beta))
        
        return softmax(np.array(weights)), beta