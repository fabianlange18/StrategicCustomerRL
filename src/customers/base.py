import numpy as np
from abc import ABC, abstractmethod

import config

class Customer(ABC):

    def __init__(self):
        self.name = "undefined"
        self.ability_to_wait = False

    @abstractmethod
    def generate_purchase_probabilities_from_offer(self, state, action) -> np.array:
        raise NotImplementedError
    
    def calculate_weight(self, price, beta):
        return ( -config.alpha * np.exp(price - beta) - price) / beta + config.alpha
