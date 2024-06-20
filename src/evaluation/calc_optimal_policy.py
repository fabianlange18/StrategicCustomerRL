import config as config

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from customers.seasonal import Seasonal_Customer

def calculate_optimal_policy_seasonal():
    optimal_prices = []
    optimal_profits_per_customer = []

    for reference_price in config.beta:

        def profit(price):
            weight = Seasonal_Customer(name="seasonal", ability_to_wait=False).calculate_weight(price, reference_price)
            weights = [config.nothing_preference]
            weights.append(weight[0])
            return - price * softmax(np.array(weights))[1]

        optimization_result = minimize(profit, x0=0)
        optimal_prices.append(optimization_result.x[0])
        optimal_profits_per_customer.append(-optimization_result.fun)

    return optimal_prices, optimal_profits_per_customer


def calculate_expected_reward(prices):

    expected_profits_per_customer = []

    for price, reference_price in zip(prices, config.beta):
        
        weight = Seasonal_Customer(name="seasonal", ability_to_wait=False).calculate_weight(price, reference_price)
        weights = [config.nothing_preference]
        weights.append(weight)
        profit = price * softmax(np.array(weights))[1]
        expected_profits_per_customer.append(profit)
    
    return expected_profits_per_customer


def calculate_mean_difference(actual, optimal):
    diff = np.divide(actual, optimal)
    return np.mean(diff)