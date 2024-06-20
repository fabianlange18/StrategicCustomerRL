import numpy as np
from gym import Env
from gym.spaces.box import Box
from gym.spaces.multi_discrete import MultiDiscrete
from collections import defaultdict

import config

from customers.seasonal import Seasonal_Customer as seasonal
from customers.price_aware import Price_Aware_Customer as price_aware
from customers.anticipating import Anticipating_Customer as anticipating

from competitor import Undercutting_Vendor

class Market(Env):

    def __init__(self):
        
        # Init Customers
        self.customers = self.init_customers()

        # Init Waiting Pool
        self.n_waiting_types = sum([customer.ability_to_wait for customer in self.customers])

        # Init State w/ Price History
        self.price_storage_length = (1 + config.competitor) * config.n_last_prices_state
        last_prices = [config.max_price * 100 for _ in range(self.price_storage_length)]
        self.observation_space = MultiDiscrete([config.week_length, *last_prices])

        # Init Action Space
        self.action_space = Box(
            low=0.0,
            high=config.max_price,
            shape=(1,)
        )

        # Init Competitor
        self.competitor = Undercutting_Vendor()

        self.step_counter = 0
        self.reset()

    
    def reset(self, seed = 0):
        self.s = np.array([0, *[0 for _ in range(self.price_storage_length)]])
        self.reset_waiting_pool()
        return self.s # {}

    def reset_waiting_pool(self):
        self.waiting_pool = np.zeros(self.n_waiting_types)


    def step(self, action):

        # Init Logging
        info = defaultdict(int)

        reward = [0, 0] # First and Second Half Iteration, in Monopoly only first value is used
        competitor_reward = [0, 0] # ''

        # Draw Customer Arrivals Multinomially
        customer_arrivals = np.random.multinomial(config.n_customers, config.customer_mix)

        # Add Waiting Customers with p_return
        waiting_count = 0
        for i, customer in enumerate(self.customers):
            if customer.ability_to_wait:
                customer_arrivals[i] += np.random.binomial(self.waiting_pool[waiting_count], config.p_return)
                waiting_count += 1
        self.reset_waiting_pool()

        # Get Old Competitor Price and Draw Customer Arrivals if Duopoly
        if config.competitor:
            action = np.append(action, self.competitor.price)
            customer_arrivals_i0 = [np.random.binomial(customer_arrivals[i], 0.5) for i in range(len(customer_arrivals))]
            customer_arrivals_i1 = customer_arrivals - customer_arrivals_i0
            customer_arrivals = [customer_arrivals_i0, customer_arrivals_i1]
            
            # Log Customer Arrivals
            for i, customer in enumerate(self.customers):
                info[f"i0_n_{customer.name}"] = customer_arrivals_i0[i]
                info[f"i1_n_{customer.name}"] = customer_arrivals_i1[i]
        
        else:
            customer_arrivals = [customer_arrivals]
            for i, customer in enumerate(self.customers):
                info[f"i0_n_{customer.name}"] = customer_arrivals[0][i]
                info[f"i1_n_{customer.name}"] = customer_arrivals[0][i]



        # Iterate over Vendors
        for i in range(1 + config.competitor):

            # Update Competitor Price if in Second Iteration
            if i == 1:
                i0_competitor_price = self.competitor.price
                action = np.array([action[0], self.competitor.update_price(action[0])])

            # Iterate over Customers
            waiting_count = 0
            for j, customer in enumerate(self.customers):

                probability_distribution, reference = customer.generate_purchase_probabilities_from_offer(self.s, action)
                customer_decisions = np.random.multinomial(customer_arrivals[i][j], probability_distribution)

                customer_reward = customer_decisions[1] * action[0]
                reward[i] += customer_reward

                if config.competitor:
                    customer_competitor_profit = customer_decisions[2] * action[1]
                    competitor_reward[i] += customer_competitor_profit

                if customer.ability_to_wait:
                    self.waiting_pool[waiting_count] += np.random.binomial(customer_decisions[0], config.p_remain)
                    info[f"i{i}_n_{customer.name}_waiting"] += min(self.waiting_pool[waiting_count], config.max_waiting_pool)
                    waiting_count += 1

                # Logging
                info[f"i{i}_n_{customer.name}_buy"] = customer_decisions[1]
                if config.competitor:
                    info[f"i{i}_n_{customer.name}_competitor_buy"] = customer_decisions[2]
                    info[f"i{i}_{customer.name}_competitor_reward"] = customer_competitor_profit
                info[f"i{i}_{customer.name}_reference_price"] = reference
                info[f"i{i}_{customer.name}_reward"] = customer_reward

        # Store Last Prices
        if not config.competitor:
            for i in range(self.price_storage_length - 1):
                self.s[1 + i] = self.s[2 + i]
            self.s[-1] = min(action)
        else:
            for i in range(self.price_storage_length - 2):
                self.s[1 + i] = self.s[3 + i]
            self.s[-2] = min(action[0], i0_competitor_price) * 100
            self.s[-1] = min(action) * 100


        # State and Waiting Pool Max
        self.s = np.minimum(self.s, config.max_price * 100 - 1)
        self.waiting_pool = np.minimum(self.waiting_pool, config.max_waiting_pool)

        # Update State
        self.s[0] += 1
        self.s[0] %= config.week_length
        self.step_counter += 1
        done = self.step_counter == config.episode_length

        # Logging
        info["i0_agent_offer_price"], info["i1_agent_offer_price"] = action[0], action[0]
        info["i0_total_reward"], info["i1_total_reward"] = reward[0], reward[1]
        if config.competitor:
            info["i0_competitor_offer_price"], info["i1_competitor_offer_price"] = i0_competitor_price, action[1]
            info["i0_total_competitor_reward"], info["i1_total_competitor_reward"] = competitor_reward[0], competitor_reward[1]

        return self.s, sum(reward), done, info



    def init_customers(self):

        customers = []

        for customer_name in config.customers_types:
            if customer_name == 'seasonal':
                customer = seasonal('myopic', False)
            elif customer_name == 'recurring':
                customer = seasonal('recurring', True)
            elif customer_name == 'price-aware':
                customer = price_aware()
            elif customer_name == 'anticipating':
                customer = anticipating()
            customers.append(customer)
        
        return customers