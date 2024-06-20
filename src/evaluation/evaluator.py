import config
import numpy as np
from itertools import chain

import matplotlib.pyplot as plt

from evaluation.setup_folder import setup_folder
from evaluation.calc_optimal_policy import calculate_mean_difference

from market import Market
from customers.seasonal import Seasonal_Customer

class Evaluator:

    folder_exists = False

    def __init__(self) -> None:

        self.customers = Market().customers

        if not self.folder_exists:
            setup_folder()
            self.print_header()
            self.folder_exists = True

    def write_output(self, output):
        f = open(f'{config.summary_dir}summary.txt', 'a')
        f.write(output)
        f.close()


    def print_header(self):
        self.write_output(f"Summary for Run {config.run_name}\n\n")

        # Customer mix
        [self.write_output(f'{config.customer_mix[i] * 100} % {customer.name}\n') for i, customer in enumerate(self.customers)]

        # Competitor
        if config.competitor:
            self.write_output(f"Playing against undercutting competitor\n")
        

    def print_simulation_statistics(self, infos, save_infos = True):

        if save_infos:
            f = open(f'{config.summary_dir}data.txt', 'a')
            f.write(str(infos))
            f.close()

        infos = {key: value[config.episode_length // 2:] for key, value in infos.items()}

        infos = self.add_concatenated_infos(infos)


        self.write_output("\n\n\nStatistics for one deterministic simulation episode\n\n")
        self.write_output(f'{"Property": >40}{"Sum": >10}{"Mean": >12}{"Std": >10}{"Min": >12}{"Median": >14}{"Max": >13}\n\n')

        for key in infos.keys():
            stats = [np.sum(infos[key]), np.mean(infos[key]), np.std(infos[key]), np.min(infos[key]), np.median(infos[key]), np.max(infos[key])]
            self.write_output(f'{key: >40}{stats[0]: >10.2f}{stats[1]: >12.2f}{stats[2]: >10.2f}{stats[3]: >13.2f}{stats[4]: >13.2f}{stats[5]: >13.2f}\n')

        for customer in self.customers:
            reward = np.sum(infos[f'{customer.name}_reward'])
            buy = np.sum(infos[f'n_{customer.name}_buy'])
            self.write_output(f"\nAverage Agent Sales Price {customer.name}: {np.round(reward / buy, 3)}\n")
        
        self.write_output(f"Average Offer Price Agent: {np.round(np.mean(infos['agent_offer_price']), 3)}\n")
        if config.competitor:
            self.write_output(f"Average Offer Price Competitor: {np.round(np.mean(infos['competitor_offer_price']), 3)}\n")



    def add_concatenated_infos(self, infos, extend_one = False):
        if not config.competitor:
            infos[f'agent_offer_price'] = infos[f'i0_agent_offer_price'].copy()
            infos['total_reward'] = infos[f'i0_total_reward'].copy()
            for customer in self.customers:
                infos[f'{customer.name}_reward'] = infos[f'i0_{customer.name}_reward'].copy()
                infos[f'n_{customer.name}_buy'] = infos[f'i0_n_{customer.name}_buy'].copy()
                infos[f'n_{customer.name}'] = infos[f'i0_n_{customer.name}'].copy()
                infos[f'{customer.name}_reference_price'] = infos[f'i0_{customer.name}_reference_price'].copy()
                if customer.ability_to_wait:
                    infos[f'n_{customer.name}_waiting'] = infos[f'i0_n_{customer.name}_waiting'].copy()

        else:
            infos['agent_offer_price'] = list(chain(*zip(infos[f'i0_agent_offer_price'], infos[f'i1_agent_offer_price'])))
            infos['competitor_offer_price'] = list(chain(*zip(infos[f'i0_competitor_offer_price'], infos[f'i1_competitor_offer_price'])))
            infos['total_reward'] = list(chain(*zip(infos[f'i0_total_reward'], infos[f'i1_total_reward'])))
            infos['total_competitor_reward'] = list(chain(*zip(infos[f'i0_total_competitor_reward'], infos[f'i1_total_competitor_reward'])))
            for customer in self.customers:
                infos[f'{customer.name}_reward'] = list(chain(*zip(infos[f'i0_{customer.name}_reward'], infos[f'i1_{customer.name}_reward'])))
                infos[f'{customer.name}_competitor_reward'] = list(chain(*zip(infos[f'i0_{customer.name}_competitor_reward'], infos[f'i1_{customer.name}_competitor_reward'])))
                infos[f'n_{customer.name}_buy'] = list(chain(*zip(infos[f'i0_n_{customer.name}_buy'], infos[f'i1_n_{customer.name}_buy'])))
                infos[f'n_{customer.name}_competitor_buy'] = list(chain(*zip(infos[f'i0_n_{customer.name}_competitor_buy'], infos[f'i1_n_{customer.name}_competitor_buy'])))
                infos[f'n_{customer.name}'] = list(chain(*zip(infos[f'i0_n_{customer.name}'], infos[f'i1_n_{customer.name}'])))
                infos[f'{customer.name}_reference_price'] = list(chain(*zip(infos[f'i0_{customer.name}_reference_price'], infos[f'i1_{customer.name}_reference_price'])))
                if customer.ability_to_wait:
                    infos[f'n_{customer.name}_waiting'] = [(x + y) for x, y in zip(infos[f'i0_n_{customer.name}_waiting'], infos[f'i1_n_{customer.name}_waiting']) for _ in range(2)]
        
        if extend_one:
            for key in infos.keys():
                infos[key].append(infos[key][-1])
        
        return infos


    def plot_trajectories(self, infos, show = False, save = "", extend_one=True):

        infos = self.add_concatenated_infos(infos, extend_one=1-config.competitor)

        step = 0.5 if config.competitor else 1

        x = np.arange(stop=len(infos['total_reward']) / (config.competitor + 1), step=step)

        # check whether a waiting pool exists
        waiting_pool = False
        for customer in self.customers:
            waiting_pool = customer.ability_to_wait or waiting_pool

        if waiting_pool:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
            # if config.n_last_prices_state > 0:
            #     ax4.axvline(x=config.n_last_prices_state, color='black', linestyle='--', label='Attunement')
            ax4.set_title('Waiting Pool')
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        
        plt.xlabel("t")

        # Agent Profits by Customer Type
        ax1.set_title('Reward')
        ax1.step(x, infos['total_reward'], color='black', label='agent', where='post')

        # Reference Prices Customers
        ax2.set_title("Offer Price (black) / Price-anticipating Forecast (blue)") #/ Demand Beta (blue)
        if not config.competitor:
            ax2.step(x, infos[f'agent_offer_price'], color='black', label='agent', where='post')

        # Customer Decisions by Type
        ax3.set_title(f'Customer Decisions {customer.name}')

        # Number of Customers per Type
        # ax5.set_title("Number of Customers per Type")
        # for customer in customers
            
        # Waiting Pool
        # for customer in customers

        for customer in self.customers:
            if len(self.customers) > 1:
                ax1.step(x, infos[f'{customer.name}_reward'], label=customer.name, where='post')
            ax2.step(x, infos[f'{customer.name}_reference_price'], label=customer.name, where='post')
            ax3.step(x, infos[f'n_{customer.name}_buy'], label=customer.name, where='post')
            # ax5.step(x, infos[f'n_{customer.name}'], label=customer.name, where='post')
            if waiting_pool:
                if customer.ability_to_wait:
                    ax4.step(x, infos[f'n_{customer.name}_waiting'], label=customer.name, where='post')
                    # ax4.set_ylim(bottom=0)
                else:
                    next(ax4._get_lines.prop_cycler)

        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels)

        plt.subplots_adjust(hspace=0.4)

        # ax1.set_ylim(top=600)
        ax2.set_ylim(bottom=0, top=config.max_price + 1)
        # ax3.set_ylim(bottom=0, top=50)
        # ax5.set_ylim(bottom=0, top=config.max_waiting_pool + 105)
        # if config.n_last_prices_state > 0:
        #     ax1.axvline(x=config.n_last_prices_state, color='black', linestyle='--', label='Attunement')
        #     ax2.axvline(x=config.n_last_prices_state, color='black', linestyle='--', label='Attunement')
        #     ax3.axvline(x=config.n_last_prices_state, color='black', linestyle='--', label='Attunement')
            # ax5.axvline(x=config.n_last_prices_state, color='black', linestyle='--', label='Attunement')
        
        if show:
            plt.show()

        if save != "":
            plt.savefig(save)

        plt.close()

        if config.competitor:
            self.plot_competition_trajectories(x, infos, show, save)

    def plot_competition_trajectories(self, x, infos, show, save):

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

        plt.xlabel("t")

        # Profits Agent / Competitor
        ax1.set_title('Reward')
        ax1.step(x, infos[f'total_reward'], color='black', label='agent', where='post')
        ax1.step(x, infos[f'total_competitor_reward'], color='green', label='competitor', where='post')

        # Prices
        ax2.set_title("Offer Prices")
        ax2.step(x, infos[f'agent_offer_price'], color='black', label='agent', where='post')
        ax2.step(x, infos[f'competitor_offer_price'], color='green', label='competitor', where='post')

        # Customer Decisions Agent / Competitor
        ax3.set_title(f'Customer Decisions {self.customers[-1].name}')
        total_buying_decisions = np.sum([infos[f'n_{customer.name}_buy'] for customer in self.customers], axis=0)
        total_competitor_buying_decisions = np.sum([infos[f'n_{customer.name}_competitor_buy'] for customer in self.customers], axis=0)
        
        ax3.step(x, total_buying_decisions, color='black', label='agent', where='post')
        ax3.step(x, total_competitor_buying_decisions, color='green', label='competitor', where='post')

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels)
        plt.subplots_adjust(hspace=0.4)

        ax1.set_ylim()
        ax2.set_ylim(bottom=0, top=config.max_price + 1)
        # ax3.set_ylim(top=config.n_customers)
        #ax1.axvline(x=config.n_last_prices_state, color='black', linestyle='--', label='Attunement')
        #ax2.axvline(x=config.n_last_prices_state, color='black', linestyle='--', label='Attunement')
        #ax3.axvline(x=config.n_last_prices_state, color='black', linestyle='--', label='Attunement')
        
        if show:
            plt.show()

        if save != "":
            plt.savefig(save + '_c')

        plt.close()


    def plot_seasonal_diff(self, values, title):
        plt.plot(values)
        plt.title("Difference between actual and optimal " + title)
        plt.xlabel("Evaluation Steps")
        plt.savefig(f"{config.plot_dir}0_Seasonal_{title}_Diff")
        plt.close()


    def plot_prices(self, prices, save_prices=True):

        if save_prices:
            f = open(f'{config.summary_dir}data.txt', 'a')
            f.write("\n\nprices\n\n")
            f.write(str(prices))
            f.close()

        for s in range(config.week_length):
            y = np.array(prices[s]['mean'])
            std = np.array(prices[s]['std'])
            x = np.arange(len(y))
            plt.plot(x, y, 'k-')
            plt.fill_between(x, (y - std).clip(min=0), (y + std).clip(max=config.max_price), alpha=0.5)
            plt.title(f'Price for season {s}')
            plt.xlabel("Evaluation Steps")
            plt.ylabel("Price")
            plt.savefig(f'{config.plot_dir}prices/state_{s}')
            plt.close()


    def plot_rewards(self, rewards, save_rewards = True):

        if save_rewards:
            f = open(f'{config.summary_dir}data.txt', 'a')
            f.write("\n\nrewards\n\n")
            f.write(str(rewards))
            f.write("\n\n")
            f.close()

        plt.plot(rewards)
        plt.title("Rewards of the deterministic policy")
        plt.xlabel("Evaluation Steps")
        plt.ylabel("Simulated reward of one episode")
        plt.savefig(f'{config.summary_dir}rewards')
        plt.close()

    def print_seasonal_policy_stats(self, actual_prices, expected_profits_per_customer, perfect_prices, perfect_profits_per_customer):

        self.write_output('\nStatistics for Seasonal Customers:\n')

        # ATTENTION: This is not 100% correct since the other states are not always 0
        self.write_output(f"\nACTUAL (empty waiting pool)\n")
        self.seasonal_policy_writer(actual_prices, expected_profits_per_customer)

        self.write_output("\nOPTIMAL\n")
        self.seasonal_policy_writer(perfect_prices, perfect_profits_per_customer)

        self.write_output("\nPERFORMANCE\n")
        self.write_output(f"Reaching {np.round(calculate_mean_difference(actual_prices, perfect_prices) * 100, 2)} % of optimal prices\n")
        self.write_output(f"Reaching {np.round(calculate_mean_difference(expected_profits_per_customer, perfect_profits_per_customer) * 100, 2)} % of optimal profits")


    def seasonal_policy_writer(self, prices, profits_per_customer):

        seasonal_index = config.customers_types.index('seasonal')
                
        profits_all_customers = np.multiply(profits_per_customer, config.n_customers * config.customer_mix[seasonal_index])

        week_rewards = np.sum(profits_all_customers)
        episode_rewards = week_rewards * int(config.episode_length / config.week_length)

        rounded_prices = [np.round(price, 2) for price in prices]
        rounded_profits_per_cust = [np.round(profit, 2) for profit in profits_per_customer]
        rounded_profits_all_cust = [np.round(profit, 2) for profit in profits_all_customers]
        rounded_optimal_week_rewards = np.round(week_rewards, 2)
        rounded_optimal_episode_rewards = np.round(episode_rewards, 2)

        output = f'Policy Statistics for Prices: {rounded_prices}\nProfits per seasonal Customer: {rounded_profits_per_cust}\nProfits all seasonal Customers: {rounded_profits_all_cust}\nReward per Week: {rounded_optimal_week_rewards}\nReward per Episode: {rounded_optimal_episode_rewards}\n'
        self.write_output(output)


# Evaluation Helper
    def plot_competition_one_picture(self, infos, show= False, save=""):

        infos = self.add_concatenated_infos(infos, extend_one=1-config.competitor)

        step = 0.5 if config.competitor else 1

        x = np.arange(stop=len(infos['total_reward']) / (config.competitor + 1), step=step)

        waiting_pool = False
        for customer in self.customers:
            waiting_pool = customer.ability_to_wait or waiting_pool

        if waiting_pool:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
            # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
            # ax4.set_ylim(bottom=0, top=config.max_waiting_pool + 105)
            # ax4.axvline(x=config.n_last_prices_state, color='black', linestyle='--', label='Attunement')
            ax5.set_title('Waiting Pool')
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

        plt.xlabel("t")

        for customer in self.customers:
            if customer.ability_to_wait:
                l3, = ax5.step(x, infos[f'n_{customer.name}_waiting'], label=customer.name, where='post')
            else:
                next(ax5._get_lines.prop_cycler)

        # Profits Agent / Competitor
        ax1.set_title('Reward')
        l1, = ax1.step(x, infos[f'total_reward'], color='black', label='agent', where='post')
        l2, = ax1.step(x, infos[f'total_competitor_reward'], color='green', label='competitor', where='post')

        # Prices
        ax2.set_title("Offer Prices")
        ax2.step(x, infos[f'agent_offer_price'], color='black', label='agent', where='post')
        ax2.step(x, infos[f'competitor_offer_price'], color='green', label='competitor', where='post')

        # Customer Decisions Agent / Competitor
        # ax3.set_title(f'Customer Decisions {self.customers[-1].name}')
        # total_buying_decisions = np.sum([infos[f'n_{customer.name}_buy'] for customer in self.customers], axis=0)
        # total_competitor_buying_decisions = np.sum([infos[f'n_{customer.name}_competitor_buy'] for customer in self.customers], axis=0)
        
        # ax3.step(x, total_buying_decisions, color='black', label='agent', where='post')
        # ax3.step(x, total_competitor_buying_decisions, color='green', label='competitor', where='post')

        ax3.set_title(f'Customer Decisions {self.customers[0].name}')
        ax3.step(x, infos[f'n_{self.customers[0].name}_buy'], color='black', label='agent', where='post')
        ax3.step(x, infos[f'n_{self.customers[0].name}_competitor_buy'], color='green', label='competitor', where='post')

        ax4.set_title(f'Customer Decisions {self.customers[-1].name}')
        ax4.step(x, infos[f'n_{self.customers[-1].name}_buy'], color='black', label='agent', where='post')
        ax4.step(x, infos[f'n_{self.customers[-1].name}_competitor_buy'], color='green', label='competitor', where='post')

        ax2.set_ylim(bottom=0, top=config.max_price + 1)
        ax5.set_ylim(bottom=0)
        #ax1.axvline(x=config.n_last_prices_state, color='black', linestyle='--')
        #ax2.axvline(x=config.n_last_prices_state, color='black', linestyle='--')
        #ax3.axvline(x=config.n_last_prices_state, color='black', linestyle='--')

        handles = [l1, l2, l3]
        labels = [l1.get_label(), l2.get_label(), l3.get_label()]

        fig.legend(handles, labels)

        plt.subplots_adjust(hspace=0.6)

        if show:
            plt.show()

        if save != "":
            plt.savefig(save)

        plt.close()