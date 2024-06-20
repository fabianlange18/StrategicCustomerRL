# To Be Adjusted
run_name = '01_monopoly'
customers_types = ['seasonal']
customer_mix = [1]
competitor = False

# State Space
week_length = 7
max_waiting_pool = 1000
n_last_prices_state = 7

# Action Space
max_price = 10

# Market
n_customers = 50
gamma = 0.9999

# Customers
alpha = 4
beta = [4, 6, 7, 3, 6, 5, 7]
nothing_preference = 1

# Waiting Probabilities
p_remain = 0.95
p_return = 0.95

# Price-Aware and Anticipating
undercut_min = 0.9
max_buying_price = 7
n_last_prices_customer = week_length - 1

# Competitor
competitor_step = 1
competitor_floor = 1

# Training
episode_length = week_length * 10
n_episodes = 100000
learning_rate = 0.00003

# Evaluation
n_eval_episodes = 1000

# Logging Directories
summary_dir = f"./results/{run_name}/"
plot_dir = f"{summary_dir}plots/"

# Callbacks
peval_cb_n_episodes = n_episodes / 1000
sim_cb_n_episodes = n_episodes / 20