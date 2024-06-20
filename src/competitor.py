import config

class Undercutting_Vendor():

    def __init__(self):
        self.price = config.max_price
    
    def update_price(self, market_price):
        self.price = max(market_price - config.competitor_step, config.competitor_floor)
        return self.price