import config
from market import Market

import numpy as np

def simulate_policy(model, deterministic=True):

    e = Market()
    s_next = e.s

    infos = {}
    
    for i in range(config.episode_length):

        for attempt in range(10):
            try:
                action = model.predict(s_next, deterministic=deterministic)[0][0]
            except RuntimeError:
                print(f"RuntimeError on attempt {attempt}.")
                print(s_next)
            else:
                break

        s_next, reward, done, info = e.step(np.array([action])) # _ ,
    
        for key in info.keys():
            if i == 0:
                infos[key] = []
            infos[key].append(info[key])

        if done:
            e.reset()
    
    return infos