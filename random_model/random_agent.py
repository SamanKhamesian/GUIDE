from config import EnvConfig
import numpy as np

class RandomAgent:
    @staticmethod
    def get_action():
        action_type = np.random.choice([0, 1, 2])
        time_index = np.random.randint(0, 12)
        amount = 0.0

        if action_type == 0:
            pass

        elif action_type == 1:
            amount = np.random.uniform(*EnvConfig.CARB_RANGE)

        elif action_type == 2:
            amount = np.random.uniform(*EnvConfig.INSULIN_RANGE)

        action = (action_type, amount, time_index)
        return action