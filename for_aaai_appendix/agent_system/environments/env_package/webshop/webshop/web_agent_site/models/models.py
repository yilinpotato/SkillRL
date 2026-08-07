""
import random

random.seed(4)


class BasePolicy:
    def __init__(self):
        pass

    def forward(observation, available_actions):
        ""
        raise NotImplementedError


class HumanPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def forward(self, observation, available_actions):
        action = input('> ')
        return action


class RandomPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def forward(self, observation, available_actions):
        if available_actions['has_search_bar']:
            action = 'search[shoes]'
        else:
            try:
                action_arg = random.choice(available_actions['clickables'])
            except:
                action_arg = 'None'
            action = f'click[{action_arg}]'
        return action
