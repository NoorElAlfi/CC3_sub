from CybORG.Agents.Wrappers import PettingZooParallelWrapper
from Agent.DDQN import Agent;


agents = {f"blue_agent_{agent}": Agent() for agent in range(18)}

def wrap(env):
    return PettingZooParallelWrapper(env=env)

submission_name = 'Noor El Alfi'
submission_team = 'L1NNA'
submission_technique = 'DDQN' #vs. PPO
