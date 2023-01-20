
from CybORG import CybORG
import inspect

from SimpleAgents.DDQN import Agent
from CybORG.Agents import RandomAgent
from CybORG.Agents.Wrappers import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Simulator.Scenarios.FileReaderScenarioGenerator import FileReaderScenarioGenerator


MAX_STEPS_PER_GAME = 50
MAX_EPS = 100

def run_training_example(scenario):
    agent_name = 'Blue'
    path = str(inspect.getfile(CybORG))
    path = path[:-7] + f'/Simulator/Scenarios/scenario_files/{scenario}.yaml'
    sg = FileReaderScenarioGenerator(path)
    cyborg = CybORG(scenario_generator=sg)
    cyborg = OpenAIGymWrapper(agent_name=agent_name,
                              env=FixedFlatWrapper(cyborg))

    observation = cyborg.reset()
    action_space = cyborg.action_spaces
    #print(f"Observation size {len(observation)}, Action Size {action_space}")
    action_count = 0
    agent = Agent(observation,action_space)
    agent.set_initial_values(action_space, observation)
    for i in range(MAX_EPS):  # laying multiple games
        #print(f"\rTraining Game: {i}", end='', flush=True)
        reward = 0
        for j in range(MAX_STEPS_PER_GAME):  # step in 1 game
            action = agent.get_action(observation, action_space)
            #if type(action) != int:
            #s    action = action.item()
            print(action)
            next_observation, r, done, info = cyborg.step(action)
            action_space = cyborg.action_space
            reward += r

            #agent.train(observation)  # training the agent
            observation = next_observation
            if done or j == MAX_STEPS_PER_GAME - 1:
                print(f"Training reward: {reward}")
                break
        #agent.end_episode()
        observation = cyborg.reset()
        action_space = cyborg.action_space
        agent.set_initial_values(action_space, observation)
        #print(reward)
if __name__ == "__main__":
    run_training_example('Scenario2')
