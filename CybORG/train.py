import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator

from datetime import datetime
from CybORG.Agents.Wrappers import PettingZooParallelWrapper
from CybORG.Agents.DDQN import AgentDDQN
from CybORG.Agents.PPO.PPO import PPO
import numpy as np
import os 
import torch
# cuda:0, cpu, or mps
device = 'cpu'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def wrap(env):
    return PettingZooParallelWrapper(env=env)


def run_training(name, team, name_of_agent, max_eps, write_to_file=False):

    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario3'

    sg = DroneSwarmScenarioGenerator()
    cyborg = CybORG(sg, 'sim')
    wrapped_cyborg = wrap(cyborg)
    observation = wrapped_cyborg.reset()
    if technique == 'DDQN':
        agents = {f"blue_agent_{agent}": AgentDDQN(
            wrapped_cyborg,
            observation,
            wrapped_cyborg.action_space(
                f'blue_agent_{agent}'
            ),
            f'blue_agent_{agent}',
            device=device
        ) for agent in range(18)}
        print(f'Using agents {agents}, if this is incorrect please update the code to load in your agent')
        if write_to_file:
            file_name = str(inspect.getfile(CybORG))[
                :-7] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S")
            print(
                f'Saving evaluation results to {file_name}_summary.txt and {file_name}_full.txt')
        start = datetime.now()

        print(f'using CybORG v{cyborg_version}, {scenario}\n')

        total_reward = []
        actions_log = []
        obs_log = []
        for i in range(max_eps):
            observations = wrapped_cyborg.reset()
            action_spaces = wrapped_cyborg.action_spaces
            r = []
            a = []
            o = []
            count = 0
            for j in range(500):
                actions = {agent_name: agent.get_action(
                    observations[agent_name]) for agent_name, agent in agents.items() if agent_name in wrapped_cyborg.agents}
                observations, rew, done, info = wrapped_cyborg.step(actions)
                for agent_name in agents:
                    agent_state = observations[agent_name] if agent_name in observations.keys(
                    ) else None
                    agent_action = actions[agent_name] if agent_name in actions.keys(
                    ) else action_spaces[agent_name].sample()
                    agent_reward = rew[agent_name] if agent_name in rew.keys(
                    ) else 0
                    agent_done = done[agent_name] if agent_name in done.keys(
                    ) else True
                    if type(agent_state) != np.ndarray:
                        pass
                    else:
                        agents[agent_name].memory.update(
                            agent_state, agent_action, agent_reward, agent_done)
                    agents[agent_name].train()
                if all(done.values()):
                    break
                r.append(mean(rew.values()))
                if write_to_file:
                    a.append({agent_name: wrapped_cyborg.get_action_space(agent_name)[
                            actions[agent_name]] for agent_name in actions.keys()})
                    o.append({agent_name: observations[agent_name]
                            for agent_name in observations.keys()})
            print(i, j, mean(r), stdev(r))
            total_reward.append(sum(r))
            for agent_name in agents:
                agents[agent_name].memory.reset()
            if write_to_file:
                actions_log.append(a)
                obs_log.append(o)

        end = datetime.now()
        difference = end-start
        print(
            f'Average reward is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
        print(f'file took {difference} amount of time to finish evaluation')
        if write_to_file:
            with open(file_name+'_summary.txt', 'w') as data:
                data.write(f'CybORG v{cyborg_version}, {scenario}\n')
                data.write(
                    f'author: {name}, team: {team}, technique: {name_of_agent}\n')
                data.write(
                    f'Average reward is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
                data.write(f'Using agents {agents}')

            with open(file_name+'_full.txt', 'w') as data:
                data.write(f'CybORG v{cyborg_version}, {scenario}\n')
                data.write(
                    f'author: {name}, team: {team}, technique: {name_of_agent}\n')
                data.write(
                    f'mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                    data.write(
                        f'actions: {act},\n observations: {obs} \n total reward: {sum_rew}\n')
    elif technique == 'PPO':
        agents = {f"blue_agent_{agent}": PPO(wrapped_cyborg.observation_space(f"blue_agent_{agent}").shape[0], len(wrapped_cyborg.get_action_space(f'blue_agent_{agent}')), 0.002, [0.9, 0.990], 0.99, 4, 0.2, True, 'CybORG\Evaluation\submission\Models\\5110.pth') for agent in range(18)}
        print(f'Using agents {agents}, if this is incorrect please update the code to load in your agent')
        if write_to_file:
            file_name = str(inspect.getfile(CybORG))[:-7] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S")
            print(f'Saving evaluation results to {file_name}_summary.txt and {file_name}_full.txt')
        start = datetime.now()

        print(f'using CybORG v{cyborg_version}, {scenario}\n')

        total_reward = []
        actions_log = []
        obs_log = []
        timestep = 0
        update_timestep = 2000
        for i in range(max_eps):
            observation = wrapped_cyborg.reset()
            r = []
            a = []
            o = []
            for j in range(1000):
                timestep += 1
                actions = {agent_name: agent.get_action(observation[agent_name], agent.memory) for agent_name, agent in agents.items() if agent_name in wrapped_cyborg.agents}
                observation, rew, done, info = wrapped_cyborg.step(actions)
                for agent_name, agent in agents.items():
                    if agent_name in actions:
                        if agent_name in wrapped_cyborg.agents:
                            agent.memory.rewards.append(rew[agent_name])
                            agent.memory.is_terminals.append(done[agent_name])
                        else:
                            agent.memory.rewards.append(-1)
                            agent.memory.is_terminals.append(True)
                if all(done.values()):
                    break
                if write_to_file:
                    a.append({agent_name: wrapped_cyborg.get_action_space(agent_name)[actions[agent_name]] for agent_name in actions.keys()})
                    o.append({agent_name: observation[agent_name] for agent_name in observation.keys()})
                r.append(mean(rew.values()))
                
                if timestep % update_timestep == 0:
                    for agent_name, agent in agents.items():
                        if agent_name in wrapped_cyborg.agents:
                            agent.update()
                            agent.memory.clear_memory()
                    timestep = 0
            total_reward.append(sum(r))
            if write_to_file:
                actions_log.append(a)
                obs_log.append(o)
            print(i)
        ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i))
        torch.save(agent.policy.state_dict(), ckpt)
        print('Checkpoint saved')
        end = datetime.now()
        difference = end-start
        print(
            f'Average reward is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
        print(f'file took {difference} amount of time to finish evaluation')
        if write_to_file:
            with open(file_name+'_summary.txt', 'w') as data:
                data.write(f'CybORG v{cyborg_version}, {scenario}\n')
                data.write(
                    f'author: {name}, team: {team}, technique: {name_of_agent}\n')
                data.write(
                    f'Average reward is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
                data.write(f'Using agents {agents}')

            with open(file_name+'_full.txt', 'w') as data:
                data.write(f'CybORG v{cyborg_version}, {scenario}\n')
                data.write(
                    f'author: {name}, team: {team}, technique: {name_of_agent}\n')
                data.write(
                    f'mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                    data.write(
                        f'actions: {act},\n observations: {obs} \n total reward: {sum_rew}\n')

if __name__ == "__main__":
    name = 'Noor El Alfi'
    team = 'L1NNA'
    folder = 'CybORG\Evaluation\submission'
    ckpt_folder = os.path.join(os.getcwd(), folder, 'Models')
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    technique = 'PPO'
    run_training(name, team, technique, 100, True)

