import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator

from datetime import datetime
from CybORG.Agents.Wrappers import PettingZooParallelWrapper
from CybORG.Evaluation.evaluation2 import run_evaluation
from emu_agents import AgentPPO, AgentRDQN, AgentPPOSEQ
import numpy as np
import pickle
import os
from pathlib import Path

# cuda:0, cpu, or mps
device = 'cpu'

model_folder = 'emu_saved'


def wrap(env):
    return PettingZooParallelWrapper(env=env)


def run_training(name, team, name_of_agent, max_eps, write_to_file=False, agent_class='AgentPPO'):

    agent_folder = Path(os.path.join(
        model_folder, agent_class, name_of_agent))
    if agent_class == 'AgentPPO':
        agent_class = AgentPPO
    elif agent_class == 'AgentRDQN':
        agent_class = AgentRDQN
    elif agent_class == 'AgentPPOSEQ':
        agent_class = AgentPPOSEQ
    else:
        raise Exception('unknown agent')

    if not os.path.exists(agent_folder.parent):
        os.makedirs(agent_folder.parent)

    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario3'

    sg = DroneSwarmScenarioGenerator()
    cyborg = CybORG(sg, 'sim')
    wrapped_cyborg = wrap(cyborg)
    observation = wrapped_cyborg.reset()

    agents = {}
    for agent_index in range(18):
        agent_name = f"blue_agent_{agent_index}"
        agents[agent_name] = agent_class(
            wrapped_cyborg,
            observation[agent_name],
            wrapped_cyborg.action_space(
                agent_name
            ),
            agent_name=agent_name,
            device=device
        )
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
    train_steps = 0
    for i in range(max_eps):
        observations = wrapped_cyborg.reset()
        action_spaces = wrapped_cyborg.action_spaces
        r = []
        a = []
        o = []
        # cyborg.env.tracker.render()
        count = 0
        for j in range(500):
            actions_info = {agent_name: agent.get_action_with_prob(
                observations[agent_name]) for agent_name, agent in agents.items()
                if agent_name in wrapped_cyborg.agents}
            actions = {k: v[0] for k, v in actions_info.items()}
            actions_log_prob = {k: v[1] for k, v in actions_info.items()}
            observations_next, rew, done, info = wrapped_cyborg.step(actions)

            # print(wrapped_cyborg.env.environment_controller.reward)

            for agent_name in agents:
                agent_state = observations.get(agent_name)
                agent_state_next = observations_next.get(agent_name)
                agent_action = actions.get(agent_name)
                agent_action_log_prob = actions_log_prob.get(agent_name)
                agent_reward = rew.get(agent_name)
                agent_done = done.get(agent_name)

                if agent_done and j < 499:
                    # agent dead or win
                    agent_dw = True
                else:
                    # agent's final step
                    agent_dw = False

                # print(j, agent_name, agent_reward, agent_done,
                #       agent_action_log_prob, agent_state, agent_state_next)

                if agent_state is not None and agent_reward is not None and agent_action is not None:
                    train_steps += 1
                    # if successfully trained
                    if not agents[agent_name].train(
                        agent_state,
                        agent_state_next,
                        agent_action,
                        agent_action_log_prob,
                        agent_reward,
                        agent_done,
                        agent_dw,
                        train_steps
                    ):
                        train_steps -= 1
            if all(done.values()):
                break
            r.append(mean(rew.values()))
            observations = observations_next
            if write_to_file:
                # a.append({agent_name: str(cyborg.get_last_action(agent_name)) for agent_name in wrapped_cyborg.agents})
                a.append({agent_name: wrapped_cyborg.get_action_space(agent_name)[
                         actions[agent_name]] for agent_name in actions.keys()})
                o.append({agent_name: observations[agent_name]
                         for agent_name in observations.keys()})

        print(i, j, mean(r), stdev(r))
        total_reward.append(sum(r))
        for agent_name in agents:
            agents[agent_name].end_episode()
            # agents[agent_name].save(agent_folder)
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
    return agents


if __name__ == "__main__":
    # ask for a name
    # name = input('Name: ')
    # ask for a team
    # team = input("Team: ")
    # ask for a name for the agent
    # technique = input("Name of technique: ")
    name = 'Noor El Alfi'
    team = 'L1NNA'
    technique = 'AgentRDQN-shared-state-shared-memory'
    agents = run_training(name, team, technique, 100, agent_class='AgentPPOSEQ')
    run_evaluation(name, team, technique, 100,
                   agents, wrap, write_to_file=False)
