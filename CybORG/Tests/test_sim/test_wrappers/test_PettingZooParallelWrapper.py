import inspect

import pytest
import random
import numpy as np
from gym import spaces
from pettingzoo.test import parallel_api_test

from CybORG import CybORG
from CybORG.Agents import RandomAgent
from CybORG.Agents.Wrappers.CommsPettingZooParallelWrapper import AgentCommsPettingZooParallelWrapper, ActionsCommsPettingZooParallelWrapper, ObsCommsPettingZooParallelWrapper
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator

@pytest.fixture(scope="function", params=[PettingZooParallelWrapper, AgentCommsPettingZooParallelWrapper, ActionsCommsPettingZooParallelWrapper, ObsCommsPettingZooParallelWrapper])
def create_wrapped_cyborg(request):
    sg = DroneSwarmScenarioGenerator()
    cyborg = CybORG(scenario_generator=sg, seed=123)
    return request.param(env=cyborg)


@pytest.mark.skip('Agents are able to return to life')
def test_petting_zoo_parallel_wrapper(create_wrapped_cyborg):
    parallel_api_test(create_wrapped_cyborg, num_cycles=1000)



#Test if actions inputted are valid
def test_valid_actions():
    sg = DroneSwarmScenarioGenerator(num_drones=2, max_length_data_links=10000, starting_num_red=0)
    cyborg_raw = CybORG(scenario_generator=sg, seed=123)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    cyborg.reset()

    for i in range(50):
        actions = {}
        for agent in cyborg.active_agents:
            actions[agent] = random.randint(0, len(cyborg.get_action_space(agent))-1)

        obs, rews, dones, infos = cyborg.step(actions)
        for agent in cyborg.active_agents:
                assert cyborg.get_last_actions(agent) != 'InvalidAction'

#test reward bug 
def test_equal_reward():
    sg = DroneSwarmScenarioGenerator(num_drones=17, max_length_data_links=1000, starting_num_red=0)
    cyborg_raw = CybORG(scenario_generator=sg, seed=123)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    cyborg.reset()

    rews_tt = {}
    for i in range(10):
        actions = {}
        for agent in cyborg.agents:
            actions[agent] = random.randint(0,len(cyborg.get_action_space(agent))-1)

        obs, rews, dones, infos = cyborg.step(actions)
        rews_tt[i] = rews

    for i in rews_tt.keys():
        assert len(set(rews_tt[1].values())) == 1


def test_blue_retake_on_red():
    sg = DroneSwarmScenarioGenerator(num_drones=2, max_length_data_links=1000000, starting_num_red=1, red_spawn_rate=0,
                                    starting_positions=[np.array([0, 0]), np.array([0.1,0.1])])
    cyborg_raw = CybORG(scenario_generator=sg, seed=111)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    cyborg.reset()
    actions = {}

    if cyborg.active_agents[0] == 'blue_agent_0':
        agent = cyborg.active_agents[0]
        actions[cyborg.active_agents[0]]=1
    else:
        agent = cyborg.active_agents[0]
        actions[cyborg.active_agents[0]]=0


    assert len(cyborg.active_agents) == 1
    obs, rews, dones, infos = cyborg.step(actions)

    assert obs[agent][0] == 0
    assert len(cyborg.active_agents) == 2

def test_action_space():
    sg = DroneSwarmScenarioGenerator(num_drones=2, starting_num_red=1)
    cyborg_raw = CybORG(scenario_generator=sg, seed=123)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    cyborg.reset()

    breakpoint()
    for i in range(cyborg.action_space):
        actions = {}
        for j in range(len(cyborg.active_agents)):
            actions[cyborg.active_agents[j]] = i

        obs, rews, dones, infos = cyborg.step(actions)

        if i == 0:
            assert(cyborg.get_last_action(cyborg.active_agents[0]) == 'RetakeControl drone 0')
        elif i == 1:
            assert(cyborg.get_last_action(cyborg.active_agents[0]) == 'RetakeControl drone 1')
        elif i == 2:
            assert(cyborg.get_last_action(cyborg.active_agents[0]) == 'RemoveOtherSessions blue_agent_0')
        elif i == 3:
            assert(cyborg.get_last_action(cyborg.active_agents[0]) == 'BlockTraffic drone 0')
        elif i == 4:
            assert(cyborg.get_last_action(cyborg.active_agents[0]) == 'BlockTraffic drone 0')
        elif i == 5:
            assert(cyborg.get_last_action(cyborg.active_agents[0]) == 'AllowTraffic drone 0')
        elif i == 6:
            assert(cyborg.get_last_action(cyborg.active_agents[0]) == 'AllowTraffic drone 0')
        elif i == 7:
            assert(cyborg.get_last_action(cyborg.active_agents[0]) == 'Sleep')
       

def test_blue_remove_on_itself_no_red():
    sg = DroneSwarmScenarioGenerator(num_drones=2, max_length_data_links=100000, starting_num_red=0, red_spawn_rate=0,
                                    starting_positions=[np.array([0, 0]), np.array([1,1])])
    cyborg_raw = CybORG(scenario_generator=sg, seed=110)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    cyborg.reset()
    actions = {}

    for i in range(len(cyborg.active_agents)):
        actions[cyborg.active_agents[i]]=2

    assert len(cyborg.active_agents) == 2

    obs, rews, dones, infos = cyborg.step(actions)

    assert obs[cyborg.active_agents[i]][0] == 2
    assert len(cyborg.active_agents) == 2


def test_blue_retake_on_blue():
    sg = DroneSwarmScenarioGenerator(num_drones=2, max_length_data_links=100000, starting_num_red=0, red_spawn_rate=0,
                                    starting_positions=[np.array([0, 0]), np.array([1,1])])
    cyborg_raw = CybORG(scenario_generator=sg, seed=110)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    cyborg.reset()
    actions = {}
    actions['blue_agent_0']=1
    actions['blue_agent_1']=0

    assert len(cyborg.active_agents) == 2

    obs, rews, dones, infos = cyborg.step(actions)

    assert obs['blue_agent_0'][0] == 2
    assert obs['blue_agent_1'][0] == 2

    assert len(cyborg.active_agents) == 2


#test blocked IP bug
def test_block_and_check_IP():
    sg = DroneSwarmScenarioGenerator(num_drones=2, max_length_data_links=100000, starting_num_red=0, red_spawn_rate=0,
                                    starting_positions=[np.array([0, 0]), np.array([1,1])])
    cyborg_raw = CybORG(scenario_generator=sg, seed=110)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    cyborg.reset()

    actions = {}
    for i in range(2):
        count = 0
        for agent in cyborg.active_agents:
            actions[agent] = 4 -count
            count += 1
        obs, rews, dones, infos = cyborg.step(actions)

    assert obs['blue_agent_0'][2] == 1
    assert obs['blue_agent_1'][1] == 1


def test_attributes(create_wrapped_cyborg):
    # Create cyborg and reset it
    create_wrapped_cyborg.reset()

    # assert isinstance(obs, object) 
    assert isinstance(create_wrapped_cyborg.observation_spaces, dict)
    assert isinstance(create_wrapped_cyborg.rewards, dict)
    assert isinstance(create_wrapped_cyborg.dones, dict)
    assert isinstance(create_wrapped_cyborg.infos, dict)

    # Check action spaces is a dictionary
    assert isinstance(create_wrapped_cyborg.action_spaces, dict)

    # Check observation space is a dictionary
    assert isinstance(create_wrapped_cyborg.observation_spaces, dict)


def test_agent_data_change(create_wrapped_cyborg):
    create_wrapped_cyborg.reset()
    for agent in create_wrapped_cyborg.agents:
        assert isinstance(create_wrapped_cyborg.observation_space(agent), spaces.MultiDiscrete)
        assert isinstance(create_wrapped_cyborg.action_space(agent), spaces.Discrete)
        assert isinstance(create_wrapped_cyborg.get_reward(agent), float)
        assert isinstance(create_wrapped_cyborg.get_done(agent), bool)
        assert isinstance(create_wrapped_cyborg.infos, dict)

    actions = {}
    for agent in create_wrapped_cyborg.agents:
        actions[agent] = create_wrapped_cyborg.action_spaces[agent].sample()
    assert isinstance(actions, dict)
    obs, rews, dones, infos = create_wrapped_cyborg.step(actions)

    for agent in create_wrapped_cyborg.agents:
        assert isinstance(obs[agent], np.ndarray)
        assert isinstance(create_wrapped_cyborg.action_space(agent), spaces.Discrete)
        assert isinstance(rews[agent], float)
        assert isinstance(dones[agent], bool)
        assert isinstance(infos, dict)


def test_steps_random(create_wrapped_cyborg):
    '''
    Select n random actions and steps through the environment x times
    '''
    # Create cyborg and reset it
    create_wrapped_cyborg.reset()
    # Steps through the environment, takes actions, resets and repeats
    MAX_STEPS_PER_GAME = 20
    MAX_EPS = 5

    for i in range(MAX_EPS):
        for j in range(MAX_STEPS_PER_GAME):
            # Create a dictionary that contains the actions selected by every agent
            actions = {}
            for agent in create_wrapped_cyborg.agents:
                actions[agent] = create_wrapped_cyborg.action_spaces[agent].sample()
            assert isinstance(actions, dict)

            obs, rews, dones, infos = create_wrapped_cyborg.step(actions)
            if j == MAX_STEPS_PER_GAME - 1:
                break
        create_wrapped_cyborg.reset()


'''
def test_get_attr(cyborg):
    for attribute in ['observation_spaces','action_spaces','observation_space','action_space',
            'get_action_space', 'observation_change', 'get_rewards', 'get_reward', 'get_dones', 'get_done']:
        assert cyborg.get_attr(attribute) == cyborg.env.get_attr(attribute)
'''


def test_observation_change(create_wrapped_cyborg):
    # Create cyborg and reset it
    create_wrapped_cyborg.reset()
    initial_obs = create_wrapped_cyborg.observation_spaces
    for i in range(5):
        actions = {}
        for agent in create_wrapped_cyborg.agents:
            actions[agent] = create_wrapped_cyborg.action_spaces[agent].sample()

        obs, rews, dones, infos = create_wrapped_cyborg.step(actions)
        assert isinstance(obs[agent], np.ndarray)
        assert isinstance(rews, dict)
        assert isinstance(dones, dict)
        assert isinstance(infos, dict)

    final_obs = create_wrapped_cyborg.observation_spaces
    assert (initial_obs == final_obs)


def test_action_space(create_wrapped_cyborg):
    # Create cyborg and reset it
    create_wrapped_cyborg.reset()

    act_ss = create_wrapped_cyborg.action_spaces
    min_action_space_size = 1
    for agent in create_wrapped_cyborg.agents:
        act_s = act_ss[agent]
        assert isinstance(act_s, spaces.Discrete)


'''
def test_get_last_actions(cyborg):
    cyborg.reset()
    assert cyborg.get_last_action('Red') == cyborg.get_attr('get_last_action')('Red')
    assert cyborg.get_last_action('Blue') == cyborg.get_attr('get_last_action')('Blue')
    cyborg.step()
    assert cyborg.get_last_action('Red') == cyborg.get_attr('get_last_action')('Red')
    assert cyborg.get_last_action('Blue') == cyborg.get_attr('get_last_action')('Blue')
'''


def test_extreme_positions_drones():
    sg = DroneSwarmScenarioGenerator(num_drones=2, max_length_data_links=10000,
                                     starting_positions=[np.array([0, 0]), np.array([100, 100])])
    cyborg_raw = CybORG(scenario_generator=sg)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    for agent in cyborg.agents:
        obs = cyborg.get_observation(agent)
        assert cyborg.observation_space(agent).contains(obs)


def test_invalid_positions_drones():
    sg = DroneSwarmScenarioGenerator(num_drones=2, max_length_data_links=10000,
                                     starting_positions=[np.array([-1, -1]), np.array([101, 101])])
    cyborg_raw = CybORG(scenario_generator=sg)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    for agent in cyborg.agents:
        obs = cyborg.get_observation(agent)
        assert cyborg.observation_space(agent).contains(obs)


def test_active_agent_in_observation():
    sg = DroneSwarmScenarioGenerator(num_drones=20, max_length_data_links=10, starting_num_red=0)
    cyborg_raw = CybORG(scenario_generator=sg, seed=123)
    cyborg = PettingZooParallelWrapper(env=cyborg_raw)
    agents = {agent: RandomAgent() for agent in cyborg.possible_agents}
    action_spaces = cyborg.action_spaces
    for i in range(100):
        obs = cyborg.reset()
        for agent in cyborg.agents:
            assert agent in obs
            assert agent in action_spaces
            assert agent in agents
        for j in range(100):
            actions = {agent_name: agents[agent_name].get_action(obs[agent_name], action_spaces[agent_name])
                       for agent_name in cyborg.agents}
            obs, _, dones, _ = cyborg.step(actions)
            for agent in cyborg.agents:
                assert agent in obs
            if any(dones.values()):
                break

@pytest.mark.parametrize('num_drones', [2,10,18,25])
@pytest.mark.parametrize('wrapper', [PettingZooParallelWrapper, AgentCommsPettingZooParallelWrapper, ActionsCommsPettingZooParallelWrapper, ObsCommsPettingZooParallelWrapper])
def test_observation(num_drones, wrapper):
    sg = DroneSwarmScenarioGenerator(num_drones=num_drones)
    cyborg = wrapper(CybORG(scenario_generator=sg, seed=123))
    cyborg.reset()
    for i in range(10):
        for j in range(600):
            obs, rew, dones, infos = cyborg.step({agent: cyborg.action_space(agent).sample() for agent in cyborg.agents})
            for agent in cyborg.agents:
                if type(cyborg) == PettingZooParallelWrapper:
                    assert len(obs[agent]) == (num_drones*6)
                elif type(cyborg) == ObsCommsPettingZooParallelWrapper:
                    assert len(obs[agent]) == (num_drones*22)
                else:
                    assert len(obs[agent]) == (num_drones*7)
            if any(dones.values()) or len(cyborg.agents) == 0:
                assert all(dones)
                break
            if j > 499:
                breakpoint()
            assert j <= 500
        cyborg.reset()
