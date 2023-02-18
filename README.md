# Submission Information

- submission file located at [here](CybORG/Evaluation/submission/submission.py)
- model checkpoints are saved [here](CybORG/Evaluation/submission/Models)
- brief summary is [here](CybORG/Evaluation/submission/L1NNA_submission/L1NNA_CC3.pdf)
- pre-built docker image: [l1nna/cc3:v0.0.1](https://hub.docker.com/layers/l1nna/cc3/v0.0.1/images/sha256-c076e7b922c881f8e03a75d02d89a8daf9b0fb6f470552c416e103566102574f)
  - docker run -e PYTHONUNBUFFERED=1 l1nna/cc3:v0.0.1 
- 1000 episodes result: -4969.657 with a standard deviation of 2285.1910702561786

# Copyright DST Group. Licensed under the MIT license.

# Cyber Operations Research Gym (CybORG)

A cyber security research environment for training and development of security human and autonomous agents. Contains a common interface for both emulated, using cloud based virtual machines, and simulated network environments.

## Installation

Install CybORG locally using pip from the main directory that contains this readme

```
pip install -e .
```


## Creating the environment

Create a CybORG environment with the DroneSwarm Scenario that is used for CAGE Challenge 3:

```python
from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator

sg = DroneSwarmScenarioGenerator()
cyborg = CybORG(sg, 'sim')
```

The default_red_agent parameter of the DroneSwarmScenarioGenerator allows you to alter the red agent behaviour. Here is an example of a red agent that randomly selects a drone to exploit and seize control of:

```python
from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.SimpleAgents.DroneRedAgent import DroneRedAgent

red_agent = DroneRedAgent
sg = DroneSwarmScenarioGenerator(default_red_agent=red_agent)
cyborg = CybORG(sg, 'sim')
```


## Wrappers


To alter the interface with CybORG, [wrappers](CybORG/Agents/Wrappers) are avaliable.

 

* [OpenAIGymWrapper](CybORG/Agents/Wrappers/OpenAIGymWrapper.py) - alters the interface to conform to the OpenAI Gym specification. Requires the observation to be changed into a fixed size array.
* [FixedFlatWrapper](CybORG/Agents/Wrappers/FixedFlatWrapper.py) - converts the observation from a dictionary format into a fixed size 1-dimensional vector of floats
* [PettingZooParallelWrapper](CybORG/Agents/Wrappers/PettingZooParallelWrapper.py) - alters the interface to conform to the PettingZoo Parallel specification
    * [ActionsCommsPettingZooParallelWrapper](CybORG/Agents/Wrappers/CommsPettingZooParallelWrapper.py) - Extends the PettingZoo Parallel interface to automatically communicate what action an agent performed to other agents
    * [ObsCommsPettingZooParallelWrapper](CybORG/Agents/Wrappers/CommsPettingZooParallelWrapper.py) - Extends the PettingZoo Parallel interface to automatically communicate elements of an agent's observation to other agents
    * [AgentCommsPettingZooParallelWrapper](CybORG/Agents/Wrappers/CommsPettingZooParallelWrapper.py) - Extends the PettingZoo Parallel interface to allow agents to select what message they want to broadcast to other agents as part of the agent's action space

## How to Use

### OpenAI Gym Wrapper

The OpenAI Gym Wrapper allows interaction with a single external agent. The name of that external agent must be specified at the creation of the OpenAI Gym Wrapper.

```python
from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper

sg = DroneSwarmScenarioGenerator()
cyborg = CybORG(sg, 'sim')
agent_name = 'blue_agent_0'
open_ai_wrapped_cyborg = OpenAIGymWrapper(agent_name=agent_name, env=FixedFlatWrapper(cyborg))
observation, reward, done, info = open_ai_wrapped_cyborg.step(0)
```

### PettingZoo Parallel Wrapper

The PettingZoo Parallel Wrapper allows multiple agents to interact with the environment simultaneously.

```python
from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper

sg = DroneSwarmScenarioGenerator()
cyborg = CybORG(sg, 'sim')
open_ai_wrapped_cyborg = PettingZooParallelWrapper(cyborg)
observations, rewards, dones, infos = open_ai_wrapped_cyborg.step({'blue_agent_0': 0, 'blue_agent_1': 0})
```

### Ray/RLLib wrapper  
```python
from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env

def env_creator_CC3(env_config: dict):
    sg = DroneSwarmScenarioGenerator()
    cyborg = CybORG(scenario_generator=sg, environment='sim')
    env = ParallelPettingZooEnv(PettingZooParallelWrapper(env=cyborg))
    return env

register_env(name="CC3", env_creator=env_creator_CC3)
```
 


## Evaluating agent performance

To evaluate an agent's performance please use the [evaluation script](CybORG/Evaluation/evaluation.py) and the [submission file](CybORG/Evaluation/submission/submission.py).

Please see the [submission instructions](CybORG/Evaluation/submission/submission_readme.md) for further information on submission and evaluation of agents.

## Additional Readings
For further guidance on the CybORG environment please refer to the [tutorial notebook series.](CybORG/Tutorial)

## Citing this project
```
@misc{cage_cyborg_2022, 
  Title = {Cyber Operations Research Gym}, 
  Note = {Created by Maxwell Standen, David Bowman, Son Hoang, Toby Richer, Martin Lucas, Richard Van Tassel, Phillip Vu, Mitchell Kiely, KC C., Natalie Konschnik, Joshua Collyer}, 
  Publisher = {GitHub}, 
  Howpublished = {\url{https://github.com/cage-challenge/CybORG}}, 
  Year = {2022} 
}
```

