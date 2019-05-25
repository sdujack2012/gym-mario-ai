from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import pylab

from dnq_agent import DQNAgent
from memory_db import MemoryDB
from state_generator import StateGenerator
from training_parameters import esplison, esplison_decay, gamma, input_size, frame_size, stack_size, max_steps, render, max_episodes, sample_size, epoch, esplison, esplison_decay, experiences_before_training, training_before_update_target, e, a, beta, beta_increment_per_sampling, capacity, max_priority

if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    print(SIMPLE_MOVEMENT, env.action_space.n)
    # get size of state and action from environment
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    memorydb_instance = MemoryDB(
        e, a, beta, beta_increment_per_sampling, capacity, max_priority)
    agent_instance = DQNAgent(input_size, action_size,
                              esplison, esplison_decay, True)
    state_generator_instance = StateGenerator(frame_size, stack_size)
    scores, episodes = [], []

    for e in range(max_episodes):
        done = False
        score = 0
        raw_state = env.reset()
        state = state_generator_instance.get_stacked_frames(raw_state, True)

        steps = 0  # up to 500

        while not done and steps < max_steps:
            if render:  # if True
                env.render()
            steps += 1
            # get e greedy action
            action = agent_instance.get_action(np.array([state]))
            raw_state, reward, done, info = env.step(action)
            frame = 1
            while frame < stack_size and not done:
                raw_state, reward, done, info = env.step(action)
                frame += 1

            next_state = state_generator_instance.get_stacked_frames(
                raw_state, False)

            if info["flag_get"]:
                reward = 15

            new_experience = []
            new_experience.append(state)
            new_experience.append(action)
            new_experience.append(reward)
            new_experience.append(done)
            new_experience.append(next_state)
            memorydb_instance.add(new_experience)
            state = next_state

            if memorydb_instance.get_experiences_size() > experiences_before_training:
                sampled_experiences, b_idx, b_ISWeights = memorydb_instance.sample(
                    sample_size)
                errors = agent_instance.train_with_experience(
                    sampled_experiences, b_ISWeights, epoch, gamma)
                agent_instance.save_model()
                memorydb_instance.update_batch(
                    b_idx, errors, sampled_experiences)

            score += reward

            if done:
                 # every episode update the target model to be same with model (donkey and carrot), carries over to next episdoe
                agent_instance.sync_target()
                scores.append(score)
                episodes.append(e)
                # 'b' is type of marking for plot
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./mario.png")

        # save the model every 50th episode
        agent_instance.decrease_esplison()
        if e % 20 == 0:
            agent_instance.save_model()
