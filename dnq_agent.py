
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

from dnq import DQN


class DQNAgent:
    def __init__(self, input_shape, output_size, esplison, esplison_decay, continueTraining):
        self.model = DQN(input_shape, output_size, continueTraining, 'model')
        self.target = DQN(input_shape, output_size, False, 'target')
        self.target.copy_model(self.model)
        self.input_shape = input_shape
        self.output_size = output_size
        self.esplison = esplison
        self.esplison_decay = esplison_decay

    def model_predict(self, input):
        return self.model.predict(input)

    def target_predict(self, input):
        return self.target.predict(input)

    def train_model(self, x_train, y_train, sample_weight, epochs):
        return self.model.train(x_train, y_train, sample_weight, epochs)

    def save_model(self):
        self.model.save_model()
        self.model.save_model("./backup")

    def sync_target(self):
        self.target.copy_model(self.model)

    def get_action(self, state):
        if random.uniform(0.0, 1.0) > self.esplison:
            reward = self.model_predict(state)
            action_index = np.argmax(reward).item()
            print("Model selected action and rewards:", action_index, reward)
        else:
            action_index = random.randint(0, self.output_size - 1)
            print("Random selected action:", action_index)
        return action_index

    def decrease_esplison(self):
        self.esplison -= self.esplison_decay

    def train_with_experience(self, sampled_experiences, sample_weights, epoch, discount):
        state_index = 0
        action_index_experience = 1
        reward_index = 2
        done_index = 3
        next_state_index = 4

        train_x_image_state_before = np.array([experience[state_index]
                                               for experience in sampled_experiences])

        model_q_values = self.model_predict(train_x_image_state_before)

        train_x_image_state_after = np.array([experience[next_state_index]
                                              for experience in sampled_experiences])

        predicted_next_values = self.model_predict(
            train_x_image_state_after)

        target_q_values = self.target_predict(
            train_x_image_state_after)

        for j in range(len(sampled_experiences)):
            experience = sampled_experiences[j]
            model_q_value = model_q_values[j]
            model_q_value_original = np.array(model_q_value, copy=True)

            target_q_value = target_q_values[j]
            predicted_next_action_index = np.argmax(predicted_next_values[j])
            action_index = experience[action_index_experience]
            model_q_value[action_index] = experience[reward_index] + (discount *
                                                                      target_q_value[predicted_next_action_index] if experience[done_index] != True else 0)
            print("action index", action_index, "model_q_value_original", model_q_value_original, "model_q_value_updated",
                  model_q_value, "actual reward", experience[reward_index], "target reward", model_q_value[action_index])

        self.train_model(
            train_x_image_state_before, model_q_values, sample_weights, epoch)

        model_q_values_after = self.model_predict(
            train_x_image_state_before)

        errors = []
        for j in range(len(sampled_experiences)):
            experience = sampled_experiences[j]
            model_q_value_after = model_q_values_after[j]
            model_q_value = model_q_values[j]
            action_index = experience[action_index_experience]
            error = np.abs(
                model_q_value_after[action_index] - model_q_value[action_index])
            errors.append(error)

        return errors
