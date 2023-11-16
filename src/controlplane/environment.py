import os
import numpy as np
import subprocess
import gymnasium as gym
from gymnasium import spaces

#const
INCREASE_TARGET_DELAY = 0
#DECREASE_TARGET_DELAY_SW3 = 1
#INCREASE_TARGET_DELAY_SW2 = 2
DECREASE_TARGET_DELAY = 1
#INCREASE_TARGET_DELAY_SW1 = 4
#DECREASE_TARGET_DELAY_SW1 = 5
DO_NOTHING = 2

FPS_INDEX = 0
BUFFER_INDEX = 1

TIMESTAMP_INDEX = 0
FIRST_SECOND = 0
FOURTH_SECOND = -1

BUFFER_SIZE_64 = 0
BUFFER_SIZE_32 = 1

class ControlPlaneEnvironment(gym.Env):
    def __init__(self, observations):
        self.observations = observations
        self.observation_space = None
        
        self.action_space = spaces.Discrete(3)

        self.reward = 0
        self.done = False
        self.current_state = None
        self.next_state = None

        self.buffer_size = []
        self.current_target_delay_sw1 = 50000
        self.current_target_delay_sw2 = 50000
        self.current_target_delay_sw3 = 50000

        self.target_delay_history_sw1 = [50000]
        self.target_delay_history_sw2 = [50000]
        self.target_delay_history_sw3 = [50000]
        self.actions_history = [6]
        self.reward_history = [0]
        self.fps_history = []

        self.received_dashStates = []
        self.received_intStates = []


    def take_action(self, action, intState, dashState):
    
        assert self.action_space.contains(action), f"Invalid action {action}"

        self.current_state = intState

        if action == INCREASE_TARGET_DELAY:
            # increase the target dely on S3
            self.current_target_delay_sw3 = min((self.current_target_delay_sw3 + 10000), 70000)
            self.target_delay_history_sw3.append(self.current_target_delay_sw3)
            cmd = "echo 'register_write MyEgress.targetDelay_reg 0 {0}' | simple_switch_CLI --thrift-port 9090 --thrift-ip 192.168.56.203".format(self.current_target_delay_sw3)
            subprocess.run(cmd, shell=True)

            # increase the target delay on S2
            self.current_target_delay_sw2 = min((self.current_target_delay_sw2 + 10000), 70000)
            self.target_delay_history_sw2.append(self.current_target_delay_sw2)
            cmd = "echo 'register_write MyEgress.targetDelay_reg 0 {0}' | simple_switch_CLI --thrift-port 9090 --thrift-ip 192.168.56.202".format(self.current_target_delay_sw2)
            subprocess.run(cmd, shell=True)
            
            # increase the target delay on S1
            self.current_target_delay_sw1 = min((self.current_target_delay_sw1 + 10000), 70000)
            self.target_delay_history_sw1.append(self.current_target_delay_sw1)
            cmd = "echo 'register_write MyEgress.targetDelay_reg 0 {0}' | simple_switch_CLI --thrift-port 9090 --thrift-ip 192.168.56.201".format(self.current_target_delay_sw1)
            subprocess.run(cmd, shell=True)
        elif action == DECREASE_TARGET_DELAY:
            # decrease the target dely on S3
            self.current_target_delay_sw3 = max((self.current_target_delay_sw3 - 5000), 20000)
            self.target_delay_history_sw3.append(self.current_target_delay_sw3)
            cmd = "echo 'register_write MyEgress.targetDelay_reg 0 {0}' | simple_switch_CLI --thrift-port 9090 --thrift-ip 192.168.56.203".format(self.current_target_delay_sw3)
            subprocess.run(cmd, shell=True)

            # decrease the target dely on S2
            self.current_target_delay_sw2 = max((self.current_target_delay_sw2 - 5000), 20000)
            self.target_delay_history_sw2.append(self.current_target_delay_sw2)
            cmd = "echo 'register_write MyEgress.targetDelay_reg 0 {0}' | simple_switch_CLI --thrift-port 9090 --thrift-ip 192.168.56.202".format(self.current_target_delay_sw2)
            subprocess.run(cmd, shell=True)

            # decrease the target dely on S1
            self.current_target_delay_sw1 = max((self.current_target_delay_sw1 - 5000), 20000)
            self.target_delay_history_sw1.append(self.current_target_delay_sw1)
            cmd = "echo 'register_write MyEgress.targetDelay_reg 0 {0}' | simple_switch_CLI --thrift-port 9090 --thrift-ip 192.168.56.201".format(self.current_target_delay_sw1)
            subprocess.run(cmd, shell=True)
        elif action == DO_NOTHING:
            self.target_delay_history_sw1.append(self.current_target_delay_sw1)
            self.target_delay_history_sw2.append(self.current_target_delay_sw2)
            self.target_delay_history_sw3.append(self.current_target_delay_sw3)

        self.received_intStates.append(intState)
        self.received_dashStates.append(dashState)

        self.actions_history.append(action)
        self.fps_history.append(dashState[FOURTH_SECOND][FPS_INDEX])
        
        if len(self.received_intStates) == 2:
            # state observed when the action was taken
            self.current_state = self.received_intStates[0]
            # resulting state after the taken action
            self.next_state = self.received_intStates[-1]    
            
            # dash state when the action was taken
            current_dash = self.received_dashStates[0]
            # dash state after the taken action
            next_dash = self.received_dashStates[-1]

            print("\ncurrent state 4th second timestamp: ", self.received_dashStates[0][FOURTH_SECOND][TIMESTAMP_INDEX], end='\r')
            print("next state 4th second timestamp: ", self.received_dashStates[-1][FOURTH_SECOND][TIMESTAMP_INDEX], end='\r')
            print("\n")

            self.calculate_reward(current_dash[FOURTH_SECOND][BUFFER_INDEX], next_dash[FOURTH_SECOND][BUFFER_INDEX], next_dash[FOURTH_SECOND][FPS_INDEX])
            
            self.received_intStates = []
            self.received_dashStates = []
        
        return self.current_state, self.next_state, self.reward, self.done, {}


    def calculate_reward(self, buffer_action_step, buffer_reward_step, fps_reward_step):
        
        if buffer_reward_step > buffer_action_step:    #increasing
       
            if buffer_reward_step > 30:
            
                self.reward = 2
            
            elif buffer_reward_step < 30:
                
                if fps_reward_step == 30:
                    
                    self.reward = 1

                elif fps_reward_step == 24:
                    
                    self.reward = .5
                
                else:
                
                    self.reward = .1
        
        if buffer_reward_step < buffer_action_step:   #decreasing
            
            if buffer_reward_step > 30:
            
                self.reward = 2
            
            elif buffer_reward_step < 30:
                
                if fps_reward_step == 30:
                    
                    self.reward = 1

                elif fps_reward_step == 24:
                    
                    self.reward = .5
                
                else:
                
                    self.reward = -2
            

        self.reward_history.append(self.reward)


    def reset(self):
        self.reward = 0
        self.done = False
        
        self.observation_space = self.observations[FOURTH_SECOND]

        return self.observation_space.values
    

    def reset_metrics(self):
        self.target_delay_history_sw1 = [50000]
        self.target_delay_history_sw2 = [50000]
        self.target_delay_history_sw3 = [50000]
        self.actions_history = [6]
        self.reward_history = [0]
        self.fps_history = []