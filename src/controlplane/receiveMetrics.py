import tail 
import threading
import pandas as pd
import numpy as np
from collections import deque
import datetime
from agent import DDQN
from environment import ControlPlaneEnvironment
import os
import subprocess
import csv
from io import StringIO
import argparse
import time 

#const
INT_TYPE = 0
DASH_TYPE = 1
FIRST_SECOND = 0
FOURTH_SECOND = -1
DEFAULT_QUEUE_SIZE = 64
DEFAULT_TARGET_DELAY = 50000


def sendtoRl(sample):
    global ddqn
    global env
    global experiment_id

    
    df_INT = sample.iloc[:,:19]
    df_dash = sample.iloc[:,-3:]

    env.observation_space = df_INT.values
    current_state = env.observation_space
    dash_state = df_dash.values

    state_dim = df_INT.shape[1]

    training_start = datetime.datetime.now()
    print("Received state, entering in the loop steps: {0}\n".format(training_start), end='\r')


    action = ddqn.epsilon_greedy_policy(current_state[FOURTH_SECOND].reshape(-1, state_dim))

    current_state, next_state, reward, done, _ = env.take_action(action, current_state, dash_state)
    print(reward)

    if next_state is not None:
        print("next state received, memorizing transition", end='\r')

        ddqn.memorize_transition(current_state[FOURTH_SECOND],
                                 env.actions_history[-2], #action performed before reward
                                 reward,
                                 next_state[FOURTH_SECOND],
                                 0.0 if done else 1.0)

        if ddqn.train:
            ddqn.experience_replay()
        
    
    training_end = datetime.datetime.now()
    print("Exiting the loop steps: {0}".format(training_end), end='\r')

    print("\n========================================\n", end='\r')
    print("last action: {0} | reward: {1} | fps: {2} | "
          "Target_Delay (sw1): {3:.0f} | Target_Delay(sw2): {4} | Target_Delay(sw3): {5}".format(
           env.actions_history[-1], env.reward_history[-1],
           env.fps_history[-1], env.target_delay_history_sw1[-1], 
           env.target_delay_history_sw2[-1], env.target_delay_history_sw3[-1]), end='\r')
    print("\n========================================\n", end='\r')

    save_training_logs(env, ddqn, experiment_id)



def jointoRl(sample, t):
    global sampleJoin
    global aggregatedINT
    global aggregatedDash


    observationSpace = 4
    
    sampleJoin[t] = sample

    if len(sampleJoin) == 2:

        df = sampleJoin[INT_TYPE].merge(sampleJoin[DASH_TYPE], on=['timestamp', 'timestamp'], how='left', indicator=False)
        df = changeNaNtoMean(df)
        
        #Check if all FPS values are NaN
        if df['FPS'].isnull().all() == False:

            if df['timestamp'].nunique() == observationSpace:
                print("Sending sample to RL", end='\r')
                sendtoRl(df)

def removeFeaturesSameValue(dataset):

    dataset.drop(columns=['switchID_t3','ingress_port3','egress_port3','egress_spec3',
                          'switchID_t2','ingress_port2','egress_port2','egress_spec2',
                          'switchID_t1','ingress_port1','egress_port1','egress_spec1'], inplace=True)
    return dataset

def changeNaNtoMean(dataset):

    dataset = dataset.fillna(dataset.mean())

    return dataset

def AggregatedSampleINT(aggregatedSample):
    global aggregatedINT

    observationSpace = 4
 
    if aggregatedINT['timestamp'].nunique() < (observationSpace+1):
        
        aggregatedSample = removeFeaturesSameValue(aggregatedSample)
        aggregatedINT = pd.concat([aggregatedINT, aggregatedSample], ignore_index=True)
        
        if aggregatedINT['timestamp'].nunique() == observationSpace:
            jointoRl(aggregatedINT, t=INT_TYPE)
            aggregatedINT = aggregatedINT[0:0]


def AggregatedSampleDash(aggregatedSample):
    global aggregatedDash

    observationSpace = 4
    
    aggregatedSample['timestamp'] = aggregatedSample['timestamp'].str[:11]
    aggregatedSample['timestamp'] = aggregatedSample['timestamp'].astype(int)
    
    if aggregatedDash['timestamp'].nunique() < (observationSpace+1):
    
        aggregatedDash = pd.concat([aggregatedDash, aggregatedSample], ignore_index=True)
        
        if aggregatedDash['timestamp'].nunique() == observationSpace:
            jointoRl(aggregatedDash, t=DASH_TYPE)
            aggregatedDash = aggregatedDash[0:0]


def dashMetrics(line):
    np.set_printoptions(precision=2, formatter={'float_kind':'{:16.2f}'.format})

    if len(line.split(";")[0]) != 14:
        
        DataDashZero = {'timestamp': [str(0)],
                        'FPS': [0],
                        'bufferLocal': [0],
                        'BitrateCalc': [0]}
        
        sampleDashZero = pd.DataFrame.from_dict(DataDashZero)
        AggregatedSampleDash(sampleDashZero)
    else:
        DataDash = {'timestamp': [str(line.split(";")[0])],
                'FPS': [float(line.split(";")[2])],
                'bufferLocal':  [float(line.split(";")[1])],
                'BitrateCalc': [float(line.split(";")[3])]}
        
        sampleDash = pd.DataFrame.from_dict(DataDash)
        AggregatedSampleDash(sampleDash)

def intMetrics(line):
    np.set_printoptions(precision=2, formatter={'float_kind':'{:16.2f}'.format})
    
    global sampleINT
    global count
    
    INTnames = ['timestamp','switchID_t3','ingress_port3','egress_port3','egress_spec3','ingress_global_timestamp3','egress_global_timestamp3',
                'enq_timestamp3','enq_qdepth3','deq_timedelta3','deq_qdepth3','switchID_t2','ingress_port2','egress_port2','egress_spec2',
                'ingress_global_timestamp2','egress_global_timestamp2','enq_timestamp2','enq_qdepth2','deq_timedelta2','deq_qdepth2',
                'switchID_t1','ingress_port1','egress_port1','egress_spec1','ingress_global_timestamp1','egress_global_timestamp1','enq_timestamp1',
                'enq_qdepth1','deq_timedelta1','deq_qdepth1'] 
    
    Data = StringIO(line)
    sampleINT_temp = pd.read_csv(Data, sep=",", names=INTnames)

    # Check if is a new second. If yes, start a count
    if  len(sampleINT) == 0:
        sampleINT = sampleINT_temp

    # Check if the sample is in the same second. If yes, 
    # add this new sample in the current matrix
    elif sampleINT.iloc[-1]['timestamp'] == sampleINT_temp.iloc[-1]['timestamp']:
        sampleINT = pd.concat([sampleINT, sampleINT_temp], ignore_index=True)
    
    # Check if the new sample is the next second. If yes,
    # send the current matrix to the next stages and restart
    # the process
    else:

        AggregatedSampleINT(sampleINT)
        sampleINT.drop(sampleINT.index, inplace=True)


def getDashMetrics():
    global experiment_id
    path='../logs/dash_{0}.txt'.format(experiment_id)
    t = tail.Tail(path)
    t.register_callback(dashMetrics)
    t.follow(s=0.1)


def getINTMetrics():
    global experiment_id
    path='../logs/log_INT_{0}.txt'.format(experiment_id)
    t = tail.Tail(path)
    t.register_callback(intMetrics)
    t.follow(s=0.1)


def instantiateDQN():
    gamma = .99,  # discount factor
    tau = 10000     # target network update frequency.
    
    architecture = (24, 24)  # units per layer
    learning_rate = 1e-3     # learning rate (default 1e-3)
    l2_reg = 1e-6            # L2 regularization
    
    replay_capacity = int(1e6)
    batch_size = 32
    minimum_experience_memory = 100 
    
    epsilon_start = 1.0
    epsilon_end = .01
    epsilon_decay_steps = 250 # 250 default
    epsilon_exponential_decay = .99

    num_actions = 3
    state_dim = 19  # INT features
    
    online_network_filepath='agent_models/online_network_GAN-BS_1000.h5'
    target_network_filepath='agent_models/target_network_GAN-BS_1000.h5'

   
    initialization='standard' # standard or pretrained
    experiment_type = 1       # 1 - train the dqn; 2 - inference only; 3 - transfer learnin with weight freeznig (doesnt work as expected)


    ddqn = DDQN(state_dim=state_dim,
                num_actions=num_actions,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay_steps=epsilon_decay_steps,
                epsilon_exponential_decay=epsilon_exponential_decay,
                replay_capacity=replay_capacity,
                architecture=architecture,
                l2_reg=l2_reg,
                tau=tau,
                batch_size=batch_size,
                minimum_experience_memory=minimum_experience_memory,
                initialization=initialization,
                online_network_filepath=online_network_filepath,
                target_network_filepath=target_network_filepath,
                experiment_type=experiment_type)

    return ddqn


def save_training_logs(environment, agent, experiment_id):
    with open('agent_logs/action_history_{0}.csv'.format(experiment_id), 'w+', newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], environment.actions_history))

    with open('agent_logs/reward_history_{0}.csv'.format(experiment_id), 'w+', newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], environment.reward_history))
    
    with open('agent_logs/fps_history_{0}.csv'.format(experiment_id), 'w+', newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], environment.fps_history))
    
    with open('agent_logs/sw1_target_delay_history_{0}.csv'.format(experiment_id), 'w+', newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], environment.target_delay_history_sw1))
    
    with open('agent_logs/sw2_target_delay_history_{0}.csv'.format(experiment_id), 'w+', newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], environment.target_delay_history_sw2))
    
    with open('agent_logs/sw3_target_delay_history_{0}.csv'.format(experiment_id), 'w+', newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], environment.target_delay_history_sw3))
    
    with open('agent_logs/loss_history_{0}.csv'.format(experiment_id), 'w+', newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], agent.losses))
    
    with open('agent_logs/q-values_history_{0}.csv'.format(experiment_id), 'w+', newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], agent.q_values))


def main():
    
    global sampleINT
    global aggregatedINT
    global aggregatedDash
    global sampleJoin
    global count
    global ddqn
    global env
    global experiment_id

    parser = argparse.ArgumentParser(description="Experiment execution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("number", type=str, help="Number of executions")
    parser.add_argument("type", type=str, help="TD, iRED, iRED-RL")
    args = parser.parse_args()
    
    config = vars(args)
    
    experiment_id = config['type']+"_"+config['number']

    
    print("Set default Target Delay in SW3")
    cmd = "echo 'register_write MyEgress.targetDelay_reg 0 {0}' | simple_switch_CLI --thrift-port 9090 --thrift-ip 192.168.56.203".format(DEFAULT_TARGET_DELAY)
    subprocess.run(cmd, shell=True)

    print("Set default Target Delay in SW2")
    cmd = "echo 'register_write MyEgress.targetDelay_reg 0 {0}' | simple_switch_CLI --thrift-port 9090 --thrift-ip 192.168.56.202".format(DEFAULT_TARGET_DELAY)
    subprocess.run(cmd, shell=True)

    print("Set default Target Delay in SW1")
    cmd = "echo 'register_write MyEgress.targetDelay_reg 0 {0}' | simple_switch_CLI --thrift-port 9090 --thrift-ip 192.168.56.201".format(DEFAULT_TARGET_DELAY)
    subprocess.run(cmd, shell=True)
    

    np.set_printoptions(precision=2)

       
    INTnames = ['timestamp','ingress_global_timestamp3','egress_global_timestamp3','enq_timestamp3','enq_qdepth3','deq_timedelta3','deq_qdepth3', 
                'ingress_global_timestamp2','egress_global_timestamp2','enq_timestamp2','enq_qdepth2','deq_timedelta2','deq_qdepth2',
                'ingress_global_timestamp1','egress_global_timestamp1','enq_timestamp1', 'enq_qdepth1','deq_timedelta1','deq_qdepth1'] 
    
    DASHnames = ['timestamp','FPS','bufferLocal','BitrateCalc']
    count = 0
    sampleINT = np.array([])
    sampleJoin = dict()
    aggregatedINT = pd.DataFrame([], columns=INTnames)
    aggregatedDash = pd.DataFrame([], columns=DASHnames)

    ddqn = instantiateDQN()
    env = ControlPlaneEnvironment(sampleJoin)
    
    dashthread = threading.Thread(target=getDashMetrics)
    dashthread.start()

    intthread = threading.Thread(target=getINTMetrics)
    intthread.start()

    time.sleep(3660)

    ddqn.save_agent(experiment_id)
    print("logs saved")
     
if __name__ == '__main__':
    main()
