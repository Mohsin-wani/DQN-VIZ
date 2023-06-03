#Double DQN algorithm 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import os
import uuid

class Environment:
    def __init__(self, env_name='CartPole-v1'):
    #def make_env(env_name = 'CartPole-v1'):
        self.env_name = 'CartPole-v1' 
        self.env = gym.make(env_name)
        self.state_shape, self.action_shape = self.env.observation_space.shape, self.env.action_space.shape
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
    def get_env(self):
        return self.env
    

class QNet(tf.keras.Model):
    def __init__(self, n_actions, num_layers = 4, dimensions = [256,128,64], activations = ['relu','relu','relu'], final_activation='linear'):
        super(QNet,self).__init__()
        self.mlp_layers = []
        for i in range(num_layers-1):
            self.mlp_layers.append(tf.keras.layers.Dense(dimensions[i], activation= activations[i]))
        self.mlp_layers.append(tf.keras.layers.Dense(n_actions, activation= final_activation))
        #print(self.mlp_layers)
    
    def call(self, state):
        x = state
        for i, mlp_layer in enumerate(self.mlp_layers):           
            x = mlp_layer(x)
            #print(f'inside call : {x}')
        out = x
        return out
    
class Cnn_Qnet(tf.keras.Model): 
    def __init__(self, n_actions, num_cnn_layers = 3, channels = [32,64,32], activations = ['relu','relu','relu'], final_activation = 'linear', filter_size = [(3,3),(3,3),(3,3)], strides= [2,2,2]):
            super(Cnn_Qnet,self).__init__()

            self.cnn_layers = []
            for i in range(num_cnn_layers):
                self.cnn_layers.append(tf.keras.layers.Conv2D(channels[i], filter_size[i], strides[i], activation='relu'))            

            self.fc1 = tf.keras.layers.Dense(128,activation = 'relu')
            self.fc2 = tf.keras.layers.Dense(n_actions, activation = final_activation)

    def call(self, state):
            
            x = state
            for i, cnn_layer in enumerate(self.cnn_layers):           
                x = cnn_layer(x)

            x = tf.keras.layers.Flatten()(x)

            x = self.fc1(x)
            out = self.fc2(x)
            return out

#maintains a replay buffer used for learning
class ReplayBuffer:
    def __init__(self, size=10000):
        self.size = int(size) #max number of items in buffer
        self.buffer = [] #array to hold buffer
        self.counter = 0
        self.next_id = 0
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        #print(f'size = {self.size}')
        #print(f'buffer length : {len(self.buffer)}')
        
        #if len(self.buffer) < self.size:
        if self.counter < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size
        if self.counter < self.size:
            self.counter += 1
        
    def sample(self, batch_size=32):
        idxs = np.random.choice(self.counter, batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        #print(f' states {states} actions : {actions} rewards : {rewards}:  next_states {next_states} dones flags : {done_flags}')

        return np.array(states,np.float32), np.array(actions,np.float32), np.array(rewards,np.float32), np.array(next_states,np.float32), np.array(done_flags)
    
    #converts all the numpy arrays returned by sample method to tensors of proper dimesnions
    def to_tensors(self, state_dim, act_dim=0):
        states, actions, rewards, next_states, done_flags = self.sample()
        #print(type(states))
        states = np.array(states,np.float32)
        states = np.reshape(states, (-1, state_dim))
    
        actions = np.reshape(actions, (-1))
        rewards = np.reshape(rewards,(-1,1))
        rewards = rewards.squeeze()

        next_states = np.array(next_states,np.float32)
        next_states = np.reshape(next_states, (-1, state_dim))
    
        done_flags = np.reshape(done_flags, (-1,1))
        done_flags = np.squeeze(done_flags)

        #print(f' states {states} actions : {actions} rewards : {rewards}: next_states {next_states} dones flags : {done_flags}')

        state_ts = tf.convert_to_tensor(states, dtype= tf.float32)
        action_ts = tf.convert_to_tensor(actions, dtype=tf.int32)
        reward_ts = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_state_ts = tf.convert_to_tensor(next_states,dtype=tf.float32)
    
        #print(f'Tensor states {state_ts} actions : {action_ts} rewards : {reward_ts}:  next_states {next_state_ts} dones flags : {done_flags}')

        return state_ts, action_ts, reward_ts, next_state_ts, done_flags
    
    def save_buffer(self, training_episode, path = None):
        ##change at deployment
        idxs = len(self.buffer)
        samples = [self.buffer[i] for i in range(idxs)]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))

        s, a, r, s_, d = np.array(states,np.float32), np.array(actions,np.float32), np.array(rewards,np.float32), np.array(next_states,np.float32), np.array(done_flags)

        
        os.chdir(path)
        os.makedirs(f'Replay_Buffer_{training_episode}') 
        dir = os.path.join(path, f'Replay_Buffer_{training_episode}')  
        os.chdir(dir)
        #print(dir) 
        state_file = os.path.join(dir, f'State_{training_episode}.csv')
        action_file = os.path.join(dir, f'Action_{training_episode}.csv')
        reward_file = os.path.join(dir, f'Reward_{training_episode}.csv')
        next_state_file = os.path.join(dir, f'Next_State_{training_episode}.csv')
        done_file = os.path.join(dir, f'Done_{training_episode}.csv')

        file_name = f'buffer_{training_episode}'
            
        path = os.path.join(dir, file_name)

        np.savetxt(state_file, s, delimiter=",")
        np.savetxt( action_file, a, delimiter=",")
        np.savetxt(reward_file, r, delimiter=",")
        np.savetxt( next_state_file, s_, delimiter=",")
        np.savetxt( done_file, d, delimiter=",")

        with open(path, 'w') as f:
            for state, action, reward, next_state, done in zip(s,a,r,s_,d):
                line = f'State : {state} Action : {action} Reward: {reward} Next State : {next_state} Done : {done}'
                f.write(line)
                f.write('\n')

    def initialize(self,env, initial_steps=500):
        state = env.reset()
        for i in range(initial_steps):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
        #   print(f' s: {state} action {action} reward {reward} next state : {next_state} done : {done}')
            self.add(state, action, reward, next_state, done)
            if done:
                state = env.reset()
            state = next_state
    

class Utils:
    def __init__(self):
        pass
    def epsilon_greedy_policy(self,state, env, agent, eps = 0.5):
        if np.random.rand() < eps:
        #print('rnd')
            return env.action_space.sample()        
        else:
            q_val = agent(state[np.newaxis])
        #print(q_val[0])
            return np.argmax(q_val[0])

    def epsilon_schedule(self, episode,limit = 500):
        return max(1-episode/400,0.01)
        
    def test_agent(self,env, network, num_test_episodes, max_ep_len, disp=False):
        ep_rets, ep_lens = [], []
        for j in range(num_test_episodes):
            state, done, ep_ret, ep_len = env.reset(), False, 0, 0
            while not(done or (ep_len == max_ep_len)):
                if disp:
                    env.render()
                #print(state)
                qvals = network(state[np.newaxis])
                action = np.argmax(qvals.numpy()[0])
                #print(action)
                #act1 = np.array(act1, np.float32)    
                ##act1 = act1.squeeze(1)    
                state_, reward, done, _ = env.step(action)
                state = state_           
                ep_ret += reward
                ep_len += 1
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
        return np.mean(ep_rets), np.mean(ep_lens)
    
    def record_video(self, env, network, num_test_episodes, episode, max_ep_len=200, path = None):              
        
        os.chdir(path)
        os.makedirs(f'Recordings_{episode}') 
        dir = os.path.join(path, f'Recordings_{episode}')
        os.chdir(dir)

        test_env = gym.wrappers.record_video.RecordVideo(env,f'video_{episode}')
        rets, len = self.test_agent(test_env, network, num_test_episodes, max_ep_len)    
        return rets, len
    
    def plot(self,rewards, loss_per_episode, episode, path=None):
        os.chdir(path)
        os.makedirs(f'Plots_{episode}') 
        dir = os.path.join(path, f'Plots_{episode}')
        os.chdir(dir)

        fig = plt.figure(figsize=(12,6))
        fig.suptitle("DQN")

        ax1 = fig.add_subplot(1,2,1)
        ax1.set_title(f'Reward till episode : {episode}')
        ax1.plot(rewards, 'b-+', label="rewards")
    
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_title(f'Loss till episode {episode}')
        ax2.plot(loss_per_episode, 'r-+', label="Loss")
        fig.show()
        fig.savefig(f'Plot_{episode}.png')

class Agent:
    def __init__(self, env, hyp_params):
        self.env = env
        self.hyp_params = hyp_params
        self.replay_buffer = ReplayBuffer(hyp_params.buffer_size)
        self.state_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.network = QNet(self.n_actions)
        self.target_net =  QNet(self.n_actions)

        self.num_episodes = hyp_params.num_episodes
        self.num_steps = hyp_params.steps_per_epoch
        self.target_update_ctr = 10
        self.rewards = []
        self.loss_per_episode = []        
        self.eps_loss = 0.00

        self.util = Utils()

        self.target_net.set_weights(self.network.get_weights())

        s = env.reset()
        q1 = self.network(s[np.newaxis])
        q2 = self.target_net(s[np.newaxis])
    
    
    #Double DQN algorithms using bellman update to learn
    def compute_loss_ddqn(self, states, actions, rewards, next_states, done_flags, gamma = 0.99):


        pred_qs = self.network(states)
        indices = tf.range(len(actions))
        pred_indices = tf.transpose([indices, actions])
        qval_preds = tf.gather_nd(pred_qs,pred_indices)
        
        target_qs = self.network(next_states)
        target_acts = tf.argmax(target_qs, axis=-1, output_type = tf.dtypes.int32)

        target_qvals = self.target_net(next_states)
        #t_indices = tf.range(len(actions))
        tar_indices = tf.transpose([indices,target_acts])
        

        qval_target = tf.gather_nd(target_qvals,tar_indices)

        calc_tar = rewards + gamma*(1-done_flags)*qval_target

        loss = tf.keras.losses.MSE(qval_preds,calc_tar)

        return loss        
        

    def dqn(self):

        for episode in range(self.num_episodes):
            state = self.env.reset()
            self.replay_buffer.initialize(self.env)            
            eps = self.util.epsilon_schedule(episode)
            self.eps_loss = 0.0

            for step in range(self.num_steps):
                action = self.util.epsilon_greedy_policy(state,self.env,self.network, eps)

                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(state,action,reward,next_state,done)

                if done:
                    break

                s,a,r,s_,d = self.replay_buffer.to_tensors(self.state_dim)
                state = next_state

                with tf.GradientTape() as tape:
                    loss = self.compute_loss_ddqn(s,a,r,s_,d)
                gradients = tape.gradient(loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients,self.network.trainable_variables))

                if step % self.target_update_ctr == 0:
                    self.target_net.set_weights(self.network.get_weights())

                self.eps_loss += loss
                
            self.rewards.append(step)
            self.loss_per_episode.append(self.eps_loss)
            #print(f'episode : {episode} reward : {step} loss: {self.eps_loss}')
            ###record videos
            if self.hyp_params.record == True:
                if episode >= self.hyp_params.start_eps_rec:
                    if episode % self.hyp_params.frequency_rec == 0 and episode != 0:                                   
                        self.util.record_video(self.env,self.network, 2, episode, self.hyp_params.req_reward, path = "/results") # self.hyp_params.path) ## change results to self.hyp_params.path if executing on your own machine
                    if episode % self.hyp_params.frequency_buffer == 0 and episode != 0:                                   
                        self.replay_buffer.save_buffer(episode,path=self.hyp_params.path)
                    if episode % self.hyp_params.frequency_plot == 0 and episode != 0:                                   
                        self.util.plot(self.rewards, self.loss_per_episode, episode, path=self.hyp_params.path)
                
            if episode % 20 == 0:
                rets, len = self.util.test_agent(self.env,self.network,5,self.hyp_params.req_reward)
                print(f'average return after {episode} : {rets} length : {len}')
                #self.replay_buffer.save_buffer(episode)
                if rets >= self.hyp_params.req_reward:
                    return


#print(str(uuid.uuid1()))

class Hyperparameters:
    ## Used to set learning rate, path variable for storing videos, plots and csv files, set buffer size
    def __init__(self, env, record = True, policy = 'mlp', learning_rate = 0.0001, path = None, buffer_size = 10000):
        uuid.uuid1()
        self.record = record
        self.policy = policy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.env = env
        if path == None:
            self.path = os.getcwd()
        else:
            self.path = path
        
        dir1 = f'DQN_{uuid.uuid1()}'
        
        os.makedirs(dir1) 
        dir = os.path.join(self.path, dir1)
        self.path = dir    

        if policy == 'mlp':
            self.set_mlp_params()
        elif policy == 'cnn':
            self.set_cnn_params()            

        self.set_dqn_params()
        self.set_recording_params()
        self.set_plotting_params()
        self.set_buffer_recording_params()

    def set_dqn_params(self, num_episodes = 300, steps_per_epoch = 300, req_reward=200):
        self.num_episodes = num_episodes
        self.steps_per_epoch = steps_per_epoch
        self.req_reward = req_reward
        
    def set_recording_params(self, start_eps = 100, frequency = 50):
        self.start_eps_rec = start_eps
        self.frequency_rec = frequency

    def set_plotting_params(self, start_eps = 100, frequency = 50):
        self.start_eps_plot = start_eps
        self.frequency_plot = frequency

    def set_buffer_recording_params(self, start_eps = 100, frequency = 50):
        self.start_eps_buffer = start_eps
        self.frequency_buffer = frequency

    def set_mlp_params(self, num_layers = 4, dimensions = [256,128,64], activations = ['relu','relu','relu','linear']  ):
        self.n_actions = self.env.action_space.n
        self.num_layers = num_layers
        self.dimensions = dimensions
        self.activations = activations

    def set_cnn_params(self, num_cnn_layers = 3, channels = [32,64,32], activations = ['relu','relu','relu'], final_activations = 'linear', filter_sizes = [(3,3),(3,3),(3,3)], strides= [2,2,2]):
        self.n_actions = self.env.action_space.n
        self.num_cnn_layers = num_cnn_layers
        self.channels = channels
        self.activations = activations
        self.final_activations = final_activations 
        self.filter_sizes = filter_sizes 
        self.strides = strides

    def set_loss_function(self, loss_fn):
        self.loss_fn = loss_fn

    def display_hyperparameters(self):
        print(f'lr : {self.learning_rate} buffer size = {self.buffer_size} path : {self.path}')


