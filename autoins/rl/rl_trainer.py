import glob
import gym
import numpy as np
import os 
import time
import traceback
import matplotlib.pyplot as plt

from autoins.common import common, io
from stable_baselines3 import HerReplayBuffer
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True


class RlTrainer:
    def __init__(self,  
                    env,
                    exp_dir,
                    data_name,
                    ccl_id,
                    rl_id,
                    learning_starts,
                    learning_rate,
                    gamma,
                    use_her = False,
                    pretrain_rl_id = None,
                    restore = False):
        self.env = env
        self.exp_dir = exp_dir
        self.data_name = data_name
        self.ccl_id = ccl_id
        self.rl_id = rl_id
        self.gamma = gamma
        self.use_her = use_her
        self.learning_starts = learning_starts
        self.learning_rate = learning_rate
        self.weight_dir = f'{exp_dir}/weight/rl/{data_name}/{rl_id}'
        self.log_dir = f'{exp_dir}/log/rl/{data_name}/{rl_id}'
        self.fig_dir = f'{exp_dir}/figure/{data_name}/{rl_id}/rl'

        common.create_dir(self.fig_dir, clear_dir = True)

        if restore:
            list_of_files = glob.glob(f'{self.weight_dir}/*.zip') 
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f'loaded_file:{latest_file}')
            self.rl_model = self.make_model(latest_file)
        else:
            common.create_dir(self.weight_dir, clear_dir = True)
            common.create_dir(self.log_dir, clear_dir = True)
    
            if pretrain_rl_id:
                pretrain_weight_dir = f'{exp_dir}/weight/{data_name}/{pretrain_rl_id}'
                list_of_files = glob.glob(f'{pretrain_weight_dir}/*.zip') 
                latest_file = max(list_of_files, key=os.path.getctime)
                self.rl_model = self.make_model(latest_file)
            else:
                self.rl_model = self.make_model()

    def make_model(self, file_path = None):
        policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], 
                            qf=[512, 512, 512]),
                            clip_mean = 1.0) #2.0 # 1.0 for fetch 
        use_sde = False 
        action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(8), 1e-1*np.ones(8)) #1e-2

        if self.use_her:
            replay_buffer_class = HerReplayBuffer
            replay_buffer_kwargs=dict(
                            n_sampled_goal=4, #8 for fetch
                            goal_selection_strategy="future",
                            max_episode_length=self.env.max_step,
                            online_sampling=True)
        else:
            replay_buffer_class = None
            replay_buffer_kwargs = None

        if file_path:
            rl_model = SAC.load(file_path, 
                                env = self.env,
                                custom_objects = dict(tensorboard_log = self.log_dir,
                                                        learning_rate = self.learning_rate,
                                                        learning_starts = self.learning_starts,
                                                        #replay_buffer_kwargs = replay_buffer_kwargs,
                                                        use_sde = use_sde,
                                                        gamma = self.gamma,
                                                        action_noise = action_noise))
        else:
            rl_model = SAC('MultiInputPolicy', 
                                self.env, 
                                replay_buffer_class=replay_buffer_class,
                                replay_buffer_kwargs=replay_buffer_kwargs,
                                policy_kwargs=policy_kwargs,
                                gamma = self.gamma,
                                tau=0.0005,
                                #ent_coef = 0.15,
                                verbose=1,
                                learning_rate=self.learning_rate, 
                                learning_starts = self.learning_starts,
                                use_sde = use_sde,
                                tensorboard_log = self.log_dir,
                                action_noise = action_noise)
            common.create_dir(self.weight_dir, clear_dir = True)
        return rl_model


    def train(self,
            total_timesteps = 100):
        checkpoint_callback = CheckpointCallback(save_freq= 10000, #10000, #10000, #2500 #3000
                                                save_path=f'{self.weight_dir}')

        #action_noise = NormalActionNoise(np.zeros(8), 1e-1*np.ones(8))
        #self.rl_model.collect_rollouts(env = self.env, 
        #                                train_freq = 10000,
        #                                action_noise = action_noise,
        #                                replay_buffer = )

        try:
            self.rl_model.learn(total_timesteps=total_timesteps,
                                callback = checkpoint_callback)
        except:
            traceback.print_exc()
            import IPython
            IPython.embed()
        finally:
            self.rl_model.save(f'{self.weight_dir}/weight')
            #self.rl_model.save_replay_buffer(f'{self.weight_dir}/replay_buffer')
            print('weight succesfully saved')

    def rollout(self):
        nb_rollout = 10
        max_step = int(self.env.max_step/4)
        deterministic = False #True

        for i in range(nb_rollout):            
            total_reward = 0
            obs = self.env.reset()
            goal = self.env.get_goal()
            original_reward_list = [] 
            shaped_reward_list = []
            node_list = []

            for _ in range(max_step):
                prev_ag = obs['achieved_goal']
                action, _states = self.rl_model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward

                self.env.render()
                ag = obs['achieved_goal']
                node = self.env.get_node(ag[None])[0]
                original_reward = self.env.compute_original_reward(prev_ag[None],goal[None])[0]
                shaped_reward = self.env.compute_reward_shaping(prev_ag[None], ag[None], goal[None])[0]
                is_success = self.env.compute_is_success(ag, goal)

                original_reward_list.append(original_reward)
                shaped_reward_list.append(shaped_reward)
                node_list.append(node)
                print(f"node:{node}, reward:{reward:.4f}, ag: {ag}")
                #print(action)
                time.sleep(0.02)
                
                # for ant
                if done or is_success:
                    print("done")
                    break
            
            original_reward_list = np.asarray(original_reward_list)
            shaped_reward_list = np.asarray(shaped_reward_list)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(original_reward_list, label = 'original')
            ax.plot(original_reward_list+shaped_reward_list, label = 'shaped')
            ax.legend()
            fig.savefig(f'{self.fig_dir}/reward_{i:04d}.png')
            plt.close(fig)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.asarray(node_list))
            fig.savefig(f'{self.fig_dir}/node_{i:04d}.png')
            plt.close(fig)
            
