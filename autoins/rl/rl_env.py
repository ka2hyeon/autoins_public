import copy
import gym
import numpy as np

class EnvWrapper:
    def __init__(self, 
                    env, 
                    ccl_model, 
                    svdd_model, 
                    graph_model,
                    reward_type, 
                    svdd_type,
                    gamma):
        self.env = env
        self.ccl_model = ccl_model
        self.svdd_model = svdd_model
        self.graph_model =  graph_model
        self.reward_type = reward_type
        self.svdd_type = svdd_type
        self.gamma = gamma
        self._make_env()

    def _make_env(self):
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self.spec = self.env.spec
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def reset(self):
        return self.env.reset()

    def seed(self, seed):
        return self.env.seed(seed)

    def step(self):
        return self.env.step()

    def compute_reward(self, ag_list, g_list, info_list):
        original_reward = self.compute_original_reward(ag_list, g_list)

        ag_shaped = np.asarray([info['ag'] for info in info_list])
        next_ag_shaped = np.asarray([info['next_ag'] for info in info_list])
        shaped_reward = self.compute_reward_shaping(ag_shaped, next_ag_shaped, g_list)

        #print('ag_list:',ag_list.shape)
        #print('ag_shaped:',ag_shaped.shape)
        #print('original_reward:', original_reward.shape)
        #print('shaped_reward:', shaped_reward.shape)
        
        return original_reward + shaped_reward

    def compute_original_reward(self, ag0, g):
        raise NotImplementedError
    
    def compute_reward_shaping(self, ag0, ag1, g):
        if self.reward_type == 0:
            shaped = np.sum(np.zeros_like(ag0), axis = -1)     

        elif self.reward_type >= 1:
            if self.svdd_type == 'None':
                psi1 = self.svdd_model.predict_binary(ag1)
                psi0 = self.svdd_model.predict_binary(ag0)
                psi1 = np.ones_like(psi1)
                psi0 = np.ones_like(psi0)
                
            elif self.svdd_type == 'hard':
                psi1 = self.svdd_model.predict_binary(ag1)
                psi0 = self.svdd_model.predict_binary(ag0)

            elif self.svdd_type == 'soft':
                psi1 = self.svdd_model.predict_nonbinary(ag1)
                psi0 = self.svdd_model.predict_nonbinary(ag0)
            else:
                raise NotImplementedError
                
            if self.reward_type == 1:
                phi1_c = self.compute_phi_c(ag1, g) 
                phi1_g = self.compute_phi_g(ag1, g)            
                phi0_c = self.compute_phi_c(ag0, g) 
                phi0_g = self.compute_phi_g(ag0, g)
                phi1 = phi1_c + phi1_g
                phi0 = phi0_c + phi0_g
                #shaped = 1*(self.gamma*psi1*phi1-psi0*phi0)
                shaped = 1*(psi1*phi1-psi0*phi0)
                
            elif self.reward_type == 2:
                phi1 = self.compute_phi_c(ag1, g)
                phi0 = self.compute_phi_c(ag0, g)
                #shaped = 1*(self.gamma*psi1*phi1-psi0*phi0)
                shaped = 1*(psi1*phi1-psi0*phi0)

            elif self.reward_type == 3:
                phi1 = self.compute_phi_g(ag1, g)
                phi0 = self.compute_phi_g(ag0, g)
                #shaped = 1*(self.gamma*psi1*phi1-psi0*phi0)
                shaped = 1*(psi1*phi1-psi0*phi0)

            elif self.reward_type == 4:
                #shaped = 1*(self.gamma*psi1-psi0)
                shaped = 1*(psi1-psi0)

            else:
                raise NotImplementedError        
        return shaped


    def compute_phi_c(self, ag, g):
        goal = self.get_goal()
        goal_tile = np.tile(goal[None], [len(ag), 1])
        phi_c = self.ccl_model.predict_similarity(ag, 
                                            g, 
                                            input_type='state')
        return phi_c

    def compute_phi_g(self, ag, g):
        nb_subgoal = self.ccl_model.nb_subgoal
        #goal_tile =  np.tile(self._goal[None], [len(ag), 1])
        n0 = self.get_node(ag)
        n1 = self.get_node(g)

        phi_g = self.graph_model.dist_mat[n0,n1]

        #import IPython
        #IPython.embed()

        #print(f'node0:{n0},node1:{n1},phi_g:{-phi_g/nb_subgoal}')
        #return 1-(phi_g/nb_subgoal)
        return -(phi_g)
        
    def get_node(self, ag):

        feature = self.ccl_model.feature_model(ag).numpy()
        ag_label_prob = self.ccl_model.classifier_model(feature).numpy()
        ag_label = np.argmax(ag_label_prob, -1)
        return ag_label

    def get_ag(self):
        return NotImplementedError

    def get_goal(self):
        raise NotImplementedError

class AntEnvWrapper(EnvWrapper):
    def _make_env(self):
        self.observation_space = gym.spaces.Dict(
            achieved_goal = gym.spaces.box.Box(low = np.asarray([-np.inf]*2),
                                            high = np.asarray([np.inf]*2)),
            desired_goal = gym.spaces.box.Box(low = np.asarray([-np.inf]*2),
                                            high = np.asarray([np.inf]*2)),
            observation = self.env.observation_space)

        self.action_space = self.env.action_space

        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self.spec = self.env.spec
        self.max_step = 200

    def render(self):
        if self.env.env.viewer:
            self.env.env.viewer.cam.distance = 38 #76
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        ag = self.get_ag()
        dg = self.get_goal()
        
        obs_dict = dict(
            observation = np.copy(obs),
            achieved_goal = np.copy(ag),
            desired_goal = np.copy(dg)
        )
        self._count = 0
        return obs_dict

    def step(self, action):
        prev_ag = self.get_ag()
        obs, _, _, info = self.env.step(action)
        ag = self.get_ag()
        dg = self.get_goal()
        
        obs_dict = dict(
            observation = np.copy(obs),
            achieved_goal = np.copy(ag),
            desired_goal = np.copy(dg)
        )
        self._count += 1
        is_success = self.compute_is_success(ag, dg)
        done =  (self._count >= self.max_step) #or is_success
        
        info['ag'] = np.copy(prev_ag)
        info['next_ag'] = np.copy(ag)
        info['is_success'] = is_success
        
        reward = self.compute_reward(prev_ag[None] , dg[None], [info])[0]
        return obs_dict, reward, done, info
    
    def compute_original_reward(self, ag_list, g_list):
        reward = []
        for ag, g in zip(ag_list, g_list):    
            if self.compute_is_success(ag, g):
                reward.append(0.)
            else:
                reward.append(-1.)
        return np.asarray(reward)
        
    def get_ag(self):
        return np.copy(self.env.wrapped_env.get_xy())

    def get_goal(self):
        return np.copy(self.env.env._task.goals[0].pos)

    def compute_is_success(self, ag, dg):
        assert len(ag.shape) == len(dg.shape) == 1
        return np.linalg.norm(ag-dg) < 1


class FetchEnvWrapper(EnvWrapper):
    def _make_env(self):
        obs_space_dict = dict()
        obs_space_dict['achieved_goal'] = gym.spaces.box.Box( \
                                                low = np.asarray([-np.inf]*6), 
                                                high = np.asarray([np.inf]*6))
        obs_space_dict['desired_goal'] = gym.spaces.box.Box( \
                                                low = np.asarray([-np.inf]*6), 
                                                high = np.asarray([np.inf]*6))
        obs_space_dict['observation'] = self.env.observation_space['observation']
        
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self.max_step = 50

    def reset(self):
        original_obs_dict = self.env.reset()
        obs_dict = self._convert_obs_dict(original_obs_dict)
        self._step = 0
        return obs_dict

    def step(self, action):
        self._step +=1 

        ag = self.get_ag()
        goal = self.get_goal()
        original_next_obs_dict, _, env_done, env_info = self.env.step(action)
        next_obs_dict = self._convert_obs_dict(original_next_obs_dict)
        

        next_ag = next_obs_dict['achieved_goal']
        env_info['ag'] = np.copy(ag)
        env_info['next_ag'] = np.copy(next_ag)
        reward = self.compute_reward(next_ag[None], goal[None], [env_info])[0]

        if (self._step == self.max_step):
            env_done = True

        return next_obs_dict, reward, env_done, env_info

    def _convert_obs_dict(self, obs_dict):
        obs_dict_reformed = copy.copy(obs_dict)
        obs_dict_reformed['achieved_goal'] = obs_dict['achieved_goal'][0:6]
        obs_dict_reformed['desired_goal'] = obs_dict['desired_goal'][0:6]
        return obs_dict_reformed

    def compute_original_reward(self, ag_list, g_list):
        original = []
        for ag, g in zip(ag_list, g_list):
            ag_aug = np.concatenate([ag, np.zeros(3)], -1)
            g_aug = np.concatenate([g, np.zeros(3)], -1)
            original.append(self.env.compute_reward(ag_aug, g_aug))
        original = np.asarray(original)
        return original

    def get_ag(self):
        pass

    def get_goal(self):
        pass