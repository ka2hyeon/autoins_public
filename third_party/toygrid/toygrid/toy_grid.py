import gym
import matplotlib.pyplot as plt
import numpy as np
import IPython

from gym import spaces
from .rrt import RRT

class ToyGridEnv(gym.Env):
    EMPTY = [255, 255, 255]
    WALL =  [0, 0, 0]
    AGENT = [0, 255, 0]
    GOAL =  [0, 0, 255]

    def __init__(self):
        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (-1., 1.)
        self.spec = None

        # Set these in ALL subclasses
        self.observation_space = spaces.Box(np.asarray([0, 7.5]), 
                                    np.asarray([0.5, 7.5]), 
                                    dtype = np.float32) 
        self.action_space = spaces.Box(np.asarray([-1, -1]), 
                                        np.asarray([1,1]), 
                                        dtype = np.float32) 

        # init
        self.max_step = 20
        self.goal = np.asarray([7, 7])
        self.world = self._make_world()
        self.seed()
        self.reset()
        
    def render(self):
        pass

    def close(self):
        return None

    def reset(self):
        if self._seed is not None:
            self.state = np.asarray([1., 1.], dtype = np.float32)
            return self.state
        else:
            while True:
                x = np.random.uniform(0.5, 7.5)
                y = np.random.uniform(0.5, 7.5)
                pos = np.asarray([x,y])
                if not self._check_collision(pos):
                    self.state = np.copy(pos)
                    self.nb_step = 0
                    return self.state

    def seed(self, seed = None):
        self._seed = seed
        return [seed]

    def step(self, act):
        new_state = np.copy(self.state)
        new_state[0] +=  act[0]
        new_state[1] +=  act[1]

        new_state[0] = np.clip(new_state[0], 0.5, 7.5)
        new_state[1] = np.clip(new_state[1], 0.5, 7.5)
        
        goal_dist = np.sqrt(np.sum(np.square(self.state-self.goal)))
        if goal_dist <= 1:
            reward = 1
        else:
            reward = 0

        done = bool(self.nb_step>self.max_step)
        info = {}
        self.state = np.copy(new_state)
        self.nb_step += 1
        return (new_state, reward, done, info)


    def _check_collision(self, pos):
        x_floor = int(np.floor(pos[0]))
        y_floor = int(np.floor(pos[1]))
        
        x_ceil = int(np.ceil(pos[0]))
        y_ceil = int(np.ceil(pos[1]))

        x_round = int(np.rint(pos[0]))
        y_round = int(np.rint(pos[1]))
        
        if (x_floor <1) or (y_floor<1):
            return True
        elif (x_ceil >7) or (y_ceil>7):
            return True
        elif np.array_equal(self.world[x_round, y_round,:], self.WALL):
            return True
        else:
            return False
    
    def _make_world(self):
        world = self.EMPTY*np.ones([9, 9, 3])
        
        ## side wall
        world[0,:] = self.WALL
        world[-1,:] = self.WALL
        world[:,0] = self.WALL
        world[:,-1] = self.WALL

        ## center wall (horizontal)
        world[4, 0:2] = self.WALL
        world[4, 3:6] = self.WALL
        world[4, 7:8] = self.WALL

        ## center wall (vertical)
        world[0:2, 4] = self.WALL
        world[3:6, 4] = self.WALL
        world[7:8, 4] = self.WALL

        ## GOAL
        #world[self.goal[0], self.goal[1]] = self.GOAL
        return world

    def render_path(self, path_list = None, label_list = None, save_path = None):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.imshow(self.world)
        #ax.scatter(self.state[0], self.state[1], s = 100, c = 'green')

        if path_list is not None:

            ## set color
            color_list = []
            plt_label_list = []
            ax_labels_list = []
            ax_color_list = []
            ax_int_list = []
            if label_list is not None:
                for label in label_list:
                    color = ['C%d'%i for i in label]
                    color_list.append(color)
                    plt_label = ['node %d'%i for i in label]
                    plt_label_list.append(plt_label)
                    ax_labels_list += plt_label
                    ax_color_list += color
                    ax_int_list += ['%04d'%i for i in label]
                ax_labels_list = list(dict.fromkeys(ax_labels_list))
                ax_color_list = list(dict.fromkeys(ax_color_list))
                ax_int_list = list(dict.fromkeys(ax_int_list))
            else:
                color_list = ['C0']*len(path_list)
                plt_label_list = ['']*len(path_list)
                ax_labels_list = ['']
                ax_color_list = ['C0']

            ## plot path
            for path, color, plt_label in zip(path_list, color_list, plt_label_list):
                path_array = np.asarray(path)
                #ax.plot(path_array[:,0], path_array[:,1], alpha = 0.3, color = color)
                ax.plot(path_array[:,0]-0.5, path_array[:,1]-0.5, alpha = 0.2, color = 'k', zorder =1)
                ax.scatter(path_array[:,0]-0.5, 
                            path_array[:,1]-0.5, 
                            s=20, 
                            alpha = 0.5, 
                            color = color,
                            zorder = 2)

        for ax_int in sorted(ax_int_list):
            idx = ax_int_list.index(ax_int)
            ax_color = ax_color_list[idx]
            ax_label = ax_labels_list[idx]
        
            ax.scatter([],
                        [],
                        color = ax_color,
                        label = ax_label,
                        alpha =1)

        ax.set_xticks([])
        ax.set_xticks([], minor = True)
        ax.set_yticks([])
        ax.set_yticks([], minor = True)
        ax.legend(loc='upper left', bbox_to_anchor = (1.04, 1))

        if save_path is not None:
            fig.savefig(save_path)
        #plt.show()
        plt.close(fig)

class ToyGridDemonstrator():
    def __init__(self, env):
        self.env = env
    
    def demonstrate(self):
        init_state = np.copy(self.env.state)
        final_state = np.asarray(self.env.goal)
        step_size = 0.2

        rrt = RRT(self.env.world, init_state, final_state, step_size)
        path = rrt.solve()
        return path

    def render_path(self, path_list = None, label_list = None, save_path = None):
        self.env.render_path(path_list, label_list, save_path)

if __name__ == '__main__':
    env = ToyGridEnv()
    demonstrator = ToyGridDemonstrator(env)

    path_list = []
    for _ in range(10):
        env.reset()
        path = demonstrator.demonstrate()
        if path is not None:
            path_list.append(path)
    demonstrator.render_path(path_list = path_list)


