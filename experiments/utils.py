from autoins.common import common

def make_ccl_labeler(config, 
                        ccl_model, 
                        visualization = True, 
                        initialize = True):
    from autoins.ccl.labeler import CclLabeler
    return CclLabeler(ccl_model,
                    exp_dir = config['exp']['dir'], 
                    data_name = config['exp']['data'], 
                    ccl_id = config['exp']['ccl_id'],
                    max_beam_search = config['ccl']['labeler']['max_beam_search'],
                    nb_subgoal = config['ccl']['classifier']['nb_subgoal'],
                    initialize = initialize,
                    plot_cycle = config['ccl']['labeler']['plot_cycle'])


def make_ccl_model(config, feature_model, similarity_model, classifier_model, initialize = False):
    from autoins.ccl.model import CclModel
    return CclModel(exp_dir = config['exp']['dir'],
                    data_name = config['exp']['data'],
                    ccl_id = config['exp']['ccl_id'],
                    ag_shape = config['env']['ag_shape'],
                    nb_subgoal = config['ccl']['classifier']['nb_subgoal'],
                    feature_dim = config['ccl']['feature']['feature_dim'],
                    beta = config['ccl']['ccl']['beta'],
                    lamda = config['ccl']['ccl']['lamda'],
                    c1 = config['ccl']['ccl']['c1'],
                    learning_rate = config['ccl']['ccl']['learning_rate'],
                    contrastive_sample_size = config['ccl']['data']['contrastive_sample_size'],
                    feature_model = feature_model,
                    similarity_model = similarity_model,
                    classifier_model = classifier_model,
                    initialize = initialize)


def make_ccl_tester(config, ccl_model):
    from autoins.ccl.tester import CclTester
    return CclTester(ccl_model = ccl_model,
                    exp_dir= config['exp']['dir'],
                    ccl_id= config['exp']['ccl_id'],
                    data_name= config['exp']['data'],)


def make_ccl_trainer(config, ccl_model, ccl_labeler, data_generator, initialize = False):
    from autoins.ccl.trainer import CclTrainer
    return CclTrainer(model= ccl_model,
                        labeler = ccl_labeler,
                            data_generator = data_generator,
                            exp_dir= config['exp']['dir'],
                            ccl_id= config['exp']['ccl_id'],
                            data_name= config['exp']['data'],
                            relabel_freq = config['ccl']['ccl']['relabel_freq'],
                            batch_size = config['ccl']['ccl']['batch_size'],
                            nb_epoch = config['ccl']['ccl']['nb_epoch'],
                            initialize = initialize)

def make_classifier_model(config):
    from autoins import classifier
    return classifier.make(classifier_type = config['ccl']['classifier']['type'],
                            node_list = config['ccl']['classifier']['node_list'],
                            activation_list = config['ccl']['classifier']['activation_list'])

def make_contrastive_data_generator(config, distance_model):
    from autoins.ccl.data_generator import DataGenerator 
    return DataGenerator(
                distance_model = distance_model,
                exp_dir = config['exp']['dir'],
                ccl_id = config['exp']['ccl_id'],
                data_name = config['exp']['data'],
                ag_shape = config['env']['ag_shape'],
                nb_subgoal = config['ccl']['classifier']['nb_subgoal'],
                nb_data_gen = config['ccl']['data']['nb_data_gen'],
                contrastive_sample_size = config['ccl']['data']['contrastive_sample_size'])

def make_distance_model(config):
    from autoins import distance
    return distance.make(config['ccl']['distance']['type'], 
                            scale = config['ccl']['distance']['scale'])

def make_feature_model(config):
    from autoins import feature
    return feature.make(feature_type = config['ccl']['feature']['type'],
                           node_list = config['ccl']['feature']['node_list'],
                            activation_list = config['ccl']['feature']['activation_list'])

def make_graph_model(config):
    from autoins.rl.graph import GraphModel
    graph_model = GraphModel(exp_dir= config['exp']['dir'],
                                ccl_id= config['exp']['ccl_id'],
                                data_name= config['exp']['data'])
    return graph_model

def make_rl_env(config, ccl_model, svdd_model, graph_model):
    reward_type = config['rl']['reward_type']
    svdd_type = config['rl']['svdd_type']
    gamma = config['rl']['gamma']

    if config['exp']['data'] == 'ant':
        import gym
        import mujoco_maze
        from autoins.rl.rl_env import AntEnvWrapper
        env = gym.make('Ant4Rooms-v0')
        env = AntEnvWrapper(env, 
                            ccl_model, 
                            svdd_model, 
                            graph_model,
                            reward_type,
                            svdd_type,
                            gamma)

    elif config['exp']['data']=='fetch':
        import gym
        import gym_fetch_stack
        from autoins.rl.rl_env import FetchEnvWrapper
        env = gym.make('FetchStack2SparseStage3-v1')
        env = FetchEnvWrapper(env, 
                                ccl_model, 
                                svdd_model, 
                                graph_model,
                                reward_type,
                                svdd_type,
                                gamma)

    elif config['exp']['data']=='fetch_pretrain':
        import gym
        import gym_fetch_stack
        from autoins.rl.rl_env import FetchEnvWrapper
        env = gym.make('FetchStack2Stage3-v1')
        env = FetchEnvWrapper(env, 
                                ccl_model, 
                                svdd_model, 
                                graph_model,
                                reward_type,
                                svdd_type,
                                gamma)
    return env


def make_rl_trainer(config, env, restore = False):
    from autoins.rl.rl_trainer import RlTrainer
    return RlTrainer(env,
                        exp_dir = config['exp']['dir'],
                        ccl_id = config['exp']['ccl_id'],
                        data_name = config['exp']['data'],
                        rl_id = config['exp']['rl_id'],
                        pretrain_rl_id = config['exp']['pretrain_rl_id'],
                        learning_starts = config['rl']['learning_starts'],
                        learning_rate = config['rl']['learning_rate'],
                        gamma = config['rl']['gamma'],
                        use_her = config['rl']['use_her'],
                        restore = restore)

def make_reward_shaping_tester(config, env):
    from autoins.rl.tester import RewardShapingTester
    return RewardShapingTester(env,
                        exp_dir = config['exp']['dir'],
                        ccl_id = config['exp']['ccl_id'],
                        data_name = config['exp']['data'])



def make_similarity_model(config):
    from autoins import similarity
    return similarity.make(config['ccl']['similarity']['type'], 
                                units = config['ccl']['similarity']['units'])

def make_svdd(config, initialize):
    from autoins.svdd.svdd import Svdd
    return Svdd(exp_dir = config['exp']['dir'],
                ccl_id = config['exp']['ccl_id'],
                data_name = config['exp']['data'],
                node_list = config['svdd']['node_list'],
                l2_reg = config['svdd']['l2_reg'],
                ag_shape = config['env']['ag_shape'],
                quantile = config['svdd']['quantile'],
                initialize = initialize)

def make_svdd_tester(config, svdd):
    from autoins.svdd.tester import SvddTester
    return SvddTester(svdd,
                exp_dir = config['exp']['dir'],
                ccl_id = config['exp']['ccl_id'],
                data_name = config['exp']['data'])

