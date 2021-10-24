'''
To train rl:
    python -m experiments.main_rl --command train --config ./experiments/configure/ant/exp0/exp0_0.yaml

command: [train/check/test]
'''

import argparse

from networkx.generators.small import make_small_undirected_graph

from autoins.common import common, tf_setting
from experiments.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type = str, default = '')
    parser.add_argument('--config', type = str, default = '')
    return parser.parse_args() 

if __name__ == '__main__':
    args = parse_args()
    config = common.load_yaml(args.config)
    
    tf_setting.set_device()
    feature_model       = make_feature_model(config)
    similarity_model    = make_similarity_model(config)
    classifier_model    = make_classifier_model(config)    
    ccl_model           = make_ccl_model(config, feature_model, similarity_model, 
                                                classifier_model, initialize = False)
    ccl_model.restore()

    svdd_model = make_svdd(config, initialize = False)
    svdd_model.restore()

    graph_model = make_graph_model(config)

    #graph_model = make_graph_model(config)
    rl_env = make_rl_env(config, ccl_model, svdd_model, graph_model)

    if args.command == 'train':
        rl_trainer = make_rl_trainer(config, rl_env, restore = False)
        rl_trainer.train(config['rl']['total_timesteps'])

    elif args.command == 'test':
        rl_trainer = make_rl_trainer(config, rl_env, restore = True)
        rl_trainer.rollout()

    elif args.command == 'check':
        rl_tester = make_reward_shaping_tester(config, rl_env)
        rl_tester.test_reward_shaping()


    #import IPython
    #IPython.embed()