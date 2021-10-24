'''
To train ccl:
    python -m experiments.main_ccl --command train --config ./experiments/configure/ant/exp0/exp0_0.yaml

To evaluate ccl:
    python -m experiments.main_ccl --command test --config ./experiments/configure/ant/exp0/exp0_0.yaml
'''

import argparse

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

    if args.command == 'train':
        initialize = True
    elif args.command == 'test':
        initialize = False

    tf_setting.set_device()
    distance_model      = make_distance_model(config)
    data_generator      = make_contrastive_data_generator(config, distance_model)
    feature_model       = make_feature_model(config)
    similarity_model    = make_similarity_model(config)
    classifier_model    = make_classifier_model(config)    
    ccl_model           = make_ccl_model(config, feature_model, similarity_model, 
                                                classifier_model, initialize = initialize)
    ccl_labeler         = make_ccl_labeler(config, ccl_model, initialize = initialize)
        
    if args.command == 'train':
        ccl_trainer = make_ccl_trainer(config, ccl_model, ccl_labeler, 
                                        data_generator, initialize = initialize)
        ccl_trainer.fit()
    
    elif args.command == 'test':
        ccl_model.restore()
        ccl_tester = make_ccl_tester(config, ccl_model)

        ccl_tester.plot_adj_mat()
        if config['exp']['data'] == 'ant':
            ccl_tester.plot_ag_on_env()
    
    

    


