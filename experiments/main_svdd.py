'''
To train svd:
    python -m experiments.main_svdd --command train --config ./experiments/configure/ant/exp0/exp0_0.yaml

To evaluate svd:
    python -m experiments.main_svdd --command test --config ./experiments/configure/ant/exp0/exp0_0.yaml
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

    tf_setting.set_device()
    if args.command == 'train':
        svdd = make_svdd(config, initialize = True)
        svdd.pretrain(nb_pretraining = config['svdd']['nb_pretraining'])
        svdd.train(nb_training = config['svdd']['nb_training'])
        
    elif args.command =='test':
        svdd = make_svdd(config, initialize = False)
        svdd.restore()
        svdd_tester = make_svdd_tester(config, svdd)

        if config['exp']['data'] == 'ant':
            svdd_tester.test_ood_ant()

    