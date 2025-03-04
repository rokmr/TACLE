import json
import argparse
from trainer import train
from evaluator import test

def main():
    args = setup_parser().parse_args() 
    param = load_json(args.config) 
    args = vars(args)  
    args.update(param)  
    if args['test_only']:
        test(args)
    else:
        train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproducing the results of TACLE.')
    parser.add_argument('--config', type=str, default='./configs/cifar10/tacle_cifar10-imagenet_pretrained_0.8%.json',
                        help='get the config file from the configs folder.')
    parser.add_argument('--test_only', action='store_true')
    return parser

if __name__ == '__main__':
    main()