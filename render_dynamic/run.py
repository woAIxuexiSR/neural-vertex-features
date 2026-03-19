import argparse
import json
import os

def run_script(config_path, card):
    
    command = "CUDA_VISIBLE_DEVICES={} python dynamic_render/new_train.py -c {}".format(card, config_path)
    os.system(command)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Neural Radiosity")
    parser.add_argument("-scene", type=str, default="")
    parser.add_argument("-card", type=int, default=0)
    
    args = parser.parse_args()

    path = "zconfigs/{}-dynamic.json".format(args.scene)
                              
    run_script(path, args.card)