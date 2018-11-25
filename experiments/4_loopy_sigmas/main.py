import os
import sys

import numpy as np
from scipy.misc import imsave

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, "src", "unsupervised_image_translation"))
from experiment import prepare_argument_parser, set_up_experiment
from patches import plot_patch_vectors

def run_experiment(args):
    patches, em = set_up_experiment(args)
    
    for i in range(1, args.em_iterations + 1):
        print("Executing EM iteration", i)
        em.execute_iteration()
    imsave(os.path.join(args.output, str(args.lbp_two_sigma2) + ".png"), em.MAP_image())

if __name__ == "__main__":
    argparser = prepare_argument_parser()
    args, _ = argparser.parse_known_args()

    for two_sigma2 in [1e-4, 0.001, 0.01, 0.1, 1, 10, 100]: 
        args.lbp_two_sigma2 = two_sigma2
        run_experiment(args)

    