import os
import sys

import numpy as np
from scipy.misc import imsave

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, "src", "unsupervised_image_translation"))
from experiment import prepare_argument_parser, set_up_experiment
from patches import plot_patch_vectors


if __name__ == "__main__":
    argparser = prepare_argument_parser()
    args, _ = argparser.parse_known_args()

    patches, em = set_up_experiment(args)
    
    args.output = args.output + "2"
    patches2, em2 = set_up_experiment(args)

    for i in range(1, args.em_iterations + 1):
        print("Executing EM iteration", i)
        em.execute_iteration()
        em2.execute_iteration()
        assert(np.array_equal(em.probs, em2.probs))
    