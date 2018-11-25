import os
import sys

from scipy.misc import imsave

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, "src", "unsupervised_image_translation"))
from experiment import prepare_argument_parser, set_up_experiment
from patches import plot_patch_vectors


if __name__ == "__main__":
    argparser = prepare_argument_parser()
    args, _ = argparser.parse_known_args()

    patches, em = set_up_experiment(args)

    imsave(os.path.join(args.output, "0_initial.png"), em.MAP_image())
    
    for i in range(15):
        em.lbp_params["iterations"] = i
        em.loopy()
        imsave(os.path.join(args.output, "{}_iterations_of_loopy.png".format(i)), em.MAP_loopy_image())
        
         