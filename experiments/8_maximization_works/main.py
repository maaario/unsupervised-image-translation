import os
import sys

from scipy.misc import imsave

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, "src", "unsupervised_image_translation"))
from experiment import prepare_argument_parser, set_up_experiment


if __name__ == "__main__":
    argparser = prepare_argument_parser()
    args, _ = argparser.parse_known_args()
    
    patches, em = set_up_experiment(args)

    for i in range(1, args.em_iterations + 1):
        print("Executing EM iteration", i)
        
        print("staring loopy")
        em.loopy()
        
        print("computing expectation posteriors")
        em.compute_posteriors()
        
        print("Before maximization:", em.compute_maximized_term())
        print("computing maximization posteriors")
        em.maximization()
        print("After maximization:", em.compute_maximized_term())        
        em.compute_posteriors()
        
        imsave(os.path.join(args.output, "{}.png".format(i)), em.MAP_image())
