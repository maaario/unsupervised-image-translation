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

    imsave(os.path.join(args.output, "0_initial.png"), em.MAP_image())
    print("Initial log a posterior:", em.log_a_posterior_probability())
    
    for i in range(1, args.em_iterations + 1):
        print("Executing EM iteration", i)
        
        print("staring loopy")
        em.loopy()
        imsave(os.path.join(args.output, "{}.1.loopy.png".format(i)), em.MAP_loopy_image())

        print("computing expectation posteriors")
        em.compute_posteriors()
        imsave(os.path.join(args.output, "{}.2.exp.png".format(i)), em.MAP_image())

        print("computing maximization posteriors")
        em.maximization()
        em.compute_posteriors()
        imsave(os.path.join(args.output, "{}.3.max.png".format(i)), em.MAP_image())

        print("Log a posterior:", em.log_a_posterior_probability())
    