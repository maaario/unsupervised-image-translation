import os

from scipy.misc import imsave

from experiment import prepare_argument_parser, set_up_experiment


if __name__ == "__main__":
    argparser = prepare_argument_parser()
    args, _ = argparser.parse_known_args()
    
    patches, em = set_up_experiment(args)

    imsave(os.path.join(args.output, "0_initial.png"), em.MAP_image())
    print("Initial log a posterior:", em.log_a_posterior_probability())
    
    for i in range(1, args.em_iterations + 1):
        print("Executing EM iteration", i)
        em.execute_iteration()
        imsave(os.path.join(args.output, "{}.png".format(i)), em.MAP_image())
        print("Log a posterior", em.log_a_posterior_probability())
