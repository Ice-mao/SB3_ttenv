import ttenv
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', type=str, default='TargetTracking-v2')
parser.add_argument('--render', help='whether to render', type=int, default=1)
parser.add_argument('--record', help='whether to record', type=int, default=0)
parser.add_argument('--ros', help='whether to use ROS', type=int, default=0)
parser.add_argument('--nb_targets', help='the number of targets', type=int, default=1)
parser.add_argument('--log_dir', help='a path to a directory to log your data', type=str, default='.')
# parser.add_argument('--map', type=str, default="obstacles02")
parser.add_argument('--map', type=str, default="dynamic_map")
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--im_size', type=int, default=28)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

def main():
    env = ttenv.make(args.env,
                    render=args.render,
                    record=args.record,
                    ros=args.ros,
                    map_name=args.map,
                    directory=args.log_dir,
                    num_targets=args.nb_targets,
                    is_training=False,
                    im_size=args.im_size,
                    )

    np.random.seed(args.seed)
    for _ in range(args.repeat):
        nlogdetcov = []
        obs, done = env.reset(), False
        while not done:
            if args.render:
                env.render()
            obs, rew, done, _, info = env.step(env.action_space.sample())
            nlogdetcov.append(info['mean_nlogdetcov'])

        print("Sum of negative logdet of the target belief covariances : %.2f"%np.sum(nlogdetcov))


def train():
    env = ttenv.make(args.env,
                    render=args.render,
                    record=args.record,
                    ros=args.ros,
                    map_name=args.map,
                    directory=args.log_dir,
                    num_targets=args.nb_targets,
                    is_training=False,
                    im_size=args.im_size,
                    )
    np.random.seed(args.seed)


if __name__ == "__main__":
    main()
