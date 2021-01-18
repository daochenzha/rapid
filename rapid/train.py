from baselines import bench, logger
from baselines.a2c.policies import CnnPolicy
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from rapid.agent import learn
from rapid.utils import MlpPolicy, make_env
from rapid.buffer import RankingBuffer

def train(args):

    # exploration score type
    if 'MiniGrid' in args.env:
        args.score_type = 'discrete'
        args.train_rl = True
        policy_fn = MlpPolicy
    elif args.env == 'MiniWorld-MazeS5-v0':
        args.score_type = 'continious'
        args.train_rl = True
        policy_fn = CnnPolicy
    else: # MuJoCo
        args.score_type = 'continious'
        if args.disable_rapid:
            args.train_rl = True
        else:
            args.train_rl = False
        policy_fn = MlpPolicy
    
    # Make the environment
    def _make_env():
        env = make_env(args.env)
        env.seed(args.seed)
        return env
    env = DummyVecEnv([_make_env])
    if not 'MiniGrid' in args.env and not args.env == 'MiniWorld-MazeS5-v0': # Mujoco
        env = VecNormalize(env)

    # Initialize the buffer
    ranking_buffer = RankingBuffer(ob_space=env.observation_space,
                                   ac_space=env.action_space,
                                   args=args)

    # Start training
    learn(policy_fn, env, ranking_buffer, args)
    env.close()
