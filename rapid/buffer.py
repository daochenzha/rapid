import numpy as np
from gym.spaces import Discrete, Box

class RankingBuffer(object):
    def __init__(self, ob_space, ac_space, args):
        '''
        Args:
            w0: Weight for extrinsic rewards
            w1: Weight for local bonus
            w2: Weight for global bonus (sums of count-based exploration)
        '''
        self.size = args.buffer_size

        self.ob_shape = ob_space.shape
        self.ob_dim = 1
        for dim in self.ob_shape:
            self.ob_dim *= dim

        if isinstance(ac_space, Discrete):
            self.action_type = 'discrete'
        elif isinstance(ac_space, Box):
            self.action_type = 'box'
            self.ac_dim = ac_space.shape[0]
        else:
            ValueError('The action space is not supported.')

        self.data = None
        self.index = 0
        self.counter = Counter()
        self.w0 = args.w0
        self.w1 = args.w1
        self.w2 = args.w2
        self.score_type = args.score_type
        print('Buffer weights:', self.w0, self.w1, self.w2)

    def insert(self, obs, acs, ret):
        if self.w1 > 0:
            local_bonus = get_local_bonus(obs, self.score_type)
        else:
            local_bonus = 0.0

        num = obs.shape[0]
        if self.action_type == 'discrete':
            _ac_data = np.expand_dims(acs, axis=1)
        elif self.action_type == 'box':
            _ac_data = acs
        _data = np.concatenate((
            obs.astype(float).reshape(num,-1),
            _ac_data,
            np.zeros((num,1)),
            np.expand_dims(np.repeat(ret,num), axis=1),
            np.expand_dims(np.repeat(local_bonus,num), axis=1),
            np.zeros((num,1)),
            ), axis=1)
        if self.w2 > 0:
            episode_index = self.counter.add(_data[:, :self.ob_dim])
        else:
            episode_index = 0
        _data[:,-4] = np.repeat(episode_index,num)
        if self.data is None:
            self.data = _data
        else:
            self.data = np.concatenate((self.data, _data), axis=0)
            if self.w2 > 0:
                global_bonus = self.counter.get_bonus(self.data[:,-4].astype(int))
            else:
                global_bonus = 0.0
            self.data[:,-1] = self.w0 * self.data[:,-3] + self.w1 * self.data[:,-2] + self.w2 * global_bonus
            self.data = self.data[self.data[:,-1].argsort()][-self.size:]
        self.index = self.data.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(range(0,self.index), batch_size)
        sampled_data = self.data[idx]
        obs = sampled_data[:,:self.ob_dim]
        obs = obs.reshape((batch_size,) + self.ob_shape)
        if self.action_type == 'discrete':
            acs = sampled_data[:,self.ob_dim].astype(int)
        elif self.action_type == 'box':
            acs = sampled_data[:,self.ob_dim:self.ob_dim+self.ac_dim]
        return obs, acs

def get_local_bonus(obs, score_type):
    if score_type == 'discrete':
        obs = obs.reshape((obs.shape[0], -1))
        unique_obs = np.unique(obs, axis=0)
        total = obs.shape[0]
        unique = unique_obs.shape[0]
        score = float(unique) / total
    elif score_type == 'continious':
        obs = obs.reshape((obs.shape[0], -1))
        obs_mean = np.mean(obs, axis=0)
        score =  np.mean(np.sqrt(np.sum((obs - obs_mean) * (obs -obs_mean), axis=1)))
    else:
        raise ValueError('Score type {} is not defined'.format(score_type))

    return score

class Counter(object):
    def __init__(self):
        self.counts = dict()
        self.episodes = dict()
        self.episode_bonus = dict()
        self.episode_index = -1

    def add(self, obs):
        for ob in obs:
            ob = tuple(ob)
            if ob not in self.counts:
                self.counts[ob] = 1
            else:
                self.counts[ob] += 1
        self.episode_index += 1
        self.episodes[self.episode_index] = obs
        self.update_bonus()
        return self.episode_index

    def update_bonus(self):
        for idx in self.episodes:
            bonus = []
            obs = self.episodes[idx]
            for ob in obs:
                ob = tuple(ob)
                count = self.counts[ob]
                bonus.append(count)
            bonus = 1.0 / np.sqrt(np.array(bonus))
            bonus = np.mean(bonus)
            self.episode_bonus[idx] = bonus

    def get_bonus(self, idxs):
        self.episodes = {k:self.episodes[k] for k in idxs}
        self.episode_bonus = {k:self.episode_bonus[k] for k in idxs}
        #print(self.episode_bonus)
        bonus = []
        for idx in idxs:
            bonus.append(self.episode_bonus[idx])
        return np.array(bonus)

        
