import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines.logger import Logger
from collections import deque
from baselines.common import explained_variance
from retro.retro_env import RetroEnv
from MyFunctions.sonic_util_local import SonicDiscretizer
from MyFunctions.sonic_util_local import RewardScaler
from MyFunctions.sonic_util_local import make_env_joint
from baselines.common.atari_wrappers import WarpFrame, FrameStack
from gym_remote.client import RemoteEnv as grc


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            #print(loaded_params)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        #def load_pickle(load_path):
        #    temp_model = pickle.load(open(load_path, 'rb'))
        #    train = temp_model.

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = 1
        print('nenv=',nenv)
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())#currently a list
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            #print('infos = ', infos)
            # self.env.render()
            #for info in infos:
            #    maybeepinfo = info.get('episode')
            #    if maybeepinfo:
            #        epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).reshape((self.nsteps, 84, 84, 4))
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions).reshape((self.nsteps,))
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape(self.nsteps)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32).reshape((self.nsteps,))
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        print('mb_actions.max = ', max(mb_actions))
        print('mb_actions.min = ', min(mb_actions))
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return mb_obs, mb_rewards, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, epinfos
        #return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
        #    mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def env_initializer(games, game_states, model, nsteps, gamma, lam, nminibatches, update, numenv):
    obs_joint = []
    rewards_joint = []
    returns_joint = []
    masks_joint = []
    actions_joint = []
    values_joint = []
    neglogpacs_joint = []
    states_joint = []
    epinfos_joint = []
    epinfobuf = deque(maxlen=100)
    #for x in range(0, len(games)):
    for x in range(0, numenv):
        xx = ((update-1)*numenv+x) % len(games)
        game = games[xx]
        game_state = game_states[xx]
        # Create env
        nenvs = 1
        env = make_env_joint(game=game, state=game_state)
        #env = RetroEnv(game=game, state=game_state)
        #env = SonicDiscretizer(env)
        #env = RewardScaler(env)
        #env = WarpFrame(env)
        #env = FrameStack(env, 4)
        # INIT Runner[i]
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
        obs, rewards, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()  # pylint: disable=E0632
        #RESHAPE INTO CORRECT SHAPES
        np.reshape(obs, (nsteps, 84, 84, 4))
        np.reshape(rewards, (nsteps,))
        np.reshape(returns, (nsteps,))
        np.reshape(masks, (nsteps,))
        np.reshape(actions, (nsteps,))
        np.reshape(values, (nsteps,))
        np.reshape(neglogpacs, (nsteps,))
        if len(obs_joint) == 0:
            obs_joint = obs
            #returns_joint = returns
            #masks_joint = masks
            #actions_joint = actions
            #values_joint = values
            #neglogpacs_joint = neglogpacs
        else:
            obs_joint = np.concatenate((obs_joint, obs), 0)
            #returns_joint = np.concatenate((returns_joint, returns), 0)
            #masks_joint = np.concatenate((masks_joint, masks), 0)
            #actions_joint = np.concatenate((actions_joint, actions), 0)
            #values_joint = np.concatenate((values_joint, values), 0)
            #neglogpacs_joint = np.concatenate((neglogpacs_joint, neglogpacs), 0)
        env.close()


        #obs_joint = np.append(obs_joint, obs)
        rewards_joint = np.append(rewards_joint, rewards)
        returns_joint = np.append(returns_joint, returns)
        masks_joint = np.append(masks_joint, masks)
        actions_joint = np.append(actions_joint, actions)
        values_joint = np.append(values_joint, values)
        neglogpacs_joint = np.append(neglogpacs_joint, neglogpacs)
        states_joint = np.append(states_joint, states)
        epinfos_joint = np.append(epinfos_joint, epinfos)
        epinfobuf.extend(epinfos)
    return obs_joint, rewards_joint, returns_joint, masks_joint, actions_joint, values_joint, neglogpacs_joint, states_joint, epinfos_joint, epinfobuf

def learn(*, policy, games, game_states, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma, lam,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange,
            save_interval=0, load_model, load_model_path, save_model, save_model_path):

    #NON-ENV STUFFS
    logger = Logger(dir=save_model_path,
                    output_formats=['stdout'])
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    #ENV INIT FOR OB SPACE
    env = RetroEnv(game=games[0], state=game_states[0])
    env = SonicDiscretizer(env)
    env = RewardScaler(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    nenvs = 1
    ob_space = env.observation_space
    print('ob_space=',ob_space)
    ac_space = env.action_space
    print('ac_space=',ac_space)
    env.close()
    numenv=1
    nbatch = nenvs * nsteps *numenv#* len(games)
    nbatch_train = nbatch // nminibatches

    #MODEL
    make_model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)
    if load_model:
        make_model.load(load_model_path)
    model = make_model


    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()
    nupdates = 2500

    save_model_path = save_model_path + "/model.pkl"

    #START UPDATE
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        tstart = time.time()
        #RUN JOINT EPISODE
        numenv=1
        obs_joint, rewards_joint, returns_joint, masks_joint, actions_joint, values_joint, neglogpacs_joint,\
        states_joint, epinfos_joint, epinfobuf = env_initializer(games=games, game_states=game_states,
        model=model, nsteps=nsteps, gamma=gamma, lam=lam, nminibatches=nminibatches, update=update, numenv=numenv)

        print('returns_joint.mean() = ', returns_joint.mean())
        print('rewards_joint.sum() = ', rewards_joint.sum())
        print('epinfobuf = ', epinfobuf)

        #UPDATE LR AND FLIPRANGE
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        #TRAINING
        #epinfobuf.extend(epinfos)
        print('Loss Value / Model Training Time...')
        mblossvals = []
        if True:# states_joint is None: # nonrecurrent version
            inds = np.arange(nbatch)
            print('len(inds)=nbatch=', len(inds))
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs_joint, returns_joint, masks_joint, actions_joint,
                                                      values_joint, neglogpacs_joint))
                    #slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    #print('lrnow = ', lrnow)
                    #print('len(mbinds) = ', len(mbinds))
                    #print('len(slices) = ', len(slices))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        #else: # recurrent version
        #    #assert nenvs % nminibatches == 0
        #    envsperbatch = nenvs // nminibatches
        #    envinds = np.arange(nenvs)
        #    flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        #    envsperbatch = nbatch_train // nsteps
        #    for _ in range(noptepochs):
        #        np.random.shuffle(envinds)
        #        for start in range(0, nenvs, envsperbatch):
        #            end = start + envsperbatch
        #            mbenvinds = envinds[start:end]
        #            mbflatinds = flatinds[mbenvinds].ravel()
        #            slices = (arr[mbflatinds] for arr in (obs_joint, returns_joint, masks_joint, actions_joint,
        #                                                  values_joint, neglogpacs_joint))
        #            mbstates = states[mbenvinds]
        #            mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))
        #print('mblossvals=', mblossvals)

        lossvals = np.mean(mblossvals, axis=0)
        print('lossvals=', lossvals)
        print('model.loss_names=', model.loss_names)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values_joint, returns_joint)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0):
            model.save(save_model_path)

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving model to', save_model_path)
            model.save(savepath)

    #END UPDATE
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
