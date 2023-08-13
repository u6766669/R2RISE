import argparse
from random import choice

import numpy as np
import torch
import utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch import FloatTensor
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding,Conv2d,ReLU,Flatten, LeakyReLU, CrossEntropyLoss
from torch.distributions import Categorical, MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from matplotlib import pyplot as plt
from tqdm import tqdm

from BC import PPO2Agent
from utils import load_mask_from_im

from utils import mask_score,get_flat_grads,get_flat_params,set_params,conjugate_gradient,rescale_and_linesearch


if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

class Cnn(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)

    def forward(self,x):
        x= utils.float_tensor(x)

        conv1_output = F.relu(self.conv1(x))
        conv2_output = F.relu(self.conv2(conv1_output))
        conv3_output = F.relu(self.conv3(conv2_output))
        output = conv3_output.contiguous().view(-1)
        return output

class BCO(Module):
    def __init__(self, args, env, policy='mlp'):
        super(BCO, self).__init__()

        self.policy = policy
        self.act_n = env.action_space.n
        self.args = args

        if self.policy == 'mlp':
            self.obs_n = env.observation_space.shape[0]
            self.pol = Sequential(*[Linear(self.obs_n, 32), LeakyReLU(),
                                       Linear(32, 32), LeakyReLU(),
                                       Linear(32, self.act_n)])
            self.inv = Sequential(*[Linear(self.obs_n * 2, 32), LeakyReLU(),
                                       Linear(32, 32), LeakyReLU(),
                                       Linear(32, self.act_n)])

        elif self.policy == 'cnn':

            if self.args.cnn is not None:
                self.cnn = Cnn()
                state_dict_new = self.cnn.state_dict().copy()
                new_dict_keys = list(state_dict_new.keys())
                trained_state_dict = torch.load(args.cnn)

                for i in range(len(new_dict_keys)):
                    state_dict_new[new_dict_keys[i]]=trained_state_dict[new_dict_keys[i]]
                self.cnn.load_state_dict(state_dict_new)
                print("Successfully initialized Cnn")
            else:
                self.cnn = None

            self.obs_n = 64*7*7 #output of CNN

            self.pol = Sequential(*[Linear(self.obs_n, 32), LeakyReLU(),
                                    Linear(32, 32), LeakyReLU(),
                                    Linear(32, self.act_n)])
            self.inv = Sequential(*[Linear(self.obs_n * 2, 32), LeakyReLU(),
                                    Linear(32, 32), LeakyReLU(),
                                    Linear(32, self.act_n)])

    def pred_act(self, obs):


        if self.policy == 'cnn':
            obs = self.cnn(obs).detach()

        out = self.pol(obs)
        return out

    def pred_inv(self, obs1, obs2):

        if self.policy == 'cnn':
            obsrv1 = []
            obsrv2 = []
            assert obs1.shape[0]==obs2.shape[0]
            for i in range(obs1.shape[0]):
                ob1 = self.cnn(obs1[i])
                ob2 = self.cnn(obs2[i])
                obsrv1.append(ob1)
                obsrv2.append(ob2)
            obs1 = torch.stack(obsrv1)
            obs2 = torch.stack(obsrv2)


        obs = torch.cat([obs1, obs2], dim=-1)
        out = self.inv(obs)

        return out





class DS_Inv(Dataset):
    def __init__(self, obs, acts, after_obs=None):
        self.dat = []

        assert len(after_obs)==len(obs)==len(acts)

        for i in range(len(obs)):
            ob= obs[i]
            af_ob = after_obs[i]
            act = acts[i]
            self.dat.append([ob, af_ob, act])

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        obs, new_obs, act = self.dat[idx]

        return obs, new_obs, np.asarray(act)


class DS_Policy(Dataset):
    def __init__(self, traj):
        self.dat = []

        for dat in traj:
            obs, act = dat

            self.dat.append([obs, act])

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        obs, act = self.dat[idx]

        return obs, np.asarray(act)

def train_policy(args, env, obs, af_obs, acts, model):
    batch_size = 100

    ld_demo = DataLoader(DS_Inv(obs,acts,af_obs), batch_size=batch_size, generator=torch.Generator(device='cuda'))

    loss_func = CrossEntropyLoss().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    EPOCHS = args.num_iters
    M = 1000

    EPS = 0.9
    DECAY = 0.5

    observations = []
    af_observations = []
    actions = []
    for e in tqdm(range(EPOCHS)):
        # step1, generate inverse samples
        epn = 0
        rews = 0
        ep_cnt = 0

        while ep_cnt < M:
            cnt = 0
            ep_obs = []
            ep_af_obs = []
            ep_actions = []
            rew = 0

            obs = env.reset()
            obs = mask_score(obs, env_name)
            obs = np.transpose(obs, (0, 3, 1, 2))
            while True:
                out = model.pred_act(torch.from_numpy(obs).float().cuda())

                if np.random.rand() >= max(EPS,0.05):
                    act = np.argmax(out.cpu().detach().numpy())
                else:
                    act = env.action_space.sample()
                new_obs, r, done, _ = env.step(act)
                new_obs = mask_score(new_obs, env_name)
                new_obs = np.transpose(new_obs, (0, 3, 1, 2))
                cnt += 1
                rew += r


                if done == True:
                    epn += 1
                    print("collect trajectory ", epn)
                    rews += rew
                    ep_cnt += cnt
                    observations += ep_obs
                    af_observations += ep_af_obs
                    actions += ep_actions
                    break

                # if cnt >2000:
                #     break

                ep_obs.append(obs)
                ep_af_obs.append(new_obs)
                ep_actions.append(act)
                obs = new_obs

        rews /= epn
        print('Ep %d: reward=%.2f' % (e + 1, rews))

        # step2, update inverse model
        ld_inv = DataLoader(DS_Inv(observations,actions,af_observations), batch_size=32, shuffle=True, generator=torch.Generator(device='cuda'))
        assert len(observations) == len(actions) == len(af_observations)
        print(len(observations))

        with tqdm(ld_inv) as TQ:
            ls_ep = 0
            for obs1, obs2, act in TQ:
                out = model.pred_inv(obs1.float().cuda(), obs2.float().cuda())
                ls_bh = loss_func(out, act)

                optim.zero_grad()
                ls_bh.backward()
                optim.step()

                ls_bh = ls_bh.cpu().detach().numpy()
                TQ.set_postfix(loss_inv='%.3f' % (ls_bh))
                ls_ep += ls_bh

            ls_ep /= len(TQ)
            print('Ep %d: loss_inv=%.3f\n' % (e + 1, ls_ep))

        # step3, predict inverse action for demo samples
        traj_policy = []

        for obs1, obs2, _ in ld_demo:
            out = model.pred_inv(obs1.float().cuda(), obs2.float().cuda())

            obs = obs1
            out = out.cpu().detach().numpy()
            out = np.argmax(out, axis=1)

            for i in range(out.shape[0]):
                traj_policy.append([obs[i], out[i]])

        # step4, update policy via demo samples
        ld_policy = DataLoader(DS_Policy(traj_policy), batch_size=32, shuffle=True, generator=torch.Generator(device='cuda'))

        with tqdm(ld_policy) as TQ:
            ls_ep = 0

            for obs, act in TQ:
                out = torch.stack([model.pred_act(ob.float().cuda()) for ob in obs])
                ls_bh = loss_func(out, act.cuda())

                optim.zero_grad()
                ls_bh.backward()
                optim.step()

                ls_bh = ls_bh.cpu().detach().numpy()
                TQ.set_postfix(loss_policy='%.3f' % (ls_bh))
                ls_ep += ls_bh

            ls_ep /= len(TQ)
            print('Ep %d: loss_policy=%.3f\n' % (e + 1, ls_ep))

        # step5, save model
        torch.save(model.state_dict(), 'output_npz/BCO/model_%s_%d.pt' % (args.env_name, e + 1))

        EPS *= DECAY
    return model



def generate_expert_traj(env, env_name, agent,min_length):
    action_set = set()

    observations = []
    actions = []
    step_rewards = []
    af_observations = []
    episode_returns = np.zeros((episode_count,))



    for i in range(episode_count):
        done = False
        obsvtion = []
        af_obsvtion = []
        action_seq = []
        gt_rewards = []
        ep_starts = []
        r = 0

        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True:
            action = agent.act(ob, r, done)
            ob_processed = mask_score(ob, env_name)
            ob_processed = np.transpose(ob_processed, (0, 3, 1, 2))
            #ob_processed = np.squeeze(ob_processed)  # get rid of first dimension ob.shape = (1,84,84,4)
            ob, r, done, _ = env.step(action)
            af_ob_processed = mask_score(ob, env_name)
            af_ob_processed = np.transpose(af_ob_processed, (0, 3, 1, 2))



            # to distinguish the masked action and action 0
            action[0] = action[0] + 1
            # env.render()
    

            action_set.add(action[0])
            obsvtion.append(ob_processed)
            af_obsvtion.append(af_ob_processed)
            action_seq.append(action[0])
            gt_rewards.append(r[0])

            done = np.array([done[0]])
            ep_starts.append(done)

            steps += 1
            acc_reward += r[0]
            if len(obsvtion) >= min_length:
                break
            if done:
                if len(obsvtion) < min_length:
                    ob = env.reset()
                    r = 0
                    done = False
                    obsvtion = []
                    af_obsvtion = []
                    action_seq = []
                    gt_rewards = []
                    ep_starts = []
                    steps = 0
                    acc_reward = 0
                else:
                    break

        print("traj length", len(obsvtion), "demo length", len(observations))
        print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps, acc_reward))

        observations.append(obsvtion)
        af_observations.append(af_obsvtion)
        actions.append(action_seq)
        step_rewards.append(gt_rewards)
        episode_returns[i]=acc_reward

    return observations, af_observations, actions, step_rewards, episode_returns, action_set

def eval(env_name, agent):
    seed = 0
    epsilon_greedy = 0.1
    episode_count = 20

    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"
    env_type = "atari"
    # env id, env type, num envs, and seed
    env = make_vec_env(env_id, env_type, 1, seed, wrapper_kwargs={'clip_rewards': False, 'episode_life': False, })
    if env_type == 'atari':
        env = VecFrameStack(env, 4)
    reward = 0
    done = False
    rewards = []
    # writer = open(self.checkpoint_dir + "/" +self.env_name + "_bc_results.txt", 'w')
    for i in range(int(episode_count)):
        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True:
            # preprocess the state
            state = mask_score(ob, env_name)
            state = np.transpose(state, (0, 3, 1, 2))
            if np.random.rand() < epsilon_greedy:
                # print('eps greedy action')
                action = env.action_space.sample()
            else:
                # print('policy action')
                action = agent.pred_act(state)
                action= np.argmax(action.cpu().detach().numpy())
                if action == -1:
                    action = env.action_space.sample()
            ob, reward, done, _ = env.step(action)
            steps += 1
            acc_reward += reward
            if done:
                print("Episode: {}, Steps: {}, Reward: {}".format(i, steps, acc_reward))
                # writer.write("{}\n".format(acc_reward[0]))
                rewards.append(acc_reward)
                break

    print("Mean reward is: " + str(np.mean(rewards)))
    print("Std reward is: " + str(np.std(rewards)))
    return np.mean(rewards),np.std(rewards)


def train_mask_demo(args,env,obs, af_obs, acts,step_r,actset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # test masking
    obs = np.array(obs)
    obs = obs.astype('object')
    # for the second obs in BCO inverse model
    af_obs = np.array(af_obs)
    af_obs = af_obs.astype('object')

    acts = np.array(acts)
    step_r = np.array(step_r)

    min_length = args.min_length
    episode_count = args.episode_count
    checkpoint = args.checkpoint
    input_size = (episode_count, min_length)

    dl = args.degradation_level

    num_iter = args.num_mask
    if args.eval:
        num_iter = 1


    test_mask = generate_masks_ts(1, input_size, dl)



    print("Input size = ", obs.shape,", ",af_obs.shape,", ",acts.shape)
    print("Mask size = ", test_mask[0, :, :, :].shape)

    importance_map = np.zeros_like(test_mask[0, :, :, :]).squeeze()
    del test_mask

    rwd_list =[]
    std_list = []

    for mask_idx in range(num_iter):
        mask = generate_masks_ts(1, input_size, dl).squeeze()
        if args.eval:
            mask = load_mask_from_im(args.env_name,'GAIL',args.seed,0.5)
            mask = mask.squeeze()

        masked_a = np.squeeze(mask*acts)
        masked_r = np.squeeze(mask*step_r)

        while len(mask.shape) < len(obs.shape):
            mask = np.expand_dims(mask, axis=-1)

        masked_o = mask * obs

        masked_af_o = mask * af_obs


        print("Masked shapes = ", masked_o.shape," , ",masked_a.shape," , ",masked_r.shape)

        action_cnt_dict = {}
        masked_r_iter = []
        iter_o = []
        iter_af_o = []
        iter_a = []
        for i in range(masked_o.shape[0]):
            masked_r_iter.append(np.sum(masked_r[i]))
            out_o = []
            out_af_o = []
            out_a = []
            for j in range(masked_o.shape[1]):
                action = masked_a[i][j]
                observation = masked_o[i][j]
                af_observation = masked_af_o[i][j]

                # to distinguish the masked action and action 0
                action = action - 1
                if action == -1:
                    # action = env.action_space.sample()
                    # Skip the masked ob and act
                    continue

                if action in action_cnt_dict:
                    action_cnt_dict[action] += 1
                else:
                    action_cnt_dict[action] = 0

                #masked_a[i][j] = action
                out_a.append(action)
                out_o.append(observation)
                out_af_o.append(af_observation)
            iter_a.append(out_a)
            iter_o.append(out_o)
            iter_af_o.append(out_af_o)

        masked_a = np.concatenate(iter_a)
        masked_o = np.concatenate(iter_o).astype("float64")
        # for the second obs in BCO inverse model
        masked_af_o = np.concatenate(iter_af_o).astype("float64")

        print("Filtered masked shapes = ", masked_o.shape, " , ", masked_a.shape)
        exp_data = (masked_o,masked_a,masked_r_iter)

        ob_dim = 1
        for i in env.observation_space.shape:
            ob_dim *=i

        if args.discrete:
            act_dim=env.action_space.n
        else:
            act_dim= env.action_space.shape[0]

        assert len(masked_a) == len(masked_o)

        env.reset()
        POLICY = 'cnn'
        model = BCO(args,env,policy=POLICY).to(device)
        print("Successfully initialize BCO model......")

        # could change to other algorithm, output the trained model.
        trained_model = train_policy(args, env, masked_o, masked_af_o, masked_a, model)

        avg_rwd, std_rwd = eval(args.env_name, trained_model)
        rwd_list.append(avg_rwd)
        std_list.append(std_rwd)

        newimp = mask * avg_rwd
        importance_map = importance_map + newimp.squeeze()
        if not args.eval:
            #project value range to 0-255
            ratio = 255 / (importance_map.max() - importance_map.min())
            output_img = ratio * (importance_map - importance_map.min())
            output_img = output_img.astype('uint8')
            img = Image.fromarray(output_img)
            plt.title('Importance map for mask_{}'.format(mask_idx))
            plt.axis('off')
            plt.imshow(img,cmap='gray',vmin=0,vmax=255)
            plt.savefig("./image/BCO/" + env_name + "/0/seed_{}_figure_{}.png".format(args.seed,mask_idx), dpi=500)
            plt.close()
        del masked_o
        del masked_a
        del masked_r_iter
        del exp_data
        del mask

    env.close()
    print("Reward List as below, the mean rwd is ", np.mean(rwd_list))
    print(rwd_list)
    print("Reward List as below, the std rwd is ", np.mean(std_list))
    print(std_list)

    if not args.eval:
        np.savez('./output_npz/' + env_name + "/test_BCO_importance_map_{}".format(args.seed), importance_map)

        compressed_map = importance_map.sum(axis=0)

        # compressed_map = compressed_map.reshape(100, 10)[:, 0]

        # num_grid * step = traj length (1000)
        compressed_map = compressed_map.reshape(100, 10).sum(axis=1)  # depend on the number of block you want

        compressed_map = compressed_map - compressed_map.min()
        print(compressed_map)
        np.savez('./output_npz/' + env_name + "/test_BCO_compressed_map_{}".format(args.seed), compressed_map)

        ratio = 255 / (importance_map.max() - importance_map.min())

        output_img = ratio * (importance_map - importance_map.min())
        img = Image.fromarray(np.uint8(output_img))

        plt.title('Explanation for BCO')
        plt.axis('off')
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)
        plt.savefig("./image/BCO/" + env_name + "/test_{}_seed_{}_explanation_map.png".format(args.env_name, args.seed), dpi=600)
        plt.close()

def generate_masks_ts(N, model_input_size, p1):

    # num_grid * step = traj length (1000)
    num_grid,step_size = 100,10

    grid = np.random.rand(N, model_input_size[0],num_grid) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *model_input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Linear upsampling and cropping
        mask_resize = np.repeat(grid[i],step_size,axis=1)
        masks[i, :, :] = mask_resize

    masks = masks.reshape(-1, *model_input_size, 1)
    return masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run')
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default=".", help="top directory where checkpoint models for demos are stored")

    parser.add_argument("--min_length", type=int, default=1000,
                        help="minimum length of trajectories")
    parser.add_argument("--episode_count", type=int, default=20,
                        help="number of input trajectories")
    parser.add_argument("--checkpoint", help="expert level", default=1400, type=int)
    parser.add_argument("--num_mask", help="number of masks generated", default=100, type=int)
    parser.add_argument("--degradation_level", type=float, default=0.5,
                        help="degradation level, [0.1, 0.3, 0.5, 0.7, 0.9]")
    parser.add_argument("--checkpoint_path", default='./checkpoints/test/', help="path to checkpoint to run agent for demos")
    parser.add_argument("--discrete", default=True, type=bool, help='the environment type')
    parser.add_argument('--cnn', default='./output_npz/CnnModel', help='Add Cnn if it is image based model')

    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--num_round_per_iter", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gae_gamma", type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--max_kl", type=float, default=0.01)
    parser.add_argument("--cg_damping", type=float, default=0.1)
    parser.add_argument("--normalize_advantage", type=bool, default=True)
    parser.add_argument("--eval", action='store_true',default=False)


    args = parser.parse_args()
    min_length = args.min_length
    episode_count = args.episode_count
    checkpoint = args.checkpoint
    input_size = (episode_count, min_length)


    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"
    # set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    extra_checkpoint_info = "test"  # for finding checkpoint again

    hist_length = 4

    print(env_id)
    # env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })

    stochastic = True
    env = VecFrameStack(env, 4)
    demonstrator = PPO2Agent(env, env_type, stochastic)
    checkpoint_path = args.models_dir + "/models/" + env_name + "_25/0" + str(checkpoint)
    demonstrator.load(checkpoint_path)

    obs, af_obs, acts, step_r, episode_returns, actset = generate_expert_traj(env, env_name, demonstrator,args.min_length)
    train_mask_demo(args,env,obs, af_obs, acts, step_r, actset)











