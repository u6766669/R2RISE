import argparse
from random import choice

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import FloatTensor
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding,Conv2d,ReLU,Flatten
from torch.distributions import Categorical, MultivariateNormal
from PIL import Image
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from matplotlib import pyplot as plt
from tqdm import tqdm

from BC import PPO2Agent
from utils import load_mask_from_im

from utils import mask_score,get_flat_grads,get_flat_params,set_params,conjugate_gradient,rescale_and_linesearch

#https://github.com/hcnoh/gail-pytorch/blob/main/models/gail.py

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
        x= FloatTensor(x)
        conv1_output = F.relu(self.conv1(x))
        conv2_output = F.relu(self.conv2(conv1_output))
        conv3_output = F.relu(self.conv3(conv2_output))
        output = conv3_output.contiguous().view(-1)
        return output

class PolicyNetwork(Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()

        self.fc1 = Linear(64 * 7 * 7, 512)
        self.fc2 = Linear(512, 50)
        self.fc3 = Linear(50, 50)
        self.output = Linear(50, action_dim)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = Parameter(torch.zeros(action_dim))

    def forward(self, states):

        fc1_output = F.tanh(self.fc1(states))
        fc2_output= F.tanh(self.fc2(fc1_output))
        fc3_output=F.tanh(self.fc3(fc2_output))
        output = self.output(fc3_output)
        if self.discrete:
            probs = torch.softmax(output, dim=-1)
            distb = Categorical(probs)
        else:
            mean = output

            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim) * (std ** 2)

            distb = MultivariateNormal(mean, cov_mtx)

        return distb


class ValueNetwork(Module):
    def __init__(self, state_dim) -> None:
        super().__init__()


        self.fc1 = Linear(64 * 7 * 7, 512)
        self.fc2 = Linear(512, 50)
        self.fc3 = Linear(50, 50)
        self.output = Linear(50, 1)

    def forward(self, states):

        fc1_output = F.tanh(self.fc1(states))
        fc2_output= F.tanh(self.fc2(fc1_output))
        fc3_output=F.tanh(self.fc3(fc2_output))
        output = self.output(fc3_output)
        return output


class Discriminator(Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if self.discrete:
            self.act_emb = Embedding(
                action_dim, state_dim
            )
            self.net_in_dim = 2 * state_dim
        else:
            self.net_in_dim = state_dim + action_dim


        self.fc1 = Linear(64 * 7 * 7+1, 512)
        self.fc2 = Linear(512, 50)
        self.fc3 = Linear(50, 50)
        self.output = Linear(50, 1)

    def forward(self, states, actions):
        actions =torch.unsqueeze(actions,dim=-1)

        ob_act=torch.cat([states,actions],dim=-1)
        fc1_output = F.tanh(self.fc1(ob_act))
        fc2_output= F.tanh(self.fc2(fc1_output))
        fc3_output=F.tanh(self.fc3(fc2_output))
        output = self.output(fc3_output)
        return torch.sigmoid(output)

    def get_logits(self, states, actions):
        # print("state shape: ",states.shape, "action shape: ", actions.shape)
        # if self.discrete:
        #     actions = self.act_emb(actions.long())
        # print("embedded action shape: ", actions.shape)
        # sa = torch.cat([states, actions], dim=-1)

        return self.forward(states,actions)

class GAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        args=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.args = args

        if self.args.cnn is not None:
            self.cnn = Cnn()
            state_dict_new = self.cnn.state_dict().copy()
            new_dict_keys = list(state_dict_new.keys())
            trained_state_dict = torch.load(args.cnn)
            # trained_dict_keys = list(trained_state_dict.keys())
            # print("new_state_dict size: {}  trained state_dict size: {}".format(len(new_dict_keys),len(trained_dict_keys)) )
            # print("New State Dict: ",new_dict_keys)
            # print("Trained state Dict: ",trained_dict_keys)
            for i in range(len(new_dict_keys)):
                state_dict_new[new_dict_keys[i]]=trained_state_dict[new_dict_keys[i]]
            self.cnn.load_state_dict(state_dict_new)
            print("Successfully initialized Cnn")
        else:
            self.cnn = None


        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)

        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()

        if self.args.cnn is not None:
            state=self.cnn(state).detach()

        state = Variable(FloatTensor(state))
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()
        return action

    def train_policy(self, env, exp_data, render=False):
        exp_obs,exp_acts,exp_rwd_iter = exp_data

        #exp_obs=FloatTensor(exp_obs)
        exp_acts=FloatTensor(exp_acts)

        if self.args.cnn is not None:
            exp_obs=torch.stack([self.cnn(np.expand_dims(ob, axis=0)).detach() for ob in exp_obs])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        exp_obs=exp_obs.to(device)

        num_iters = self.args.num_iters
        num_round_per_iter = self.args.num_round_per_iter
        horizon = self.args.horizon
        lambda_ = self.args.lr
        gae_gamma = self.args.gae_gamma
        gae_lambda = self.args.gae_lambda
        eps = self.args.epsilon
        max_kl = self.args.max_kl
        cg_damping = self.args.cg_damping
        normalize_advantage = self.args.normalize_advantage

        opt_d = torch.optim.Adam(self.d.parameters())


        exp_rwd_mean = np.mean(exp_rwd_iter)
        print(
            "Expert Reward Mean: {}".format(exp_rwd_iter)
        )


        rwd_iter_means = []
        training_process = []
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []
            rnd = 0


            while rnd < num_round_per_iter:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []
                rnd += 1

                t = 0
                done = False

                ob = env.reset()

                while not done:
                    state = mask_score(ob, env_name)
                    state = np.transpose(state, (0, 3, 1, 2))
                    act = self.act(state)


                    if self.args.cnn is not None:
                        state = self.cnn(state).detach()
                    else:
                        state = state.squeeze()
                    ep_obs.append(state)
                    obs.append(state)

                    ep_acts.append(act)
                    acts.append(act)

                    if render:
                        env.render()
                    ob, rwd, done, _ = env.step(act)

                    ep_rwds.append(rwd[0])
                    ep_gms.append(gae_gamma ** t)
                    ep_lmbs.append(gae_lambda ** t)

                    t += 1


                    if rnd >num_round_per_iter:
                        done = True

                    if horizon is not None:
                        if t >= horizon:
                            done = True
                            break

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

                if self.args.cnn is not None:
                    ep_obs = torch.stack(ep_obs)
                else:
                    ep_obs = np.array(ep_obs).astype("float64")
                    ep_obs = FloatTensor(ep_obs)


                ep_acts = FloatTensor(np.array(ep_acts))
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts))\
                    .squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[i:]) for i in range(t)]
                )
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                ep_deltas = ep_costs.unsqueeze(-1)\
                    + gae_gamma * next_vals\
                    - curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            rwd_iter_means.append(np.mean(rwd_iter))
            print(
                "Iterations: {},   Reward Mean: {} from {} episodes."
                .format(i + 1, np.sum(rwd_iter)/len(rwd_iter), len(rwd_iter))
            )
            if i %10 ==0:
                training_process.append(np.sum(rwd_iter)/len(rwd_iter))

            if self.args.cnn is not None:
                obs = torch.stack(obs)
            else:
                obs = np.array(obs).astype("float64")
                obs = FloatTensor(obs)
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad()

            loss = F.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + F.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores)
                )
            loss.backward()
            opt_d.step()

            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                return (advs * torch.exp(
                            distb.log_prob(acts)
                            - old_distb.log_prob(acts).detach()
                        )).mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                            (old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)
                        ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts))\
                .mean()
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)


        if args.eval:

            sava_url = './output_npz/' + env_name + '/bc_training_process_GAIL' + str(seed) + '.xlsx'

            import xlsxwriter
            xl = xlsxwriter.Workbook(sava_url)
            sheet = xl.add_worksheet('sheet1')
            j = 0
            k = 0
            for i in training_process:
                sheet.write(j, k, i)
                k = k + 1

            xl.close()

        return exp_rwd_mean, rwd_iter_means

def generate_expert_traj(env, env_name, agent,min_length):
    action_set = set()

    observations = []
    actions = []
    step_rewards = []
    episode_starts = []
    episode_starts.append(True)
    episode_returns = np.zeros((episode_count,))



    for i in range(episode_count):
        done = False
        obsvtion = []
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
            ob_processed = np.squeeze(ob_processed)  # get rid of first dimension ob.shape = (1,84,84,4)
            ob, r, done, _ = env.step(action)


            # to distinguish the masked action and action 0
            action[0] = action[0] + 1
            # env.render()
            '''
            im = transforms.ToPILImage()(ob_processed).convert("RGB")
            plt.imshow(im)
            plt.title(action)
            plt.pause(1)
            '''

            action_set.add(action[0])
            obsvtion.append(ob_processed)
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
        actions.append(action_seq)
        step_rewards.append(gt_rewards)
        episode_returns[i]=acc_reward
        episode_starts += ep_starts

    # observations=np.concatenate(observations)
    # actions=np.concatenate(actions)
    # observations=FloatTensor(np.array(observations))
    # actions=FloatTensor(np.array(actions))
    return observations, actions, step_rewards, episode_returns, episode_starts, action_set

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
                action = agent.act(state)
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


def train_mask_demo(args,env,obs,acts,step_r,actset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # test masking
    obs = np.array(obs)
    obs = obs.astype('object')
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



    print("Input size = ", obs.shape,", ",acts.shape)
    print("Mask size = ", test_mask[0, :, :, :].shape)

    importance_map = np.zeros_like(test_mask[0, :, :, :]).squeeze()
    del test_mask

    rwd_list =[]
    std_list = []

    for mask_idx in range(num_iter):
        mask = generate_masks_ts(1, input_size, dl).squeeze()
        if args.eval:
            mask = load_mask_from_im(args.env_name,'BC',args.seed,0.5)
            mask = mask.squeeze()

        masked_a = np.squeeze(mask*acts)
        masked_r = np.squeeze(mask*step_r)

        while len(mask.shape) < len(obs.shape):
            mask = np.expand_dims(mask, axis=-1)
        masked_o = np.squeeze(mask * obs)
        masked_o = np.transpose(masked_o,(0,1,4,2,3))

        print("Masked shapes = ", masked_o.shape," , ",masked_a.shape," , ",masked_r.shape)

        action_cnt_dict = {}
        masked_r_iter = []
        iter_o = []
        iter_a = []
        for i in range(masked_o.shape[0]):
            masked_r_iter.append(np.sum(masked_r[i]))
            out_o = []
            out_a = []
            for j in range(masked_o.shape[1]):
                action = masked_a[i][j]
                observation = masked_o[i][j]

                # to distinguish the masked action and action 0
                action = action - 1
                if action == -1:
                    # action = env.action_space.sample()
                    continue

                if action in action_cnt_dict:
                    action_cnt_dict[action] += 1
                else:
                    action_cnt_dict[action] = 0

                #masked_a[i][j] = action
                out_a.append(action)
                out_o.append(observation)
            iter_a.append(out_a)
            iter_o.append(out_o)

        masked_a = np.concatenate(iter_a)
        masked_o = np.concatenate(iter_o).astype("float64")



        #masked_a = np.concatenate(masked_a)
        #masked_o = np.concatenate(masked_o).astype("float64")
        ##masked_r = np.concatenate(masked_r)

        #masked_o=np.array(masked_o).astype("float64")
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

        model = GAIL(ob_dim, act_dim, args.discrete, args).to(device)

        exp_rwd_mean, rwd_iter_means = model.train_policy(env, exp_data)
        print("expert episode reward ",exp_rwd_mean)
        print("training reward ", rwd_iter_means)

        avg_rwd, std_rwd = eval(args.env_name, model)
        rwd_list.append(avg_rwd)
        std_list.append(std_rwd)

        newimp = mask * avg_rwd
        importance_map = importance_map + newimp.squeeze()
        if not args.eval:
            #project value range to 0-255
            ratio = 255 / (importance_map.max() - importance_map.min())
            output_img = ratio * (importance_map - importance_map.min())
            img = Image.fromarray(output_img)
            plt.title('Importance map for mask_{}'.format(mask_idx))
            plt.axis('off')
            plt.imshow(img)
            plt.savefig("./image/GAIL/" + env_name + "/pre_test/seed_{}_figure_{}.png".format(args.seed,mask_idx), dpi=500)
            plt.close()
        del masked_o
        del masked_a
        del masked_r_iter
        del exp_data
        del mask

    print("Reward List as below, the mean rwd is ", np.mean(rwd_list))
    print(rwd_list)
    print("Reward List as below, the std rwd is ", np.mean(std_list))
    print(std_list)

    if not args.eval:
        np.savez('./output_npz/' + env_name + "/GAIL_importance_map_{}".format(args.seed), importance_map)

        compressed_map = importance_map.sum(axis=0)

        # compressed_map = compressed_map.reshape(100, 10)[:, 0]

        compressed_map = compressed_map.reshape(100, 10).sum(axis=1)  # depend on the number of block you want

        compressed_map = compressed_map - compressed_map.min()
        print(compressed_map)
        np.savez('./output_npz/' + env_name + "/GAIL_compressed_map_{}".format(args.seed), compressed_map)

        ratio = 255 / (importance_map.max() - importance_map.min())
        # importance_map = np.repeat(importance_map,10,axis =0)
        # importance_map = np.repeat(importance_map, 10, axis=1)
        output_img = ratio * (importance_map - importance_map.min())
        img = Image.fromarray(output_img)

        plt.title('Explanation for GAIL')
        plt.axis('off')
        plt.imshow(img)
        plt.savefig("./image/GAIL/" + env_name + "/{}_seed_{}_explanation_map.png".format(args.env_name, args.seed), dpi=600)
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

    parser.add_argument("--num_iters", type=int, default=2000)
    parser.add_argument("--num_round_per_iter", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    obs, acts, step_r, episode_returns, episode_starts, actset = generate_expert_traj(env, env_name, demonstrator,args.min_length)
    train_mask_demo(args,env,obs, acts, step_r, actset)











