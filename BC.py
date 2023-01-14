import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import dataset
import utils
from baselines.ppo2.model import Model
from baselines.common.policies import build_policy
from baselines.common.cmd_util import make_vec_env
from baselines.common.trex_utils import mask_score
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from utils import load_mask_from_im
from random import choice

from PIL import Image
from matplotlib import pyplot as plt

#from rise import generate_masks_ts



# Take a cloned policy and plot the degredation

min_length = 1000
episode_count = 20
checkpoint = 1400
input_size = (episode_count, min_length)

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

class PPO2Agent(object):
    def __init__(self, env, env_type, stochastic):
        ob_space = env.observation_space
        ac_space = env.action_space
        self.stochastic = stochastic

        if env_type == 'atari':
            policy = build_policy(env, 'cnn')
        elif env_type == 'mujoco':
            policy = build_policy(env, 'mlp')

        make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1,
                                   nsteps=1, ent_coef=0., vf_coef=0.,
                                   max_grad_norm=0.)
        self.model = make_model()

    def load(self, path):
        self.model.load(path)

    def act(self, observation, reward, done):
        if self.stochastic:
            a, v, state, neglogp = self.model.step(observation)
        else:
            a = self.model.act_model.act(observation)
        return a


def generate_novice_demos(env, env_name, agent, model_dir, checkpoint_path):
    #min_length = 1000
    #episode_count = 20
    #checkpoint = 1400
    #input_size = (episode_count, min_length)

    observations = []
    trajectories = []
    learning_returns = []
    step_rewards = []

    checkpoint_path = model_dir + "/models/" + env_name + "_25/0" + str(checkpoint)


    agent.load(checkpoint_path)

    action_set_1 = set()

    for i in range(episode_count):
        done = False
        obsvtion = []
        ob_action_seq = []
        gt_rewards = []
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
            action_set_1.add(action[0])
            obsvtion.append(ob_processed)
            ob_action_seq.append([ob_processed, action[0]])
            gt_rewards.append(r[0])
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
                    ob_action_seq = []
                    gt_rewards = []
                    steps = 0
                    acc_reward = 0
                else:
                    break

        print("traj length", len(obsvtion), "demo length", len(observations))
        print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps, acc_reward))

        observations.append(obsvtion)
        trajectories.append(ob_action_seq)
        learning_returns.append(acc_reward)
        step_rewards.append(gt_rewards)
    demonstrations = trajectories
    print("action set length: ",len(action_set_1)," action set: ", action_set_1)
    #test masking



    #return observations, trajectories, learning_returns, step_rewards
    return demonstrations, learning_returns,action_set_1

class Network(nn.Module):
    def __init__(self, num_output_actions, hist_len=4):
        super().__init__()

        # (Height(Width) - kernel)/stride +1
        self.conv1 = nn.Conv2d(hist_len, 32, kernel_size=8, stride=4) #20*20*32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) #9*9*64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) #7*7*64
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.output = nn.Linear(512, num_output_actions)
        '''
                self.conv1 = nn.Conv2d(hist_len, 32, kernel_size=8, stride=4)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
                self.fc1 = nn.Linear(64 * 7 * 7, 512)
                self.output = nn.Linear(512, num_output_actions)'''

    def forward(self, input):
        conv1_output = F.relu(self.conv1(input))
        conv2_output = F.relu(self.conv2(conv1_output))
        conv3_output = F.relu(self.conv3(conv2_output))
        fc1_output = F.relu(self.fc1(conv3_output.contiguous().view(conv3_output.size(0), -1)))
        output = self.output(fc1_output)
        '''
                trans_input = input.permute(0, 3, 1, 2)
                conv1_output = F.relu(self.conv1(trans_input))
                conv2_output = F.relu(self.conv2(conv1_output))
                conv3_output = F.relu(self.conv3(conv2_output))
                conv3_output = conv3_output.contiguous().view(conv3_output.size(0),-1)
                fc1_output = F.relu(self.fc1(conv3_output))
                final_outputs = self.output(fc1_output)'''
        return conv1_output, conv2_output, conv3_output, fc1_output, output

class Imitator:
    def __init__(self, min_action_set,
                 learning_rate,
                 checkpoint_dir,
                 hist_len,
                 l2_penalty):
        self.minimal_action_set = min_action_set
        self.network = Network(len(self.minimal_action_set), hist_len)
        if torch.cuda.is_available():
            print("Initializing Cuda Nets...")
            self.network.cuda()
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=learning_rate, weight_decay=l2_penalty)
        self.checkpoint_directory = checkpoint_dir

    def predict(self, state):
        # predict action probabilities
        outputs = self.network(Variable(utils.float_tensor(state)))
        vals = outputs[len(outputs) - 1].data.cpu().numpy()
        return vals

    def get_action(self, state):

        vals = self.predict(state)
        return np.argmax(vals)

    # potentially optimizable
    def compute_labels(self, sample, minibatch_size):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        labels = Variable(utils.long_tensor(minibatch_size))
        actions_taken = [x.action for x in sample]
        for i in range(len(actions_taken)):
            labels[i] = np.int(actions_taken[i])

        return labels

    def get_loss(self, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)

    def validate(self, dataset, minibatch_size):
        '''run dataset through loss to get validation error'''
        validation_data = dataset.get_dataset()
        v_loss = 0.0
        for i in range(0, len(validation_data) - minibatch_size, minibatch_size):
            sample = validation_data[i:i + minibatch_size]
            with torch.no_grad():
                state = Variable(utils.float_tensor(np.stack([np.squeeze(x.state) for x in sample])))

                # compute the target values for the minibatch
                labels = self.compute_labels(sample, minibatch_size)
                # print(labels.size())
                # print("labels", labels)
                self.optimizer.zero_grad()
                '''
				Forward pass the minibatch through the
				prediction network.
				'''
                activations = self.network(state)
                '''
				Extract the Q-value vectors of the minibatch
				from the final layer's activations. See return values
				of the forward() functions in cnn.py
				'''
                output = activations[len(activations) - 1]
                _, prediction = torch.max(output, 1)
                # print(labels-prediction)
                loss = self.get_loss(output, labels)
                v_loss += loss
        return v_loss

    def train(self, dataset, minibatch_size):
        # sample a minibatch of transitions
        sample = dataset.sample_minibatch(minibatch_size)
        state = Variable(utils.float_tensor(np.stack([np.squeeze(x.state) for x in sample])))
        labels = self.compute_labels(sample, minibatch_size)

        if len(state) != 32:
            print(len(state))
        # show_from_tensor(state,labels)

        self.optimizer.zero_grad()
        activations = self.network(state)
        output = activations[len(activations) - 1]
        _, prediction = torch.max(output, 1)

        loss = self.get_loss(output, labels)

        loss.backward()
        self.optimizer.step()
        return loss

    '''
	Args:
	This function checkpoints the network.
	'''

    def checkpoint_network(self, env_name, extra_info):
        print("Checkpointing Weights")
        utils.save_checkpoint({
            'state_dict': self.network.state_dict()
        }, self.checkpoint_directory, env_name, extra_info)
        print("Checkpointed.")





def train(env_name,
          minimal_action_set,
          learning_rate,
          l2_penalty,
          minibatch_size,
          hist_len,
          checkpoint_dir,
          updates,
          dataset,
          validation_dataset,
          num_eval_episodes,
          epsilon_greedy,
          extra_info):
    # create DQN agent
    agent = Imitator(list(minimal_action_set),
                     learning_rate,
                     checkpoint_dir,
                     hist_len,
                     l2_penalty)

    print("Beginning training...")
    log_frequency = 500
    log_num = log_frequency
    update = 1
    running_loss = 0.
    best_v_loss = np.float('inf')
    count = 0
    while update < int(updates):

        if update > log_num:
            print(str(update) + " updates completed. Loss {}".format(running_loss / log_frequency))
            log_num += log_frequency
            running_loss = 0
            # run validation loss test
            v_loss = agent.validate(validation_dataset, 10)
            print("Validation accuracy = {}".format(v_loss / validation_dataset.size))
            if v_loss > best_v_loss:
                count += 1
                if count > 5:
                    print("validation not improing for {} steps. Stopping to prevent overfitting".format(count))
                    break
            else:
                best_v_loss = v_loss
                print("updating best vloss", best_v_loss)
                count = 0
        l = agent.train(dataset, minibatch_size)
        running_loss += l
        update += 1
    print("Training completed.")
    agent.checkpoint_network(env_name, extra_info)

    return agent

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

    print("env actions", env.action_space)

    # 100 episodes
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
                action = agent.get_action(state)
                if action == -1:
                    #print("Meeting -1 action!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default=".", help="top directory where checkpoint models for demos are stored")
    parser.add_argument("--num_bc_eval_episodes", type=int, default=20,
                        help="number of epsilon greedy BC demos to generate")
    parser.add_argument("--num_masks", type=int, default=700,
                        help="number of masks/retrained models")
    parser.add_argument("--degradation_level", type=float, default=0.5,
                        help="degradation level, [0.1, 0.3, 0.5, 0.7, 0.9]")
    parser.add_argument("--checkpoint_path", default='./checkpoints/test/', help="path to checkpoint to run agent for demos")
    parser.add_argument("--num_demos", help="number of demos to generate", default=20, type=int)
    parser.add_argument("--num_bc_steps", default=15000, type=int, help='number of steps of BC to run')

    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--hist-len", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--l2-penalty", type=float, default=0.00)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/test/")
    parser.add_argument('--epsilon_greedy', default=0.0, type=float,
                        help="fraction of actions chosen at random for rollouts")

    parser.add_argument("--eval", action='store_true',default=False)

    args = parser.parse_args()
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

    ##generate demonstrations for use in BC
    demonstrations, learning_returns, actset = generate_novice_demos(env, env_name, demonstrator, args.models_dir,
                                                             args.checkpoint_path)
    # demonstrations, learning_returns = generate_demos(env, env_name, demonstrator, args.checkpoint_path, args.num_demos)

    # Run BC on demos
    dataset_size = sum([len(d) for d in demonstrations])
    print("Data set size = ", dataset_size)



    #test masking
    num_masks = args.num_masks
    if args.eval:
        num_masks = 10

    dl = args.degradation_level

    dem = np.array(demonstrations)
    dem = dem.astype('object')
    dem = np.expand_dims(dem, axis=0)
    test_mask = generate_masks_ts(1, input_size, dl)
    print("Input size = ", dem.shape)
    print("Mask size = ", test_mask[0, :, :, :].shape)

    importance_map = np.zeros_like(test_mask[0, :, :, :]).squeeze()
    del test_mask

    max_masked_rwd=0
    min_masked_rwd =9999
    rwd_list =[]
    std_list = []
    for j in range(num_masks):
        mask = generate_masks_ts(1, input_size, dl)
        if args.eval:
            mask = load_mask_from_im(args.env_name,'BC',args.seed,0.6)
        masked_dem = mask * dem

        masked_dem = np.squeeze(masked_dem)
        print("Masked size = ", masked_dem.shape)



        episode_index_counter = 0
        num_data = 0
        action_set = set()
        action_cnt_dict = {}
        data = []
        cnt = 0
        test_reward = []

        for i in range(masked_dem.shape[0]):
            #print("adding demonstration", cnt)
            cnt += 1
            episode = masked_dem[i]
            for sa in episode:
                state, action = sa

                # to distinguish the masked action and action 0
                action = action-1

                if action  == -1:
                    #action = env.action_space.sample()
                    continue

                action_set.add(action)
                if action in action_cnt_dict:
                    action_cnt_dict[action] += 1
                else:
                    action_cnt_dict[action] = 0
                # transpose into 4x84x84 format
                state = np.transpose(np.squeeze(state), (2, 0, 1))
                # state = np.squeeze(state)
                data.append((state, action))
        del masked_dem

        # take 10% as validation data
        np.random.shuffle(data)
        training_data_size = int(len(data) * 0.9)
        training_data = data[:training_data_size]
        validation_data = data[training_data_size:]
        print("training size = {}, validation size = {}".format(len(training_data), len(validation_data)))
        training_dataset = dataset.Dataset(training_data_size, hist_length)
        validation_dataset = dataset.Dataset(len(validation_data), hist_length)
        for state, action in training_data:
            training_dataset.add_item(state, action)
            num_data += 1
            if num_data == training_dataset.size:
                break

        for state, action in validation_data:
            validation_dataset.add_item(state, action)
            num_data += 1
            if num_data == validation_dataset.size:
                break
        del training_data, validation_data, data
        print("available actions", action_set)
        print(action_cnt_dict)

        agent = train(args.env_name,
                      action_set,
                      args.learning_rate,
                      args.l2_penalty,
                      args.minibatch_size,
                      args.hist_len,
                      args.checkpoint_dir,
                      args.num_bc_steps,
                      training_dataset,
                      validation_dataset,
                      args.num_bc_eval_episodes,
                      0.01,
                      extra_checkpoint_info)

        avg_rwd, std_rwd = eval(args.env_name, agent)
        newimp = mask * avg_rwd
        importance_map = importance_map + newimp.squeeze()

        rwd_list.append(avg_rwd)
        std_list.append(std_rwd)
        if avg_rwd >max_masked_rwd:
            max_masked_rwd=avg_rwd
        if avg_rwd <min_masked_rwd:
            min_masked_rwd=avg_rwd
        #     torch.save(agent.network.state_dict(),'./output_npz/CnnModel')

        if not args.eval:
            ratio = 255/(importance_map.max()-importance_map.min())
            output_img = ratio*(importance_map-importance_map.min())
            img = Image.fromarray(output_img)
            plt.title('Importance map for mask_{}'.format(j))
            plt.axis('off')
            plt.imshow(img)
            plt.savefig("./image/BC/"+env_name+"/test/test_seed_{}_figure_{}.png".format(args.seed, j),dpi=500)
            plt.close()
        del mask

    print("Size of the importance map ", importance_map.shape)
    print("Minimum reward ", min_masked_rwd, "Maximum reward ", max_masked_rwd)
    print("Reward List as below, the mean rwd is ", np.mean(rwd_list))
    print(rwd_list)
    print("Reward List as below, the std rwd is ", np.mean(std_list))
    print(std_list)

    if not args.eval:
        np.savez('./output_npz/' + env_name + "/test_BC_importance_map_{}".format(args.seed), importance_map)

        compressed_map = importance_map.sum(axis=0)

        # compressed_map = compressed_map.reshape(100,10)[:,0]

        #depend on the number of block you want
        compressed_map = compressed_map.reshape(100,10).sum(axis=1)

        dev_compressed_map = compressed_map - compressed_map.min()
        print(dev_compressed_map)
        nom_com_map = compressed_map/compressed_map.max()
        print(nom_com_map)


        np.savez('./output_npz/'+env_name+"/test_BC_compressed_map_{}".format(args.seed), compressed_map)

        ratio = 255/(importance_map.max()-importance_map.min())
        #importance_map = np.repeat(importance_map,10,axis =0)
        #importance_map = np.repeat(importance_map, 10, axis=1)


        output_img = ratio * (importance_map - importance_map.min())
        img = Image.fromarray(output_img)

        plt.title('Explanation for BC')
        plt.axis('off')
        plt.imshow(img)
        plt.savefig("./image/BC/"+env_name+"/test_seed_{}_explanation_map.png".format(args.seed),dpi=600)
        plt.close()
    '''
    episode_index_counter = 0
    num_data = 0
    action_set = set()
    action_cnt_dict = {}
    data = []
    cnt = 0
    while (demonstrations):
        print("adding demonstration", cnt)
        cnt += 1
        episode = demonstrations[0]
        for sa in episode:
            state, action = sa
            # action = action
            action_set.add(action)
            if action in action_cnt_dict:
                action_cnt_dict[action] += 1
            else:
                action_cnt_dict[action] = 0
            # transpose into 4x84x84 format
            state = np.transpose(np.squeeze(state), (2, 0, 1))
            # state = np.squeeze(state)
            data.append((state, action))
        del demonstrations[0]
    del demonstrations

    # take 10% as validation data
    np.random.shuffle(data)
    training_data_size = int(len(data) * 0.8)
    training_data = data[:training_data_size]
    validation_data = data[training_data_size:]
    print("training size = {}, validation size = {}".format(len(training_data), len(validation_data)))
    training_dataset = dataset.Dataset(training_data_size, hist_length)
    validation_dataset = dataset.Dataset(len(validation_data), hist_length)
    for state, action in training_data:
        training_dataset.add_item(state, action)
        num_data += 1
        if num_data == training_dataset.size:
            print("data set full")
            break

    for state, action in validation_data:
        validation_dataset.add_item(state, action)
        num_data += 1
        if num_data == validation_dataset.size:
            print("data set full")
            break
    del training_data, validation_data, data
    print("available actions", action_set)
    print(action_cnt_dict)

    agent = train(args.env_name,
                  action_set,
                  args.learning_rate,
                  args.l2_penalty,
                  args.minibatch_size,
                  args.hist_len,
                  args.checkpoint_dir,
                  args.num_bc_steps,
                  training_dataset,
                  validation_dataset,
                  args.num_bc_eval_episodes,
                  0.01,
                  extra_checkpoint_info)

    eval(args.env_name,agent)
'''
