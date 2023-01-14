import imageio
import numpy as np
import xlsxwriter
import torch
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from BC import PPO2Agent

from PIL import Image
from matplotlib import pyplot as plt



def save_frame(path, frames):
    cnt=0
    for frame in frames:
        imageio.imsave(path+str(cnt)+'.jpg',frame)
        cnt += 1
    print("Successfully extract frames to: ",path)


def generate_vid_from_demos(env_name, seed,model,importance_index):
    min_length = 1000
    episode_count = 20
    checkpoint = 1400

    expected_length = sum([len(d) for d in importance_index])

    env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"
    if env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"

    # env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })

    stochastic = True
    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, "atari", stochastic)

    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_frames = []


    checkpoint_path = "./models/" + env_name + "_25/0" + str(checkpoint)


    agent.load(checkpoint_path)


    for i in range(episode_count):
        done = False
        ob = env.reset()
        steps = 0
        r=0
        acc_reward = 0

        obsvtion = []

        print("Extracting frames from Trajectory ",i)

        while True:
            action = agent.act(ob, r, done)


            if steps in importance_index[i]:
                frame = env.render(
                    mode='rgb_array'
                )
                obsvtion.append(frame) # get rid of first dimension ob.shape = (1,84,84,4)

            ob, r, done, _ = env.step(action)
            steps += 1
            acc_reward += r[0]

            if steps >= min_length:
                break
            if done:
                if steps < min_length:
                    ob = env.reset()
                    r = 0
                    acc_reward = 0
                    done = False
                    obsvtion = []
                    steps = 0
                else:
                    break
        print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps, acc_reward))
        out_frames += obsvtion

    assert len(out_frames)==expected_length

    imageio.mimsave('./output_npz/' + env_name + '/' + model + '_' + str(seed) + '.mp4', out_frames, fps=12)
    save_frame('./output_npz/' + env_name + '/frames/',out_frames)


def compare_im(env_name,seed,im_a,im_b):
    assert im_a.shape == im_b.shape

    ratio_a = 255 / (im_a.max() - im_a.min())
    img_a = ratio_a * (im_a - im_a.min())

    ratio_b = 255 / (im_b.max() - im_b.min())
    img_b = ratio_b * (im_b - im_b.min())

    img_dev = np.abs(img_a-img_b)

    title = 'Difference between importance maps'
    url = "./image/deviation/" + env_name + "/importance_map_deviation_seed_{}.png".format(seed)

    save_array_to_img(img_dev, title, url)


def export_cm_to_xlsx(env_name, seed, model,im):
    sava_url = './output_npz/' + env_name + '/test_' + model + '_' + str(seed) + '.xlsx'

    cm = im.sum(axis=0)
    # obtain each grid importance
    cm =cm.reshape(100, 10).sum(axis=1)

    #summarize to 10 blocks. if 20 then (20,-1)
    cm = cm.reshape(5,-1).sum(axis=1)


    dev_compressed_map = cm - cm.min()
    nom_com_map = dev_compressed_map / (cm.max() - cm.min())
    #print(nom_com_map)

    xl = xlsxwriter.Workbook(sava_url)
    sheet = xl.add_worksheet('sheet1')

    k=0
    j=0
    for i in range(len(nom_com_map)):
        importance = nom_com_map.item(i)
        sheet.write(j, k, importance)
        k = k+1

    xl.close()


def save_array_to_img(img_ary, title, url, need_axis = True):
    img_ary = np.repeat(img_ary, 10, axis=0)  # enlarge the grid to get better observation
    img = Image.fromarray(img_ary)
    plt.title(title)
    if not need_axis:
        plt.axis('off')
    else:
        plt.ylabel("# of demos")
        plt.yticks([])
    plt.imshow(img)
    plt.savefig(url, dpi=500)
    plt.close()

def load_im(env_name,seed,model):
    print(env_name, " + ", model, " + ", str(seed))
    if model == 'GAIL':
        im = np.load('./output_npz/'+env_name+'/'+model+'_importance_map_'+str(seed)+'.npz')
    if model =='BC':
        im = np.load('./output_npz/' + env_name + '/' + model + '_importance_map_' + str(seed) + '.npz')
    im = im['arr_0']
    return im


if __name__ == "__main__":
    env_name = 'beamrider'
    seed = 0

    model_A = 'BC'
    model_B = 'GAIL'

    im_a = load_im(env_name, seed, model_A)
    im_b = load_im(env_name, seed, model_B)

    # ratio_a = 255 / (im_a.max() - im_a.min())
    # img_a = ratio_a * (im_a - im_a.min())
    # title = "Importance Map for "+model_A
    # url = "./image/"+model_A+"/" + env_name + "/importance_map_seed_{}.png".format(seed)
    # save_array_to_img(img_a, title, url)

    #extract most # important grids,
    k_th_important = -20
    ind = np.sort(np.argpartition(im_a,kth=k_th_important,axis=1)[:,k_th_important:])
    print("Most important frames of ",model_A,": \n",ind)

    ind_b = np.sort(np.argpartition(im_b,kth=k_th_important,axis=1)[:,k_th_important:])
    print("Most important frames of ",model_B,": \n",ind_b)



    #generate_vid_from_demos(env_name, seed, model_A, ind)

    #export_cm_to_xlsx(env_name, seed, model_A,im)

    compare_im(env_name,seed,im_a,im_b)








