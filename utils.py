import os.path
import torch
import numpy as np



def load_mask_from_im(env_name,model_type,seed,dl):
    inverse = True

    if model_type == 'GAIL':
        im = np.load('./output_npz/' + env_name + '/' + model_type + '_importance_map_' + str(seed) + '.npz')
    if model_type == 'BC':
        im = np.load('./output_npz/' + env_name + '/' + model_type + '_importance_map_' + str(seed) + '.npz')
    im = im['arr_0']

    im_flat = np.sort(im,axis=None)

    # dl is degradation level wrt percentage, which could be 0.1, 0.2, ....
    if inverse:
        threshold = im_flat[int(len(im_flat) * dl)]
        mask = np.where(im <= threshold, 1, 0)
    else:
        threshold = im_flat[-int(len(im_flat)*dl)]
        mask = np.where(im >= threshold,1,0)


    assert mask.shape == im.shape
    mask = np.expand_dims(mask,axis=-1)
    return mask

def get_flat_grads(f, net):
    flat_grads = torch.cat([
        grad.reshape(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads


def get_flat_params(net):
    return torch.cat([param.reshape(-1) for param in net.parameters()])


def set_params(net, new_flat_params):
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx


def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(
    g, s, Hs, max_kl, L, kld, old_params, pi, max_iter=10,
    success_ratio=0.1
):
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio \
            and actual_improv > 0 \
                and kld_new < max_kl:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params

def save_checkpoint(state, checkpoint_dir, env_name, extra_info):
    filename = checkpoint_dir + '/' + env_name + '_' + extra_info + '_network.pth.tar'
    print("Saving checkpoint at " + filename + " ...")
    torch.save(state, filename)  # save checkpoint
    print("Saved checkpoint.")


def get_checkpoint(checkpoint_dir):
    resume_weights = checkpoint_dir + '/network.pth.tar'
    if torch.cuda.is_available():
        print("Attempting to load Cuda weights...")
        checkpoint = torch.load(resume_weights)
        print("Loaded weights.")
    else:
        print("Attempting to load weights for CPU...")
        # Load GPU model on CPU
        checkpoint = torch.load(resume_weights,
                                map_location=lambda storage,
                                                    loc: storage)
        print("Loaded weights.")
    return checkpoint


def long_tensor(input):
    if torch.cuda.is_available():
        return torch.cuda.LongTensor(input)
    else:
        return torch.LongTensor(input)


def float_tensor(input):
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(input)
    else:
        return torch.FloatTensor(input)


def perform_no_ops(ale, no_op_max, preprocessor, state):
    # perform nullops
    num_no_ops = np.random.randint(1, no_op_max + 1)
    for _ in range(num_no_ops):
        ale.act(0)
        preprocessor.add(ale.getScreenRGB())
    if len(preprocessor.preprocess_stack) < 2:
        ale.act(0)
        preprocessor.add(ale.getScreenRGB())
    state.add_frame(preprocessor.preprocess())


def normalize_state(obs):
    return obs / 255.0


# custom masking function for covering up the score/life portions of atari games
def mask_score(obs, env_name):
    obs_copy = obs.copy()
    if env_name == "spaceinvaders" or env_name == "breakout" or env_name == "pong":
        # takes a stack of four observations and blacks out (sets to zero) top n rows
        n = 10
        # no_score_obs = copy.deepcopy(obs)
        obs_copy[:, :n, :, :] = 0
    elif env_name == "beamrider":
        n_top = 16
        n_bottom = 11
        obs_copy[:, :n_top, :, :] = 0
        obs_copy[:, -n_bottom:, :, :] = 0
    elif env_name == "enduro":
        n_top = 0
        n_bottom = 14
        obs_copy[:, :n_top, :, :] = 0
        obs_copy[:, -n_bottom:, :, :] = 0
    elif env_name == "hero":
        n_top = 0
        n_bottom = 30
        obs_copy[:, :n_top, :, :] = 0
        obs_copy[:, -n_bottom:, :, :] = 0
    elif env_name == "qbert":
        n_top = 12
        # n_bottom = 0
        obs_copy[:, :n_top, :, :] = 0
        # obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name == "seaquest":
        n_top = 12
        n_bottom = 16
        obs_copy[:, :n_top, :, :] = 0
        obs_copy[:, -n_bottom:, :, :] = 0
        # cuts out divers and oxygen
    elif env_name == "mspacman":
        n_bottom = 15  # mask score and number lives left
        obs_copy[:, -n_bottom:, :, :] = 0
    elif env_name == "videopinball":
        n_top = 15
        obs_copy[:, :n_top, :, :] = 0
    elif env_name == "montezumarevenge":
        n_top = 10
        obs_copy[:, :n_top, :, :] = 0
    else:
        print("NOT MASKING SCORE FOR GAME: " + env_name)
        pass
        # n = 20
        # obs_copy[:,-n:,:,:] = 0
    return obs_copy


def preprocess(ob, env_name):
    # print("masking on env", env_name)
    return mask_score(normalize_state(ob), env_name)
