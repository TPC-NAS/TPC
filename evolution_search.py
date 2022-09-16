import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, random, logging, time
import torch
from torch import nn
import numpy as np
import global_utils
import Masternet
import PlainNet
# from tqdm import tqdm

from ZeroShotProxy import compute_zen_score, compute_te_nas_score, compute_syncflow_score, compute_gradnorm_score, compute_NASWOT_score, compute_tpc_score, compute_tpc_score_hardware
import benchmark_network_latency
from random_string_param import get_FLOPs, get_num_layers, get_model_size

working_dir = os.path.dirname(os.path.abspath(__file__))

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--zero_shot_score', type=str, default=None,
                        help='could be: Zen (for Zen-NAS), TE (for TE-NAS)')
    parser.add_argument('--search_space', type=str, default=None,
                        help='.py file to specify the search space.')
    parser.add_argument('--evolution_max_iter', type=int, default=int(48e4),
                        help='max iterations of evolution.')
    parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--budget_latency', type=float, default=None, help='latency of forward inference per mini-batch, e.g., 1e-3 means 1ms.')
    parser.add_argument('--max_layers', type=int, default=None, help='max number of layers of the network.')
    parser.add_argument('--batch_size', type=int, default=None, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--population_size', type=int, default=512, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default=None, help='output directory')
    parser.add_argument('--gamma', type=float, default=1e-2, help='noise perturbation coefficient')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

def get_new_random_structure_str(structure_str, num_replaces=1, search_space="resnet"):
    block_list = []
    while len(structure_str) > 0:
        tmp_idx_1 = structure_str.find('(')
        tmp_idx_2 = structure_str.find(')')
        block_list.append(structure_str[0:tmp_idx_2+1])

        structure_str = structure_str[tmp_idx_2+1:]

    selected_random_id_set = set()
    for replace_count in range(num_replaces):
        random_id = random.randint(0, len(block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)

        def channel_mutator(channel):
            channel = round(channel * random.choice([4, 2, 1, 0.5, 0.25]))
            if channel < 8 :
                channel = 8

            return channel

        def sublayer_mutator(sublayer):
            sublayer = sublayer + random.randint(-2, 2)

            if sublayer < 1:
                sublayer_1 = 1
                sublayer_2 = 0
            elif sublayer > 6:
                sublayer_1 = sublayer // 2
                sublayer_2 = sublayer // 2
            else:
                sublayer_1 = sublayer
                sublayer_2 = 0

            return sublayer_1, sublayer_2

        def layer_type_mutator(search_space):
            if search_space == "resnet":
                random_type = random.randint(2, 7)
                if random_type == 2 :
                    return "SuperResK1K3K1"
                elif random_type == 3 :
                    return "SuperResK1K5K1"
                elif random_type == 4 :
                    return "SuperResK1K7K1"
                elif random_type == 5 :
                    return "SuperResK3K3"
                elif random_type == 6 :
                    return "SuperResK5K5"
                elif random_type == 7 :
                    return "SuperResK7K7"
            elif search_space == "mobilenet":
                random_type = random.randint(2, 13)
                if random_type == 2:
                    return "SuperResIDWE1K3"
                elif random_type == 3:
                    return "SuperResIDWE2K3"
                elif random_type == 4:
                    return "SuperResIDWE4K3"
                elif random_type == 5:
                    return "SuperResIDWE6K3"
                elif random_type == 6:
                    return "SuperResIDWE1K5"
                elif random_type == 7:
                    return "SuperResIDWE2K5"
                elif random_type == 8:
                    return "SuperResIDWE4K5"
                elif random_type == 9:
                    return "SuperResIDWE6K5"
                elif random_type == 10:
                    return "SuperResIDWE1K7"
                elif random_type == 11:
                    return "SuperResIDWE2K7"
                elif random_type == 12:
                    return "SuperResIDWE4K7"
                elif random_type == 13:
                    return "SuperResIDWE6K7"
            else:
                raise Exception("Search space is not supported.")

        old_structure_str = block_list[random_id]
        new_structure_str = ""

        index_1         = old_structure_str.find('(')
        block_type      = old_structure_str[0:index_1]
        params_block    = old_structure_str[index_1+1:-1].split(",")

        params_block    = [int(i) for i in params_block]

        sublayer_2 = 0
        if len(params_block) == 4 :
            params_block[1] = channel_mutator(params_block[1])
        else:
            block_type              = layer_type_mutator(search_space)
            params_block[1]         = channel_mutator(params_block[1])
            params_block[3]         = channel_mutator(params_block[3])
            sublayer_1, sublayer_2  = sublayer_mutator(params_block[4])
            params_block[4]         = sublayer_1

        params_block    = [str(i) for i in params_block]
        if sublayer_2 != 0:
            new_student_str_1   = block_type + "(" + ','.join(params_block) + ")"
            params_block[2]     = "1"
            new_student_str_2   = block_type + "(" + ','.join(params_block) + ")"
            block_list[random_id] = new_student_str_1
            block_list.insert(random_id+1, new_student_str_2)
        else:
            new_student_str_1   = block_type + "(" + ','.join(params_block) + ")"
            block_list[random_id] = new_student_str_1

    tmp_idx_1       = block_list[0].find('(')
    tmp_idx_2       = block_list[0].find(')')
    block_type      = block_list[0][0:tmp_idx_1]
    params_block    = block_list[0][tmp_idx_1+1:-1].split(",")

    in_channels     = params_block[1]

    for i in range(len(block_list)-1):
        current_block   = block_list[i+1]
        tmp_idx_1       = current_block.find('(')
        tmp_idx_2       = current_block.find(')')
        block_type      = current_block[0:tmp_idx_1]
        params_block    = current_block[tmp_idx_1+1:-1].split(",")

        params_block[0] = in_channels
        if i == len(block_list)-2:
            params_block[1] = str(in_channels)
        in_channels     = params_block[1]

        new_student_str_1   = block_type + "(" + ','.join(params_block) + ")"
        block_list[i+1] = new_student_str_1

    new_random_structure_str = ''.join(block_list)

    return new_random_structure_str


def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert hasattr(the_net, 'split')
    splitted_net_str = the_net.split(split_layer_threshold=6)
    return splitted_net_str

def get_latency(AnyPlainNet, random_structure_str, gpu, args):
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=False)
    if gpu is not None:
        the_model = the_model.cuda(gpu)
    the_latency = benchmark_network_latency.get_model_latency(model=the_model, batch_size=args.batch_size,
                                                              resolution=args.input_image_size,
                                                              in_channels=3, gpu=gpu, repeat_times=1,
                                                              fp16=True)
    del the_model
    torch.cuda.empty_cache()
    return the_latency

def compute_nas_score(AnyPlainNet, random_structure_str, gpu, args):
    # compute network zero-shot proxy score
    if not args.zero_shot_score == 'FLOP_hardware':
        no_create = False
        the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str, no_create=no_create, no_reslink=True)
        the_model = the_model.cuda(gpu)

    try:
        if args.zero_shot_score == 'Zen':
            the_nas_core_info = compute_zen_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    mixup_gamma=args.gamma, batch_size=args.batch_size,
                                                                    repeat=1)
            the_nas_core = the_nas_core_info['avg_nas_score']
        elif args.zero_shot_score == 'FLOP':
            the_nas_core_info = compute_tpc_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    mixup_gamma=args.gamma, batch_size=args.batch_size,
                                                                    repeat=1)
            the_nas_core = the_nas_core_info['avg_nas_score']
        elif args.zero_shot_score == 'FLOP_hardware':
            the_nas_core_info = compute_tpc_score_hardware.compute_nas_score(random_structure_str)
            the_nas_core = the_nas_core_info['avg_nas_score']
        elif args.zero_shot_score == 'TE-NAS':
            the_nas_core = compute_te_nas_score.compute_NTK_score(model=the_model, gpu=gpu,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)

        elif args.zero_shot_score == 'Syncflow':
            the_nas_core = compute_syncflow_score.do_compute_nas_score(model=the_model, gpu=gpu,
                                                                       resolution=args.input_image_size,
                                                                       batch_size=args.batch_size)

        elif args.zero_shot_score == 'GradNorm':
            the_nas_core = compute_gradnorm_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    batch_size=args.batch_size)

        elif args.zero_shot_score == 'Flops':
            the_nas_core = the_model.get_FLOPs(args.input_image_size)

        elif args.zero_shot_score == 'Params':
            the_nas_core = the_model.get_model_size()

        elif args.zero_shot_score == 'Random':
            the_nas_core = np.random.randn()

        elif args.zero_shot_score == 'NASWOT':
            the_nas_core = compute_NASWOT_score.compute_nas_score(gpu=gpu, model=the_model,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)
    except Exception as err:
        logging.info(str(err))
        logging.info('--- Failed structure: ')
        the_nas_core = -9999

    if not args.zero_shot_score == 'FLOP_hardware':
        del the_model
        torch.cuda.empty_cache()
    return the_nas_core

def main(args, argv):
    gpu = args.gpu
    if gpu is not None:
        torch.cuda.set_device('cuda:{}'.format(gpu))
        torch.backends.cudnn.benchmark = True

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    if os.path.isfile(best_structure_txt):
        print('skip ' + best_structure_txt)
        return None

    # load masternet
    AnyPlainNet = Masternet.MasterNet

    masternet = AnyPlainNet(num_classes=args.num_classes, opt=args, argv=argv, no_create=True)
    initial_structure_str = str(masternet)

    popu_structure_list = []
    popu_zero_shot_score_list = []
    popu_latency_list = []

    start_timer = time.time()
    for loop_count in range(args.evolution_max_iter):
        # too many networks in the population pool, remove one with the smallest score
        while len(popu_structure_list) > args.population_size:
            min_zero_shot_score = min(popu_zero_shot_score_list)
            tmp_idx = popu_zero_shot_score_list.index(min_zero_shot_score)
            popu_zero_shot_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)
            popu_latency_list.pop(tmp_idx)
        pass

        if loop_count >= 1 and loop_count % 1000 == 0:
            max_score = max(popu_zero_shot_score_list)
            min_score = min(popu_zero_shot_score_list)
            elasp_time = time.time() - start_timer
            logging.info(f'loop_count={loop_count}/{args.evolution_max_iter}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h')

        # ----- generate a random structure ----- #
        if len(popu_structure_list) <= 10:
            random_structure_str = get_new_random_structure_str(structure_str=initial_structure_str, num_replaces=1, search_space=args.search_space)
        else:
            tmp_idx = random.randint(0, len(popu_structure_list) - 1)
            tmp_random_structure_str = popu_structure_list[tmp_idx]
            random_structure_str = get_new_random_structure_str(structure_str=tmp_random_structure_str, num_replaces=2, search_space=args.search_space)

        # random_structure_str = get_splitted_structure_str(AnyPlainNet, random_structure_str, num_classes=args.num_classes)
        # the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str, no_create=True, no_reslink=False)

        if args.max_layers is not None:
            the_layers     = get_num_layers(random_structure_str)
            if args.max_layers < the_layers:
                continue

        if args.budget_model_size is not None:
            the_model_size = get_model_size(random_structure_str, classes=args.num_classes)
            if args.budget_model_size < the_model_size:
                continue

        if args.budget_flops is not None:
            # if the_model is None:
            #     the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
            #                             no_create=True, no_reslink=False)
            # the_model_flops = the_model.get_FLOPs(args.input_image_size)
            the_model_flops = get_FLOPs(random_structure_str, input_resolution=args.input_image_size, classes=args.num_classes)
            if args.budget_flops < the_model_flops:
                continue

        the_latency = np.inf
        if args.budget_latency is not None:
            the_latency = get_latency(AnyPlainNet, random_structure_str, gpu, args)
            if args.budget_latency < the_latency:
                continue

        the_nas_core = compute_nas_score(AnyPlainNet, random_structure_str, gpu, args)

        popu_structure_list.append(random_structure_str)
        popu_zero_shot_score_list.append(the_nas_core)
        popu_latency_list.append(the_latency)

    return popu_structure_list, popu_zero_shot_score_list, popu_latency_list






if __name__ == '__main__':
    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    info = main(args, sys.argv)
    if info is None:
        exit()



    popu_structure_list, popu_zero_shot_score_list, popu_latency_list = info

    # export best structure
    best_score = max(popu_zero_shot_score_list)
    best_idx = popu_zero_shot_score_list.index(best_score)
    best_structure_str = popu_structure_list[best_idx]
    the_latency = popu_latency_list[best_idx]

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    global_utils.mkfilepath(best_structure_txt)
    with open(best_structure_txt, 'w') as fid:
        fid.write(best_structure_str)
    pass  # end with
