import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import global_utils, argparse, ModelLoader, time
from PlainNet import basic_blocks

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def compute_nas_score(gpu, model, mixup_gamma, resolution, batch_size, repeat, fp16=False):
    # model.eval()
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    # print(model.no_reslink)

    with torch.no_grad():
        for repeat_count in range(repeat):
            # network_weight_gaussian_init(model)

            # input  = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)

            index = 0
            # input_buffer = input
            # test_log_bn_scaling_factor  = 0
            test_log_conv_scaling_factor = 0
            for name, m in model.named_modules():
                if len(list(m.children())) > 1 :
                    if index < 2 :
                        index = index + 1
                        continue

                    for small_module in m.children():
                        # input_buffer = small_module(input_buffer)
                        if isinstance(small_module, basic_blocks.ConvKX):
                            score = torch.tensor(float(small_module.out_channels * (small_module.kernel_size ** 2) / (small_module.stride **2)))
                            test_log_conv_scaling_factor += torch.log(score)
                        if isinstance(small_module, basic_blocks.ConvDW):
                            score = torch.tensor(float((small_module.kernel_size ** 2) / (small_module.stride **2)))
                            test_log_conv_scaling_factor += torch.log(score)


                    index = index + 1
                    # print(test_log_conv_scaling_factor)

            nas_score = torch.tensor(1.0)
            nas_score = torch.log(nas_score) + test_log_conv_scaling_factor
            nas_score_list.append(float(nas_score))

            assert not (nas_score != nas_score)

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)

    # print("avg_nas_score = ", avg_nas_score)

    return info


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)
    the_model = ModelLoader.get_model(opt, sys.argv)
    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)


    start_timer = time.time()
    info = compute_nas_score(gpu=args.gpu, model=the_model, mixup_gamma=args.mixup_gamma,
                             resolution=args.input_image_size, batch_size=args.batch_size, repeat=args.repeat_times, fp16=False)
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info['avg_nas_score']
    print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')
