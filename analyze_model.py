import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import ModelLoader
import global_utils
# from ptflops import get_model_complexity_info

def main(opt, argv):
    model = ModelLoader.get_model(opt, argv)
    # flops, params = get_model_complexity_info(model, (3, opt.input_image_size, opt.input_image_size),
    #                                           as_strings=False,
    #                                           print_per_layer_stat=True)

    flops  = model.get_FLOPs(opt.input_image_size)
    params = model.get_model_size()

    print('Flops:  {:4g}'.format(flops))
    print('Params: {:4g}'.format(params))

    best_flop_txt = os.path.join(opt.save_dir, 'best_flop.txt')
    best_param_txt = os.path.join(opt.save_dir, 'best_param.txt')
    global_utils.mkfilepath(best_flop_txt)
    global_utils.mkfilepath(best_param_txt)

    with open(best_flop_txt, 'w') as fid:
        fid.write(str(flops))

    with open(best_param_txt, 'w') as fid:
        fid.write(str(params))

if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)

    main(opt, sys.argv)
