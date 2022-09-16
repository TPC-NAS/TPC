
# from ZeroShotProxy import compute_zen_score, compute_te_nas_score, compute_syncflow_score, compute_gradnorm_score, compute_NASWOT_score, compute_doc_score, compute_doc_score_hardware
import Masternet
import PlainNet

def get_FLOPs(s, input_resolution, classes):
    # block_list = []
    the_FLOP = 0
    # print(s)
    while len(s) > 0:
        tmp_idx_1 = s.find('(')
        tmp_idx_2 = s.find(')')
        block_type      = s[0:tmp_idx_1]
        params_block    = s[tmp_idx_1+1:tmp_idx_2].split(",")
        s = s[tmp_idx_2+1:]

        if len(params_block) == 4 :
            in_channel    = int(params_block[0])
            out_channel   = int(params_block[1])
            stride        = int(params_block[2])
            sublayers     = int(params_block[3])
        else:
            in_channel    = int(params_block[0])
            out_channel   = int(params_block[1])
            stride        = int(params_block[2])
            bottleneck    = int(params_block[3])
            sublayers     = int(params_block[4])

        # print(block_type)
        if block_type == "SuperConvK1BNRELU":
            kernel_size = 1
            the_FLOP += in_channel * out_channel * (kernel_size**2) * (input_resolution**2)
        elif block_type == "SuperConvK3BNRELU":
            kernel_size = 3
            the_FLOP += in_channel * out_channel * (kernel_size**2) * (input_resolution**2) // (stride**2)
            if stride == 2:
                input_resolution = input_resolution // 2
        elif block_type == "SuperResK1K3K1" or block_type == "SuperResK1K5K1" or block_type == "SuperResK1K7K1":
            if block_type == "SuperResK1K3K1":
                kernel_size = 3
            elif block_type == "SuperResK1K5K1":
                kernel_size = 5
            else:
                kernel_size = 7

            the_FLOP += in_channel * out_channel * (1**2) * (input_resolution**2) // (stride**4) # resblockproj

            the_FLOP += in_channel * bottleneck * (1**2) * (input_resolution**2)
            the_FLOP += bottleneck * bottleneck * (kernel_size**2) * (input_resolution**2) // (stride**2)
            if stride == 2:
                input_resolution = input_resolution // 2
            the_FLOP += bottleneck * out_channel * (1**2) * (input_resolution**2)
            the_FLOP += bottleneck * out_channel * (1**2) * (input_resolution**2)
            the_FLOP += bottleneck * bottleneck * (kernel_size**2) * (input_resolution**2)
            the_FLOP += bottleneck * out_channel * 1 * (input_resolution**2)

            the_FLOP += out_channel * bottleneck * (1**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += bottleneck * bottleneck * (kernel_size**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += bottleneck * out_channel * (1**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += bottleneck * out_channel * (1**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += bottleneck * bottleneck * (kernel_size**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += bottleneck * out_channel * 1 * (input_resolution**2) * (sublayers-1)
        elif block_type == "SuperResK3K3" or block_type == "SuperResK5K5" or block_type == "SuperResK7K7":
            if block_type == "SuperResK3K3":
                kernel_size = 3
            elif block_type == "SuperResK5K5":
                kernel_size = 5
            else:
                kernel_size = 7

            the_FLOP += in_channel * out_channel * (1**2) * (input_resolution**2) // (stride**4) # resblockprj

            the_FLOP += in_channel * bottleneck * (kernel_size**2) * (input_resolution**2) // (stride**2)
            if stride == 2:
                input_resolution = input_resolution // 2
            the_FLOP += out_channel * bottleneck * (kernel_size**2) * (input_resolution**2)

            the_FLOP += out_channel * bottleneck * (kernel_size**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += out_channel * bottleneck * (kernel_size**2) * (input_resolution**2) * (sublayers-1)
        elif "SuperResIDW" in block_type:
            kernel_size = int(block_type[-1])
            expansion   = int(block_type[-3])
            dw_channel_1  = bottleneck  * expansion
            dw_channel_2  = out_channel * expansion

            the_FLOP += in_channel * bottleneck * (1**2) * (input_resolution**2) // (stride**4) # resblockprj

            the_FLOP += in_channel * dw_channel_1 * (1**2) * (input_resolution**2)
            the_FLOP += dw_channel_1 * (kernel_size**2) * (input_resolution**2) // (stride**2)
            if stride == 2:
                input_resolution = input_resolution // 2

            if out_channel != bottleneck:
                the_FLOP += out_channel * bottleneck * (1**2) * (input_resolution**2) * (sublayers-1)  # resblock
                the_FLOP += out_channel * bottleneck * (1**2) * (input_resolution**2) * (sublayers)    # resblock

            the_FLOP += dw_channel_1 * bottleneck * (1**2) * (input_resolution**2)
            the_FLOP += bottleneck * dw_channel_2 * (1**2) * (input_resolution**2)
            the_FLOP += dw_channel_2 * (kernel_size**2) * (input_resolution**2)
            the_FLOP += dw_channel_2 * out_channel * (1**2) * (input_resolution**2)

            the_FLOP += out_channel * dw_channel_1 * (1**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += dw_channel_1 * (kernel_size**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += dw_channel_1 * bottleneck * (1**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += bottleneck * dw_channel_2 * (1**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += dw_channel_2 * (kernel_size**2) * (input_resolution**2) * (sublayers-1)
            the_FLOP += dw_channel_2 * out_channel * (1**2) * (input_resolution**2) * (sublayers-1)

    # the_FLOP += classes * out_channel

    return the_FLOP

def get_model_size(s, classes):
    the_model_size = 0

    while len(s) > 0:
        tmp_idx_1 = s.find('(')
        tmp_idx_2 = s.find(')')
        block_type      = s[0:tmp_idx_1]
        params_block    = s[tmp_idx_1+1:tmp_idx_2].split(",")
        s = s[tmp_idx_2+1:]

        if len(params_block) == 4 :
            in_channel    = int(params_block[0])
            out_channel   = int(params_block[1])
            stride        = int(params_block[2])
            sublayers     = int(params_block[3])
        else:
            in_channel    = int(params_block[0])
            out_channel   = int(params_block[1])
            stride        = int(params_block[2])
            bottleneck    = int(params_block[3])
            sublayers     = int(params_block[4])

        # print(block_type)
        if block_type == "SuperConvK1BNRELU":
            kernel_size = 1
            the_model_size += in_channel * out_channel * (kernel_size**2)
        elif block_type == "SuperConvK3BNRELU":
            kernel_size = 3
            the_model_size += in_channel * out_channel * (kernel_size**2)
        elif block_type == "SuperResK1K3K1" or block_type == "SuperResK1K5K1" or block_type == "SuperResK1K7K1":
            if block_type == "SuperResK1K3K1":
                kernel_size = 3
            elif block_type == "SuperResK1K5K1":
                kernel_size = 5
            else:
                kernel_size = 7

            the_model_size += in_channel * out_channel * (1**2) # resblockproj

            the_model_size += in_channel * bottleneck * (1**2)
            the_model_size += bottleneck * bottleneck * (kernel_size**2)
            the_model_size += bottleneck * out_channel * (1**2)
            the_model_size += bottleneck * out_channel * (1**2)
            the_model_size += bottleneck * bottleneck * (kernel_size**2)
            the_model_size += bottleneck * out_channel * 1

            the_model_size += out_channel * bottleneck * (1**2) * (sublayers-1)
            the_model_size += bottleneck * bottleneck * (kernel_size**2) * (sublayers-1)
            the_model_size += bottleneck * out_channel * (1**2) * (sublayers-1)
            the_model_size += bottleneck * out_channel * (1**2) * (sublayers-1)
            the_model_size += bottleneck * bottleneck * (kernel_size**2) * (sublayers-1)
            the_model_size += bottleneck * out_channel * 1 * (sublayers-1)
        elif block_type == "SuperResK3K3" or block_type == "SuperResK5K5" or block_type == "SuperResK7K7":
            if block_type == "SuperResK3K3":
                kernel_size = 3
            elif block_type == "SuperResK5K5":
                kernel_size = 5
            else:
                kernel_size = 7

            the_model_size += in_channel * out_channel * (1**2) # resblockprj

            the_model_size += in_channel * bottleneck * (kernel_size**2)
            the_model_size += out_channel * bottleneck * (kernel_size**2)

            the_model_size += out_channel * bottleneck * (kernel_size**2) * (sublayers-1)
            the_model_size += out_channel * bottleneck * (kernel_size**2) * (sublayers-1)
        elif "SuperResIDW" in block_type:
            kernel_size = int(block_type[-1])
            expansion   = int(block_type[-3])
            dw_channel_1  = bottleneck  * expansion
            dw_channel_2  = out_channel * expansion

            the_model_size += in_channel * dw_channel_1 * (1**2)
            the_model_size += dw_channel_1 * (kernel_size**2) // (stride**2)
            the_model_size += dw_channel_1 * bottleneck * (1**2)
            the_model_size += bottleneck * dw_channel_2 * (1**2)
            the_model_size += dw_channel_2 * (kernel_size**2)
            the_model_size += dw_channel_2 * out_channel * (1**2)

            the_model_size += in_channel * dw_channel_1 * (1**2) * (sublayers-1)
            the_model_size += dw_channel_1 * (kernel_size**2) * (sublayers-1)
            the_model_size += dw_channel_1 * bottleneck * (1**2) * (sublayers-1)
            the_model_size += bottleneck * dw_channel_2 * (1**2) * (sublayers-1)
            the_model_size += dw_channel_2 * (kernel_size**2) * (sublayers-1)
            the_model_size += dw_channel_2 * out_channel * (1**2) * (sublayers-1)

    # the_model_size += classes * out_channel
    return the_model_size

def get_num_layers(s):
    the_num_layers = 0
    while len(s) > 0:
        tmp_idx_1 = s.find('(')
        tmp_idx_2 = s.find(')')
        block_type      = s[0:tmp_idx_1]
        params_block    = s[tmp_idx_1+1:tmp_idx_2].split(",")
        s = s[tmp_idx_2+1:]

        if len(params_block) == 4 :
            sublayers     = int(params_block[3])
        else:
            sublayers     = int(params_block[4])

        the_num_layers += sublayers

    return the_num_layers

if __name__ == "__main__":
    structure_string = "SuperConvK3BNRELU(3,64,1,1)SuperResIDWE2K7(64,256,1,16,1)SuperResIDWE2K7(256,64,2,32,3)SuperResIDWE1K7(64,64,2,64,1)SuperResIDWE2K3(64,64,2,128,4)SuperResIDWE6K7(64,64,1,128,1)SuperConvK1BNRELU(256,1080,1,1)"
    FLOPs = get_FLOPs(structure_string, input_resolution=32, classes=100)
    num_layers = get_num_layers(structure_string)
    model_size = get_model_size(structure_string, classes=100)
    print("my flop = ", FLOPs)
    print("my layers = ", num_layers)
    print("my params = ", model_size)

    AnyPlainNet = Masternet.MasterNet
    the_model = AnyPlainNet(num_classes=100, plainnet_struct=structure_string, no_create=True, no_reslink=False)
    the_model_flops = the_model.get_FLOPs(32)
    the_layers = the_model.get_num_layers()
    the_model_size = the_model.get_model_size()

    print("expected FLOP = ", the_model_flops)
    print("expected layers = ", the_layers)
    print("expected param = ", the_model_size)
