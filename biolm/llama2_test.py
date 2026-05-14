from tqdm import tqdm
import numpy as np
from biolm.modules.moe.moe_layers import LinearGLUMoELayer
from biolm.llama2 import *

def unitest_flash_attn():

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    '''
    print("Loading a pretrained model")
    ckpt_path = ('/public/home/taoshen/code/Projects/Zero2Hero/llama2/out_ckpts/mars/'
                 'mars_run-encoder-mars-s-add_attn_mask-2023_08_21_22_18_02/ckpt.pt')
    model = load_model(ckpt_path, device='cpu')
    '''

    print("Initializing a new model from scratch")
    model_type = 'decoder'
    model_args = get_model_args('es', model_type, vocab_size=3000)
    print(model_args)
    # model_args['pretrain_mode']='MLM'
    model_args['pretrain_mode']='GLM'
    gptconf = ModelArgs(**model_args)
    gptconf.layer_dropout = 0.5
    model = Transformer(gptconf)

    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    bs = 16
    seq_len = 512
    rand_array = np.random.randint(0, 11, (bs, seq_len))
    input = torch.LongTensor(rand_array)
    attn_mask = torch.ones((bs, seq_len))

    target = torch.ones((bs, seq_len)).long()
    print(f'input shape: {input.shape}')
    print(f'input: {input}')
    print(f'target shape: {target.shape}')

    is_cuda = False
    if is_cuda:
        model.half()
        model.to('cuda:0')
        input = input.to('cuda:0')
        attn_mask = attn_mask.to('cuda:0')
        target = target.to('cuda:0')

    # cal time for 100 iterations
    import time
    start = time.time()
    iters = 1

    for i in tqdm(range(iters)):
        output = model.forward(input, attn_mask, labels=target)
        model.last_loss.backward()

    end = time.time()
    print(f'time: {(end - start) / iters}')

    #12192 MiB time: 0.32055283784866334
    #12092 MiB

def unitest_transfomer():

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    '''
    print("Loading a pretrained model")
    ckpt_path = ('/public/home/taoshen/code/Projects/Zero2Hero/llama2/out_ckpts/mars/'
                 'mars_run-encoder-mars-s-add_attn_mask-2023_08_21_22_18_02/ckpt.pt')
    model = load_model(ckpt_path, device='cpu')
    '''

    print("Initializing a new model from scratch")
    model_args = get_model_args('es', 'encoder', vocab_size=3000)
    gptconf = ModelArgs(**model_args)
    gptconf.moe_args = None
    model = Transformer(gptconf)
    print(model)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    bs = 16
    seq_len = 1024
    rand_array = np.random.randint(0, 11, (bs, seq_len))
    input = torch.LongTensor(rand_array)
    attn_mask = torch.ones((bs, seq_len))

    target = torch.ones((bs, seq_len)).long()
    print(f'input shape: {input.shape}')
    print(f'input: {input}')

    global is_pre_cal_mask

    # is_pre_cal_mask = True
    # model.is_decoder = True
    # output = model.forward(input, attn_mask, target)

    device = 'cuda:0'

    model.to(device)
    input = input.to(device)
    attn_mask = attn_mask.to(device)

    # cal time for 100 iterations
    import time
    start = time.time()
    iters = 1000

    ctx = (
        torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    )
    with ctx:
        output = model.forward(input, attn_mask)

    print(output[0].dtype)

    # for i in tqdm(range(iters)):
    #     output = model.forward(input, attn_mask)
    end = time.time()
    print(f'time: {(end - start) / iters}')

    # Flash attention in pytorch 2.0
    # 3760MiB  time: 0.01803 0.0178 0.01828
    # 3908MiB  time: 0.01885 0.0187 0.01874
    # 3036MiB  time: 0.07310
    # 3212MiB

    # is_pre_cal_mask = False
    # model.is_decoder = True
    # output = model.forward(input, attn_mask, target)
    # print(output)
    #
    # is_pre_cal_mask = True
    # model.is_decoder = False
    # output = model.forward(input, attn_mask, target)
    # print(output)
    #
    # is_pre_cal_mask = False
    # model.is_decoder = False
    # output = model.forward(input, attn_mask, target)
    # print(output)

def cal_params():
    '''
    '''
    for model_size in ['lxxx']: #,'m','l','lx','lxx','es','eg','elx', 'lxx'
        for is_moe in [True, False]:
            # print("Initializing a new model from scratch")
            print(f'{model_size} is_moe {is_moe}')
            model_args = get_model_args(model_size, 'encoder', 121)
            gptconf = ModelArgs(**model_args)
            gptconf.moe_args = MoEArgs() if is_moe else None
            if gptconf.moe_args is not None:
                gptconf.moe_args.dropout = 0.3
            model = Transformer(gptconf)
            print(f'size {model_size} model, number of parameters: {sum(p.numel() for p in model.parameters())}')
            # print(model)
            # exit()

    # number of parameters: 5981472    6m
    # number of parameters: 25315840   25m
    # number of parameters: 84969216   85m
    # number of parameters: 160627104  160m
    # number of parameters: 1459981440 1.5b

    # es is_moe False
    # size es model, number of parameters:  7477120, 61513984        7.5m, 61m
    # size eg model, number of parameters:  33247680, 276643008      33m,  278m
    # size elx model, number of parameters: 148801280, 1229225600    148m, 1.2b
    # size lxx model, number of parameters: 650398720, 5407638592    650m, 5.4b
    # lxxx 1460175360 12166496832

def unitest_moe():
    '''
    '''

    config = MoEArgs()
    layer_index = 0
    gating_config = {
        # all gates
        "gate_type": config.gate_type,
        "gate_network": config.gate_network,
        "gate_use_softmax": config.gate_use_softmax,
        "gate_use_balance": config.gate_use_balance,
        "gate_balance_loss_weight": config.gate_balance_loss_weight,
        "gate_add_noise": config.gate_add_noise,
        # TopKBalancedNoisyGate
        "gate_noise_epsilon": config.gate_noise_epsilon,
    }
    calculator_config = {
        # all calculators
        "calculator_type": config.calculator_type,
        "multiply_gate_scores": config.multiply_gate_scores,
        "score_scale_factor": (
            config.score_scale_factor[layer_index]
            if isinstance(config.score_scale_factor, list)
            else config.score_scale_factor
        ),
        "add_weight_norm": config.add_weight_norm,
        # SwitchDropTokenCalculator
        "drop_tokens": config.drop_tokens,
        "dropped_padding": config.dropped_padding,
        "capacity_factor": config.capacity_factor,
    }

    hidden_size = 1024
    intermediate_size = 4096
    hidden_act = "gelu"

    mlp = LinearGLUMoELayer(
        input_size=hidden_size,
        hidden_size=intermediate_size,
        output_size=hidden_size,
        hidden_act=hidden_act,
        num_experts=config.num_experts,
        num_selects=config.num_selects,
        size_experts=None,
        bias=False,
        **gating_config,
        **calculator_config,
    )

    print(mlp)
    print(f'number of parameters: {sum(p.numel() for p in mlp.parameters())}')

    # pseudo input
    bs = 16
    seq_len = 1024
    rand_array = np.random.randint(0, 11, (bs, seq_len))
    # input = torch.LongTensor(rand_array)
    # attn_mask = torch.ones((bs, seq_len))
    fea = torch.rand(bs, seq_len, hidden_size)
    output = mlp(fea)
    print(output)

if __name__ == '__main__':
    # cal_params()
    unitest_flash_attn()
    # unitest_kv_cache()
    # unitest_moe()
    # unitest_transfomer()