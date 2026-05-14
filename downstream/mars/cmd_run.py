import os

def run(cmd):
    print(cmd)
    os.system(cmd)

def run_all():

    is_ema_model = False
    gradient_accumulation_steps = 1
    layer_dropout = 0.0

    # for data_mode in ['ss_pred']:
    #     cmd = f'torchrun --standalone --nproc_per_node=8 ./downstream/mars/run_train_bp16.py ' \
    #           f'--model_size {model_size} ' \
    #           f'--data_mode {data_mode} ' \
    #           f'--layer_dropout {layer_dropout} ' \
    #           f'--model_type {model_type} ' \
    #           f'--gradient_accumulation_steps {gradient_accumulation_steps} --is_wandb'
    #     cmd += f' --is_ema_model' if is_ema_model else ''
    #     cmd += f' --ckpt' if ckpt else ''
    #     cmd += f' --max_iters {max_iters}' if max_iters is not None else ''
    #
    #     if dropout != 0.1:
    #         cmd += f' --dropout {dropout}'
    #     print(cmd)
    #
    #     # run(cmd)

    for data_mode in ['mrna_pred']:
        cmd = f'torchrun --standalone ' \
              f'--nproc_per_node=8 ./downstream/mars/run_train_bp16.py ' \
              f'--model_size {model_size} ' \
              f'--data_mode {data_mode} ' \
              f'--layer_dropout {layer_dropout} ' \
              f'--model_type {model_type} ' \
              f'--gradient_accumulation_steps {gradient_accumulation_steps}'
        print(cmd)
        # run(cmd)

    # for data_mode in ['utr_pred']:  #
    #     cmd = f'torchrun --standalone ' \
    #           f'--nproc_per_node=8 ./downstream/mars/run_train_bp16.py ' \
    #           f'--model_size {model_size} ' \
    #           f'--data_mode {data_mode} ' \
    #           f'--layer_dropout {layer_dropout} ' \
    #           f'--model_type {model_type} ' \
    #           f'--gradient_accumulation_steps {gradient_accumulation_steps} --is_wandb'
    #     cmd += f' --dropout {dropout}'
    #     print(cmd)
    #     # run(cmd)

    # data_mode = 'cls_pred'
    # for cls_task in ['sirna']: #
    #     cmd = f'torchrun --standalone ' \
    #           f'--nproc_per_node=1 ./downstream/mars/run_train_bp16.py ' \
    #           f'--model_size {model_size} ' \
    #           f'--data_mode {data_mode} ' \
    #           f'--layer_dropout {layer_dropout} ' \
    #           f'--model_type {model_type} ' \
    #           f'--gradient_accumulation_steps {gradient_accumulation_steps} --cls_task {cls_task} --is_wandb'
    #     print(cmd)
        # run(cmd)


    #
    # for data_mode in ['en_pred']:  #
    #     cmd = f'torchrun --standalone ' \
    #           f'--nproc_per_node=8 ./downstream/mars/run_train_bp16.py ' \
    #           f'--model_size {model_size} ' \
    #           f'--data_mode {data_mode} ' \
    #           f'--layer_dropout {layer_dropout} ' \
    #           f'--model_type {model_type} ' \
    #           f'--gradient_accumulation_steps {gradient_accumulation_steps}'
    #     run(cmd)
    #
    # for data_mode in ['pa_pred']:  #
    #     cmd = f'torchrun --standalone ' \
    #           f'--nproc_per_node=8 ./downstream/mars/run_train_bp16.py ' \
    #           f'--model_size {model_size} ' \
    #           f'--data_mode {data_mode} ' \
    #           f'--layer_dropout {layer_dropout} ' \
    #           f'--model_type {model_type} ' \
    #           f'--gradient_accumulation_steps {gradient_accumulation_steps}'
    #     run(cmd)
    #
    # data_mode = 'cls_pred'
    # for cls_task in ['viral','promoter']: #
    #     cmd = f'torchrun --standalone ' \
    #           f'--nproc_per_node=8 ./downstream/mars/run_train_bp16.py ' \
    #           f'--model_size {model_size} ' \
    #           f'--data_mode {data_mode} ' \
    #           f'--layer_dropout {layer_dropout} ' \
    #           f'--model_type {model_type} ' \
    #           f'--gradient_accumulation_steps {gradient_accumulation_steps} --cls_task {cls_task}'
    #     run(cmd)

if __name__ == '__main__':
    # ckpt = True
    # model_size = 'lxx'
    # model_type = 'encoder'
    # run_all()
    # max_iters = 75 * 1000

    max_iters = None
    ckpt = True
    dropout = 0.1
    model_size = 'es'
    model_type = 'encoder-MLM_online'
    run_all()