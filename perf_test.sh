run_perf(){
    model="$1"
    seq_len="$2"
    tracer="$3"
    ckpt="$4"
    cmd="lightning run model --node-rank=0 --accelerator=cuda --devices=8 --num-nodes=1 ./scripts/pretrain_mx.py --devices 8 --max_step 10 --model_size ${model} --seq_len ${seq_len}"
    logfile="logs/perf_${model}_${seq_len}"
    if [ $tracer -ne 0 ]; then
        cmd=$cmd" --tracer"
        logfile=${logfile}_tracer
    fi
    
    if [ $ckpt -ne 0 ]; then
        cmd=$cmd" --ckpt_inter 1"
        logfile=${logfile}_ckpt
    fi
    logfile=${logfile}.log
    echo "run "$cmd
    echo "logfile is "$logfile
    $cmd 2>&1 | tee $logfile
}

# sudo apt install -y tee

mkdir -p logs


run_perf es 511 0 0

run_perf es 511 1 0

run_perf es 512 0 0

run_perf es 512 1 0


run_perf eg 511 0 0

run_perf eg 511 1 0

run_perf eg 512 0 0

run_perf eg 512 1 0


run_perf eg 511 0 1

run_perf eg 511 1 1

run_perf eg 512 0 1

run_perf eg 512 1 1


run_perf lx 511 0 1

run_perf lx 511 1 1

run_perf lx 512 0 1

run_perf lx 512 1 1


run_perf lxx 511 0 1

run_perf lxx 511 1 1

run_perf lxx 512 0 1

run_perf lxx 512 1 1