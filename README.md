# Biolm

Biolm is a cutting-edge machine learning framework designed for biological data analysis and modeling. This tool enables researchers and data scientists to train powerful models with ease, leveraging the capabilities of modern hardware configurations, including NVIDIA's A100 GPUs.

## Installation

To get started with Biolm, clone this repository to your local machine. You can then install Biolm along with its dependencies using the following command:

```bash
pip install -e .
```

For a detailed list of dependencies, please refer to the `requirements.txt` file. Additionally, an `environment.yml` file is provided, which outlines the conda environment used in our actual A100 setup. To create a similar environment, you can use:

```bash
conda env create -f environment.yml
```

## Preprocess

数据预处理分为2个步骤，一是将用户输入的单个fasta文件切分，二是将切分的文件处理成预训练的数据格式

第一步`./scripts/split_fasta.py` 用于将一个大的FASTA文件分割成多个小的FASTA文件，每个文件包含指定数量的序列。

该脚本接受以下命令行参数：

- `--fasta_file`: 输入的FASTA文件的路径。
- `--n`: 每个输出文件中的序列数量。
- `--output_dir`: 保存分割文件的目录。

以下是一个使用示例：
```bash
python ./scripts/split_fasta.py --fasta_file ./data/example/split_0.fasta --n 100000 --output_dir ./data/example/split
```
在这个示例中，`./data/example/split_0.fasta`文件将被分割成多个文件，每个文件包含100000个序列，这些文件将被保存在`./data/example/split`目录中。

第二步`./scripts/preprocess/pretok.py` 用于将FASTA文件处理成预训练的数据格式，该脚本接受以下命令行参数：

- `source_path`：FASTA文件的路径。
- `destination_path`：二进制文件的输出路径。
- `chunk_size`：每个二进制文件的大小（以字节为单位）。
- `split`：数据集的划分方式，可以是"train"、"validation"或"test"。
- `percentage`：要处理的FASTA文件的百分比。
- `seed`：用于随机打乱文件名的种子。
- `n_jobs`：同时运行的进程数量。
- `num_files_per_chunk`：每个块中的文件数量。
- `data_type`:  Protein/NA

```bash
python ./scripts/preprocess/pretok.py --source_path ./data/example/split --destination_path ./data/example/bin --split train --percentage 1.0 --seed 42 --n_jobs 64 --num_files_per_chunk 64 
```

这将会处理`./data/example/split`目录下的所有FASTA文件，然后将处理后的二进制文件保存到`./data/example/bin`目录下。

### 注意事项

- 确保输入的FASTA文件存在，且包含有效的FASTA序列。
- 确保指定的输出目录存在，或者脚本有权限创建该目录。
- 分割的文件将以`split_{file_count}.fasta`的形式命名，其中`file_count`是一个从0开始的计数器。
- 对于预训练来说 n 通常设置为 100000，不需要做太多调整
- 对于预训练来说 chunk_size 通常设置为 2049 * 1024，不需要做太多调整

## Training a Demo Model

Biolm allows you to quickly train a demo model using sample data located in `./data/example/bin`. To train a model with different sizes, you can use the `model_size` parameter, which supports the following options:

- `es`: corresponds to an 8 million parameter model.
- `eg`: corresponds to a 35 million parameter model.
- `lx`: corresponds to a 150 million parameter model.
- `lxx`: corresponds to a 650 million parameter model.

These sizes are reflective of the model dimensions reported in our testing documentation.

To train a model with the smallest size (`es`) as an example, you can run the following command:

```bash
lightning run model --node-rank=0 --accelerator=cuda --devices=8 --num-nodes=1 ./scripts/pretrain.py --devices 8 --model_size es --train_data_dir ./data/example/bin --val_data_dir ./data/example/bin
```

Please adjust the `--model_size` parameter according to your requirements.

## Downstream Tasks

### NA - Secondary Structure Prediction

To train a model for NA secondary structure prediction from pretrained model, you can use the following command, pretrained models are listed in [config_new_pipline.yaml](downstream%2Fmars%2Fconfigs%2Fconfig_new_pipline.yaml)

```bash
torchrun --standalone --nproc_per_node=8 ./downstream/mars/run_train_bp16.py --model_size es --data_mode ss_pred --layer_dropout 0.0 --model_type encoder-MLM_online --gradient_accumulation_steps 1 --is_wandb --ckpt
```

### NA - mRNA degradation prediction

To train a model for NA mRNA degradation prediction from pretrained model, you can use the following command, pretrained models are listed in [config_new_pipline.yaml](downstream%2Fmars%2Fconfigs%2Fconfig_new_pipline.yaml)

```bash
torchrun --standalone --nproc_per_node=8 ./downstream/mars/run_train_bp16.py --model_size es --data_mode mrna_pred --layer_dropout 0.0 --model_type encoder-MLM_online --gradient_accumulation_steps 1
```

### Protein (WIP [run_train_bp16.py](downstream%2Funiref%2Frun_train_bp16.py))

## Contributing

We welcome contributions from the community! If you have suggestions for improvements or bug fixes, please feel free to submit an issue or pull request.

## License

[Specify License Here]

## Citation

If you use Biolm in your research, please consider citing it. [Add citation instructions here]

For any questions or further assistance, don't hesitate to open an issue or contact the repository maintainers.

---

This template provides a solid foundation for your project's README. Remember to replace placeholder text like "[Specify License Here]" and the citation section with the appropriate details specific to Biolm.