
| model                                        | best val f1         | best test f1        |
|----------------------------------------------|---------------------|---------------------|
| encoder-s-2023_10_15_18_49_11                | 0.7091 / 0.7103     | 0.7106              |
| encoder-m-2023_10_15_19_18_29                | 0.7315 / 0.7213     | 0.7293              |
| encoder-l-2023_10_15_19_27_58                | 0.7340              | 0.7339              |
| encoder-lx-2023_10_15_19_24_59               | 0.7361 / 0.7307     | 0.7411              | 
| encoder-lxx-2023_10_15_19_29_52              | 0.7332              | 0.7361              |
|                                              |                     |                     |
| lt5t5-s-2023_10_27_00_10_52                  | 0.702               | 0.7102              |
| lt5t5-m-2023_10_27_18_33_45                  | 0.7285              | 0.7232              |
| lt5t5-l-2023_10_30_09_03_38                  | 0.7230              | 0.7159              |
|                                              |                     |                     |
| lt5t5_dbRNACentral0.8-s-2023_10_27_22_07_45  | 0.7213              | 0.7155              |
| lt5t5_dbRNACentral-s-2023_10_27_00_25_06     | 0.7156              | 0.7156              |
| lt5ul2_dbRNACentral0.8-s-2023_10_29_23_57_35 | 0.7030              | 0.7096              |
| lt5ul2_dbRNACentral-s-2023_10_29_00_18_42    | 0.7007              | 0.7036              |
| lt5t5-dbRNAcentral0.8_3tok                   | 0.7141              | 0.7070              |
|                                              |                     |                     |
| lt5ul2_demo-s-dl6-2023_10_29_00_15_53        | 0.7294              | 0.7270              |
| lt5ul2_demo-s-2023_10_25_16_28_54            | 0.7195              | 0.7147              |
| lt5t5_demo-s-dl6-2023_10_26_12_22_12         | 0.7101 (unfinished) | 0.7113 (unfinished) |
| lt5t5_demo-s-2023_10_25_16_24_22             | 0.7277              | 0.7236              |


| Model-S                                  | best val f1 | best test f1 |
|------------------------------------------|-------------|--------------|
| llama2_enc_dec-t5_dbRNACentral0.8        | 0.7213      | 0.7155       |
| llama2_enc_dec-t5+SSMASK_dbRNACentral0.8 | 0.7279      | 0.7190       |
| llama2_enc_dec-t5_dbRNACentral           | 0.7156      | 0.7156       |
| llama2_enc_dec-ul2_dbRNACentral0.8       | 0.7030      | 0.7096       |
| llama2_enc_dec-ul2_dbRNACentral          | 0.7007      | 0.7036       |
| llama2_enc_dec-t5-dbRNAcentral0.8_3tok   | 0.7141      | 0.7070       |

| Model-S                                     | Protein Abundance Prediction (codom optimization) - R |
|---------------------------------------------|-------------------------------------------------------|
| esm2-mlm                                    | 0.710                                                 |
| llama2_enc-mlm                              | 0.7362                                                |
|                                             |                                                       |
| llama2_enc_dec-t5-RNAcentral                | 0.6442                                                |
| llama2_enc_dec-t5-RNAcentral0.8             | 0.6656                                                |
| llama2_enc_dec-ul2-RNAcentral               | 0.6012                                                |
| llama2_enc_dec-ul2-RNAcentral0.8            | 0.6457                                                |
| llama2_enc_dec-t5-RNAcentral0.8_3tok - char | 0.6723                                                |
| llama2_enc_dec-t5-RNAcentral0.8_3tok - 3mer | 0.6723                                                |
| llama2_enc_dec-t5-RNAcentral0.8_3tok - bpe  | 0.6723                                                |

| Model                                | Ecoli promoter-non promoter classification - F1 | 
|--------------------------------------|-------------------------------------------------|
| esm2-mlm                             | 0.864                                           | 
| llama2_enc-mlm                       | 0.9052                                          |
|                                      |                                                 |
| llama2_enc_dec-t5-RNAcentral         | 0.8314                                          |
| llama2_enc_dec-t5-RNAcentral0.8      |                                                 |
| llama2_enc_dec-ul2-RNAcentral        | 0.8102                                          |
| llama2_enc_dec-ul2-RNAcentral0.8     | 0.8232                                          |
| llama2_enc_dec-t5-RNAcentral0.8_3tok | 0.8517                                          |

| Model                                | viral - F1 |
|--------------------------------------|------------|
| esm2-mlm                             | 0.514      |
| llama2_enc-mlm                       | 0.5604     |
|                                      |            |
| llama2_enc_dec-t5-RNAcentral         | N/A        |
| llama2_enc_dec-t5-RNAcentral0.8      | 0.5063     |
| llama2_enc_dec-ul2-RNAcentral        | 0.2875     |
| llama2_enc_dec-ul2-RNAcentral0.8     | 0.4730     |
| llama2_enc_dec-t5-RNAcentral0.8_3tok | 0.5063     |

| Model                                 | mRNA_de Pred |
|---------------------------------------|--------------|
| esm2-mlm                              | 0.331        |
| llama2_enc-mlm                        | 0.3229       |
| llama2_enc-mlm-finetune175k           |              |
| llama2_enc_dec-t5                     |              |
| llama2_enc_dec-t5-RNAcentral          | 0.3265       |
| llama2_enc_dec-t5-RNAcentral0.8       | 0.3196       |
| llama2_enc_dec-ul2-RNAcentral         | 0.3246       |
| llama2_enc_dec-ul2-RNAcentral0.8      | 0.3284       |
| llama2_enc_dec-t5-RNAcentral0.8_3tok  | 0.3164       |


