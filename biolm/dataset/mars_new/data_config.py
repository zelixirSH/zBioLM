
def get_data_config(db, ss_mlm_prob = 0.15):
    '''

    '''

    '''
    # new_db_v1
    # train_data_config = [
    #     ("/public_new/taoshen/data/MARS_split_0.1m/atgc/ncs_TinyLlamaBin_sfv2", "train", 1.0/4.0),
    #     ("/public_new/taoshen/data/MARS_split_0.1m/atgc/cds_TinyLlamaBin_sfv2", "train", 1.0/4.0),
    #     ("/public/home/taoshen/data/rna/sequence_database/rnacentral_2023-02-24/TinyLlamaBin/rnacentral_species_specific_ids_0.8_rep_seq_TinyLlamaBin", "train", 1.0/2.0),
    # ]
    '''

    val_data_config = None
    seq_db_root = '/public/home/taoshen/data/rna/sequence_database'

    if db == 'debug':
        train_data_config = [
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/ncs_TinyLlamaBin_sfv2", "train_char", 1.0/4.0, False),
        ]

    elif db == 'ss_0.8':
        train_data_config = [
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_TinyLlamaBin", "train", 1.0/2.0, True, ss_mlm_prob),
        ]

    elif db == 'ss_0.8_half':
        train_data_config = [
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_half_bin", "train", 1.0/2.0, True, ss_mlm_prob),
        ]

    elif db == 'ss_1.0':
        train_data_config = [
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_all_TinyLlamaBin", "train", 1.0/2.0, True, ss_mlm_prob),
        ]

    elif db == 'new_db_v1.1':
        train_data_config = [
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/ncs_TinyLlamaBin_sfv2", "train_char", 1.0/4.0, False),
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/cds_TinyLlamaBin_sfv2", "train_char", 1.0/4.0, False),
            (f"{seq_db_root}/rnacentral_2023-02-24/TinyLlamaBin/rnacentral_species_specific_ids_0.8_rep_seq_TinyLlamaBin", "train_char", 1.0/2.0, False),
        ]

    elif db == 'new_db_v1.2':
        train_data_config = [
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/ncs_TinyLlamaBin_sfv2", "train_char", 1.0/4.0, False),
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/cds_TinyLlamaBin_sfv2", "train_char", 1.0/4.0, False),
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_TinyLlamaBin", "train", 1.0/2.0, True, ss_mlm_prob),
        ]

    elif db == 'new_db_v1.2.0':
        train_data_config = [
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/ncs_TinyLlamaBin_sfv2", "train_char", 1.0/4.0, False),
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/cds_TinyLlamaBin_sfv2", "train_char", 1.0/4.0, False),
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_all_TinyLlamaBin", "train", 1.0/2.0, True, ss_mlm_prob),
        ]

    elif db == 'new_db_v1.2.1':
        # new_db_v1.2.1
        train_data_config = [
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/ncs_TinyLlamaBin_sfv2", "train_char", 1.0/2.0, False),
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/cds_TinyLlamaBin_sfv2", "train_char", 1.0/2.0, False),
            (f"{seq_db_root}/utrdb/fasta_TinyLlamaBin_sfv2", "train_char", 1.0/4.0, False),
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_TinyLlamaBin", "train", 1.0, True, ss_mlm_prob),
        ]

    elif db == 'new_db_v1.2.2':
        # new_db_v1.2.1
        train_data_config = [
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/ncs_TinyLlamaBin_sfv2", "train_char", 1.0/2.0, False),
            ("/public_new/taoshen/data/MARS_split_0.1m/atgc/cds_TinyLlamaBin_sfv2", "train_char", 1.0/2.0, False),
            (f"{seq_db_root}/utrdb/fasta_TinyLlamaBin_sfv2", "train_char", 1.0/4.0, False),
            ("/public/home/taoshen/gary/project/dataset/codon/CDSs/split_fastas_TinyLlamaBin_sfv2", "train_3mer", 1.0/4.0, False),
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_TinyLlamaBin", "train", 1.5, True, ss_mlm_prob),
        ]

    elif db == 'new_db_v1.3':
        # 0.8 mars & 0.8 rnacentral
        train_data_config = [
            (f"{seq_db_root}/MARS/remove_sim_0.8_8192_clstr/rep_seq/rep_seqs/fasta_0.1m_TinyLlamaBin_sfv2", "train_char", 1.0/2.0, False),
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_TinyLlamaBin", "train", 1.0/2.0, True, ss_mlm_prob),
        ]

    elif db == 'new_db_v1.3.0':
        # 0.8 mars & 1.0 rnacentral
        train_data_config = [
            (f"{seq_db_root}/MARS/remove_sim_0.8_8192_clstr/rep_seq/rep_seqs/fasta_0.1m_TinyLlamaBin_sfv2", "train_char", 1.0/2.0, False),
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_all_TinyLlamaBin", "train", 1.0 / 2.0, True, ss_mlm_prob),
        ]

    elif db == 'new_db_v1.4':
        # 0.8 mars & 0.8 rnacentral
        train_data_config = [
            (f"{seq_db_root}/MARS/remove_sim_0.8_8192_clstr/rep_seq/rep_seqs/fasta_0.1m_TinyLlamaBin_sfv2", "train_char", 0.5, False),
            (f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_TinyLlamaBin",                 "train",      0.5, True, ss_mlm_prob),
            (f"{seq_db_root}/ensembl_cds/rm_sim_0.8/all_0.8_rep_seq_split_fastas_512_csv_bin",              "train",      0.1, True, ss_mlm_prob),
            (f"{seq_db_root}/utrdb/rm_sim_0.8/all_0.8_rep_seq_split_ss_1024_csv_bin",                       "train",      0.1, True, ss_mlm_prob),
        ]

    elif db == 'ScalingLaw':
        # 0.8 mars
        train_data_config = [
            (f"{seq_db_root}/MARS/remove_sim_0.8_8192_clstr/rep_seq/rep_seqs/fasta_0.1m_TinyLlamaBin_sfv2_train", "train_char", 1, False),
        ]
        val_data_config = [
            (f"{seq_db_root}/MARS/remove_sim_0.8_8192_clstr/rep_seq/rep_seqs/fasta_0.1m_TinyLlamaBin_sfv2_test", "train_char", 1, False),
        ]

    else:
        raise NotImplementedError

    return train_data_config, val_data_config