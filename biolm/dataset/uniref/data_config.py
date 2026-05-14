# get script path

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
    seq_db_root = '/public/home/taoshen/data/protein/uniref'

    if db == 'uniref5090':
        train_data_config = [
            (f"{seq_db_root}/uniref50_bin", "train_char", 1.0 / 2.0),
            (f"{seq_db_root}/uniref90_bin", "train_char", 1.0 / 2.0),
        ]
        val_data_config = [
            (f"{seq_db_root}/uniref50_bin_val", "train_char", 1.0/2.0),
        ]
    elif db == 'demo':
        train_data_config = [
            (f"./data/demo_bin/", "train_char", 1.0),
        ]
    else:
        raise NotImplementedError

    return train_data_config, val_data_config
