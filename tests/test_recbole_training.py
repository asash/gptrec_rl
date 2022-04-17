from recbole.quick_start import run_recbole


if __name__ == "__main__":
    parameter_dict = {
        'neg_sampling': None,
    }
    run_recbole(model='BERT4Rec', dataset='ml-100k', config_dict=parameter_dict)