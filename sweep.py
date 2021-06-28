import wandb
from data.pad_images import PADDED_IMAGE_LEN
from train import train

sweep_config = {
    'method': 'bayes', #grid, random, bayes
    'metric': {
      'name': 'disc_acc',
      'goal': 'minimize',
      'target': 0.5
    },
    'parameters': {
            'image_shape': {
                'value': (200, 200, 3)
            },
            'num_examples': {
                'value': PADDED_IMAGE_LEN

            },
            'num_samples': {
                'value': 20
            },
            # 'grid_count': {
            #     'values': [1, 2]
            # },
            'training_loop': {
                'value': 'simultaneous',
                # 'values': ['simultaneous', 'batch_split', 'full_split']
                'values': ['simultaneous', 'batch_split']
            },
            'generator_seed_dim': {
                # 'value': 50
                'distribution': 'int_uniform',
                'min': 250,
                'max': 500
            },
            'adversarial_epochs': {
                # 'value': 50
                'distribution': 'int_uniform',
                'min': 100,
                'max': 200
            },
            'discriminator_examples': {
                'value': PADDED_IMAGE_LEN
            },
            'generator_examples': {
                'value': PADDED_IMAGE_LEN
            },
            'generator_epochs': {
                'value': 1
            },
            'discriminator_epochs': {
                'value': 1
            },
            'batch_size': {
                # 'value': 128
                'values': [32, 64, 128]
            },
            'generator_learning_rate': {
                # 'value': 1e-4
                'distribution': 'log_uniform',
                'min': -10,
                'max': -9
            },
            'discriminator_learning_rate': {
                # 'value': 1e-4
                'distribution': 'log_uniform',
                'min': -10,
                'max': -9
            },
            'generator_learning_rate_decay': {
                # 'value': 0.9
                'min': 0.9,
                'max': 1
            },
            'discriminator_learning_rate_decay': {
                # 'value': 0.9
                'min': 0.9,
                'max': 1
            },
            'discriminator_dropout_rate': {
                # value: 0.3
                'min': 0.15,
                'max': 0.25
            }
    },
    'early_terminate': {
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27
    }

}



if(__name__ == '__main__'):
    sweep_id = wandb.sweep(sweep_config, 
                            project="hs_cards_3")

    wandb.agent(sweep_id, train)