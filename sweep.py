import wandb
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
                'value': (28, 28, 1)
            },
            'num_examples': {
                'value': 60000
            },
            'num_samples': {
                'value': 20
            },
            # 'grid_count': {
            #     'values': [1, 2]
            # },
            'training_loop': {
                'value': 'simultaneous'
                # 'values': ['simultaneous', 'batch_split', 'full_split']
            },
            'generator_seed_dim': {
                # 'value': 50
                'distribution': 'int_uniform',
                'min': 25,
                'max': 50
            },
            'adversarial_epochs': {
                # 'value': 50
                'distribution': 'int_uniform',
                'min': 50,
                'max': 100
            },
            'discriminator_examples': {
                'value': 60000
            },
            'generator_examples': {
                'value': 60000
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
                'min': 0.2,
                'max': 0.4
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
                            project="WandB_Interview_Demo")

    wandb.agent(sweep_id, train)