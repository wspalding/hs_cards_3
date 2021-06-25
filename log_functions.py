from numpy.lib.npyio import save
from tensorflow.keras import models
import wandb
import numpy as np
from tensorflow.keras.utils import plot_model

class WandbLogger():
    def __init__(self) -> None:
        self.logs = {}

    def push_logs(self):
        wandb.log(self.logs)
        self.logs = {}

    def log_generator(self, epoch, logs):
        for key, value in logs.items():
            self.logs['generator_{}'.format(key)] = value

    def log_discriminator(self, epoch, logs):
        for key, value in logs.items():
            self.logs['discriminator_{}'.format(key)] = value

    def log_metrics(self, logs):
        for key, value in logs.items():
            self.logs[key] = value

    def sample_images(self, generator, noise, samples):
        gen_imgs = generator.predict(noise)
        self.logs['examples'] = [wandb.Image(np.squeeze(i)) for i in gen_imgs]

        for i, s in enumerate(samples):
            s.append(np.reshape(gen_imgs[i], [3, 200, 200]) * 255.0)
        self.logs['progession'] = [wandb.Video(np.array(s)) for s in samples]

    def log_model_images(self, model):
        save_file = "{}.png".format(model.name)
        plot_model(model, to_file=save_file)
        self.logs['{} architecture'.format(model.name)] = wandb.Image(save_file)

