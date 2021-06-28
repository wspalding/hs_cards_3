import tensorflow as tf
import numpy as np
from tensorflow.python.keras.metrics import Accuracy, BinaryAccuracy, \
    TruePositives, TrueNegatives, FalsePositives, FalseNegatives
import wandb
from wandb.keras import WandbCallback

import os
import time
from collections import defaultdict

from log_functions import WandbLogger
from models.generator import create_generator, generator_loss
from models.discriminator import create_discriminator, discriminator_loss

from data.dataset import image_generator


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
# # Normalize the images to [-1, 1]
# x_train = (x_train - 127.5) / 127.5  



def train(config=None):
    wandb.init(config=config)
    config = wandb.config

    # Batch and shuffle the data
    dataset = tf.data.Dataset.from_generator(image_generator, output_types=(tf.float32)).batch(config.batch_size).shuffle(config.num_examples)
    # dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(config.num_examples).batch(config.batch_size)
    sample_noise = tf.random.normal([config.num_samples, config.generator_seed_dim])
    # sample_types = np.array([0,1,2,3,4,0,1,2,3,4,5,6,7,8,9,5,6,7,8,9])
    samples = [[] for _ in range(config.num_samples)]

    generator = create_generator(config)
    discriminator = create_discriminator(config)
    generator_optimizer = tf.keras.optimizers.Adam(config.generator_learning_rate, beta_1=config.generator_learning_rate_decay)
    discriminator_optimizer = tf.keras.optimizers.Adam(config.discriminator_learning_rate, beta_1=config.discriminator_learning_rate_decay)

    wandb_logger = WandbLogger()
    gen_acc = BinaryAccuracy()
    gen_TP = TruePositives()
    gen_FN = FalseNegatives()

    disc_acc = BinaryAccuracy()
    disc_TP = TruePositives()
    disc_TN = TrueNegatives()
    disc_FP = FalsePositives()
    disc_FN = FalseNegatives()

    wandb_logger.log_model_images(generator)
    wandb_logger.log_model_images(discriminator)
    wandb_logger.sample_images(generator, sample_noise, samples)
    wandb_logger.push_logs()


    @tf.function
    def train_step(images):
        noise = tf.random.normal([config.batch_size, config.generator_seed_dim])
        metrics = {}
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


        metrics['gen_batch_loss'] = gen_loss / len(images)
        metrics['disc_batch_loss'] = disc_loss / len(images)
        gen_acc.update_state(tf.ones_like(fake_output), fake_output)
        gen_TP.update_state(tf.ones_like(fake_output), fake_output)
        gen_FN.update_state(tf.ones_like(fake_output), fake_output)

        disc_acc.update_state(tf.zeros_like(fake_output), fake_output)
        disc_acc.update_state(tf.ones_like(real_output), real_output)
        disc_TP.update_state(tf.zeros_like(fake_output), fake_output)
        disc_TP.update_state(tf.ones_like(real_output), real_output)
        disc_TN.update_state(tf.zeros_like(fake_output), fake_output)
        disc_TN.update_state(tf.ones_like(real_output), real_output)
        disc_FP.update_state(tf.zeros_like(fake_output), fake_output)
        disc_FP.update_state(tf.ones_like(real_output), real_output)
        disc_FN.update_state(tf.zeros_like(fake_output), fake_output)
        disc_FN.update_state(tf.ones_like(real_output), real_output)


        metrics['gen_gradient'] = gradients_of_generator
        metrics['disc_gradient'] = gradients_of_discriminator


        return metrics

    @tf.function
    def train_disc_step(images):
        noise = tf.random.normal([config.batch_size, config.generator_seed_dim])
        metrics = {}
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            # gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        # gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


        # metrics['gen_batch_loss'] = gen_loss
        metrics['disc_batch_loss'] = disc_loss
        # gen_acc.update_state(tf.ones_like(fake_output), fake_output)
        disc_acc.update_state(tf.zeros_like(fake_output), fake_output)
        disc_acc.update_state(tf.ones_like(real_output), real_output)
        disc_TP.update_state(tf.zeros_like(fake_output), fake_output)
        disc_TP.update_state(tf.ones_like(real_output), real_output)
        disc_TN.update_state(tf.zeros_like(fake_output), fake_output)
        disc_TN.update_state(tf.ones_like(real_output), real_output)
        disc_FP.update_state(tf.zeros_like(fake_output), fake_output)
        disc_FP.update_state(tf.ones_like(real_output), real_output)
        disc_FN.update_state(tf.zeros_like(fake_output), fake_output)
        disc_FN.update_state(tf.ones_like(real_output), real_output)
        # metrics['gen_gradient'] = gradients_of_generator
        metrics['disc_gradient'] = gradients_of_discriminator


        return metrics

    @tf.function
    def train_gen_step(images):
        noise = tf.random.normal([config.batch_size, config.generator_seed_dim])
        metrics = {}
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            # real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            # disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        # gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        # discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


        metrics['gen_batch_loss'] = gen_loss
        # metrics['disc_batch_loss'] = disc_loss
        gen_acc.update_state(tf.ones_like(fake_output), fake_output)
        gen_TP.update_state(tf.ones_like(fake_output), fake_output)
        gen_FN.update_state(tf.ones_like(fake_output), fake_output)
        # disc_acc.update_state(tf.zeros_like(fake_output), fake_output)
        # disc_acc.update_state(tf.ones_like(real_output), real_output)
        metrics['gen_gradient'] = gradients_of_generator
        # metrics['disc_gradient'] = gradients_of_discriminator


        return metrics



    for epoch in range(config.adversarial_epochs):
        print('=====================================================================')
        print('Adversarian Epoch: {}/{}'.format(epoch+1, config.adversarial_epochs))
        print('=====================================================================')
        # for i, image_batch in enumerate(dataset):
        #     print('{}/{}'.format(i+1, len(dataset)), end='\r')
        #     metrics = train_step(image_batch)
        # wandb_logger .sample_images(generator, sample_noise, samples)
        # wandb_logger.push_logs()

        start = time.time()
        gen_loss = 0
        disc_loss = 0
        logs = {}

        if config.training_loop == 'simultaneous':
            for i, image_batch in enumerate(dataset):
                print('{}'.format(i+1), end='\r')
                metrics = train_step(image_batch)
                gen_loss += metrics['gen_batch_loss']
                disc_loss += metrics['disc_batch_loss']
                wandb.log({'disc_grad': wandb.Histogram(np.array(metrics['disc_gradient'][0], dtype=float)),
                        'gen_grad': wandb.Histogram(np.array(metrics['gen_gradient'][0], dtype=float))})

        elif config.training_loop == 'batch_split':
            for i, image_batch in enumerate(dataset):
                print('{}'.format(i+1), end='\r')
                disc_metrics = train_disc_step(image_batch)
                wandb.log({'disc_grad': wandb.Histogram(np.array(disc_metrics['disc_gradient'][0], dtype=float))})
                disc_loss += disc_metrics['disc_batch_loss']

                gen_metrics = train_gen_step(image_batch)
                wandb.log({'gen_grad': wandb.Histogram(np.array(gen_metrics['gen_gradient'][0], dtype=float))})
                gen_loss += gen_metrics['gen_batch_loss']
                
                
        # elif config.training_loop == 'full_split':
        #     for i, image_batch in enumerate(dataset):
        #         print('{}/{}'.format(i+1, len(dataset)), end='\r')
        #         disc_metrics = train_disc_step(image_batch)
        #         wandb.log({'disc_grad': wandb.Histogram(np.array(disc_metrics['disc_gradient'][0].values, dtype=float))})
        #         disc_loss += disc_metrics['disc_batch_loss']
        #     print('')
        #     for i, image_batch in enumerate(dataset):
        #         print('{}/{}'.format(i+1, len(dataset)), end='\r')
        #         gen_metrics = train_gen_step(image_batch)
        #         wandb.log({'gen_grad': wandb.Histogram(np.array(gen_metrics['gen_gradient'][0].values, dtype=float))})
        #         gen_loss += gen_metrics['gen_batch_loss']
                

        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        logs['gen_loss'] = gen_loss 
        logs['disc_loss'] = disc_loss 

        logs['gen_acc'] = gen_acc.result().numpy()
        logs['gen_TP'] = gen_TP.result().numpy()
        logs['gen_FN'] = gen_FN.result().numpy()

        logs['disc_acc'] = disc_acc.result().numpy()
        logs['disc_TP'] = disc_TP.result().numpy()
        logs['disc_TN'] = disc_TN.result().numpy()
        logs['disc_FP'] = disc_FP.result().numpy()
        logs['disc_FN'] = disc_FP.result().numpy()

        wandb_logger.log_metrics(logs)

        gen_acc.reset_states()
        gen_TP.reset_states()
        gen_FN.reset_states()
        disc_acc.reset_states()
        disc_TP.reset_states()
        disc_TN.reset_states()
        disc_FP.reset_states()
        disc_FN.reset_states()
        generator.save(os.path.join(wandb.run.dir, "generator.h5"))
        discriminator.save(os.path.join(wandb.run.dir, "discriminator.h5"))

        wandb_logger.sample_images(generator, sample_noise, samples)
        wandb_logger.push_logs()






if __name__ == '__main__':
    config = {
        'image_shape': (200, 200, 3),
        'generator_seed_dim': 200,
        'adversarial_epochs': 50,
        'training_loop': 'full_split',
        'num_examples': 60000,
        'num_samples': 20,
        'batch_size': 64,
        'discriminator_dropout_rate': 0.2,
        'generator_learning_rate': 1e-4,
        'discriminator_learning_rate': 1e-4,
        'generator_learning_rate_decay': 0.9,
        'discriminator_learning_rate_decay': 0.9
    }
    train(config=config)