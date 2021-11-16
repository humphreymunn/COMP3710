import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

tf.config.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


latent_size = 128
img_size = (64,64)
batch_size = 50
dataset_name = "celeba_gan"
filters = 128
kernel_size = 4
epochs = 10000
learn_rate_disc = 0.0001
learn_rate_gen = 0.0001

data = keras.preprocessing.image_dataset_from_directory(dataset_name, label_mode=None, image_size=img_size, batch_size=batch_size)
#data = data.take(800) # use subset of data
# normalize data to [0, 1]
data = data.map(lambda x: x / 255.0)

# discriminator network
input_img = keras.Input(shape=(64, 64, 3))
x = layers.Conv2D(filters//2, kernel_size=kernel_size, strides=2, padding="same")(input_img)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Conv2D(filters, kernel_size=kernel_size, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Conv2D(filters, kernel_size=kernel_size, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.2)(x)
discriminator = layers.Dense(1, activation="sigmoid")(x)
discriminator = keras.Model(input_img, discriminator)
discriminator.summary()

# generator network
input_img = keras.Input(shape=(latent_size,))
x = layers.Dense(8 * 8 * filters)(input_img)
x = layers.Reshape((8, 8, filters))(x)
x = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
generator = layers.Conv2D(3, kernel_size=kernel_size, padding="same", activation="sigmoid")(x)
generator = keras.Model(input_img, generator)
generator.summary()

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_size):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_size = latent_size

    @property
    def metrics(self):
        return [self.generator_loss, self.discriminator_loss]
        
    def compile(self, gen_opt, disc_opt, loss):
        super(GAN, self).compile()
        self.loss = loss
        self.generator_loss = keras.metrics.Mean(name="generator loss")
        self.discriminator_loss = keras.metrics.Mean(name="discriminator loss")
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt

    @tf.function
    def train_step(self, real_images):
        
        batch_size = tf.shape(real_images)[0]

        latent_vecs = tf.random.normal([batch_size, self.latent_size])

        # get training data
        fake_images = self.generator(latent_vecs)
        all_images = tf.concat([fake_images, real_images], axis=0)

        # create real and fake labels
        fake_labels = tf.ones([batch_size, 1])
        real_labels = tf.zeros([batch_size, 1])
        classes = tf.concat([fake_labels, real_labels], axis=0)
        classes += tf.random.uniform(tf.shape(classes), maxval=0.05)

        # compute and apply gradient to discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_images)
            dis_loss = self.loss(classes, predictions)
        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.disc_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        latent_vecs = tf.random.normal([batch_size, self.latent_size])

        # compute and apply gradient to generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(latent_vecs)
            predictions = self.discriminator(fake_images)
            gen_loss = self.loss(real_labels, predictions)
        grads = tape.gradient(gen_loss, self.generator.trainable_weights)
        self.gen_opt.apply_gradients(zip(grads, self.generator.trainable_weights))

        # metrics
        self.generator_loss.update_state(gen_loss)
        self.discriminator_loss.update_state(dis_loss)
        return {
            "generator loss": self.generator_loss.result(),
            "discriminator loss": self.discriminator_loss.result(),
        }

class CallbackGAN(keras.callbacks.Callback):
    def __init__(self, latent_size):
        self.latent_size = latent_size

    ''' Generate fake images from GAN model. '''
    def on_epoch_end(self, epoch, logs=None):
        # create latent vectors
        z = tf.random.normal([12, self.latent_size])
        imgs = 255 * self.model.generator(z).numpy()
        for idx, img in enumerate(imgs):
            keras.preprocessing.image.array_to_img(img).save("D:\gan_data\GAN_image_%d_epoch_%02d.png" % (idx, epoch))

# Create model and train
gan = GAN(discriminator=discriminator, generator=generator, latent_size=latent_size)
optimizer1 = keras.optimizers.Adam(learning_rate=learn_rate_disc)
optimizer2 = keras.optimizers.Adam(learning_rate=learn_rate_gen)
gan.compile(gen_opt = optimizer2, disc_opt = optimizer1, loss=keras.losses.BinaryCrossentropy())
gan.fit(data, epochs=epochs, callbacks=[CallbackGAN(latent_size=latent_size)])
