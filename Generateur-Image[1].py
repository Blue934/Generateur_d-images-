import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Configurer le TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # Détection du TPU
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# Charger les données
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Voici le type du x_train:", type(x_train))  # Vérification du type de données

# Normaliser les images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ajouter une dimension pour les canaux
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Encoder les labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Créer des datasets TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Définir le générateur
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=100, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(28 * 28 * 1, activation="tanh"))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Définir le discriminateur
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

# Charger les checkpoints si disponibles
checkpoint_filepath = 'gan_checkpoint.h5'
initial_epoch = 0

if os.path.exists(checkpoint_filepath):
    with strategy.scope():
        gan.load_weights(checkpoint_filepath)
        print(f"Checkpoint chargé depuis {checkpoint_filepath}")
    # Extraire l'époque initiale à partir du nom du fichier (à configurer selon ta méthode de sauvegarde des époques)
    initial_epoch = int(checkpoint_filepath.split('_')[-1].split('.'))

 # Compiler les modèles dans la stratégie TPU
with strategy.scope():
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Combiner les modèles GAN
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(100,))
    gan_output = discriminator(generator(gan_input))
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

# Entraîner le GAN avec sauvegarde progressive
def train_gan(gan, generator, discriminator, dataset, epochs, initial_epoch=0, checkpoint_filepath='gan_checkpoint.h5'):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_freq='epoch'
    )
    
    for epoch in range(initial_epoch, epochs):
        for real_images, _ in dataset:  # Remarque le `_` ici, pour ignorer les labels
            noise = np.random.normal(0, 1, (real_images.shape[0], 100))
            generated_images = generator.predict(noise)
            labels_real = tf.ones((real_images.shape[0], 1))
            labels_fake = tf.zeros((real_images.shape[0], 1))

            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

            noise = np.random.normal(0, 1, (real_images.shape[0], 100))
            labels_gan = tf.ones((real_images.shape[0], 1))
            g_loss = gan.train_on_batch(noise, labels_gan)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Discriminator Loss: {d_loss_real + d_loss_fake}, Generator Loss: {g_loss}")

        # Sauvegarde à la fin de chaque epoch
        gan.save_weights(checkpoint_filepath.format(epoch=epoch+1))
# Appeler la fonction d'entraînement
train_gan(gan, generator, discriminator, train_dataset, epochs=10000, initial_epoch=initial_epoch)
# Sauvegarder le modèle
gan.save('gan_model.h5')
print("Modèle sauvegardé avec succès.")