#generateur de texte 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model


# Chargement et vectorisation du texte français depuis un fichier local
fichier = input("Entrer le nom e votre fichier.csv:\n")

print("Voici la taille du jeu de donnée:", len(fichier))  
filepath = fichier # Remplacez par le chemin de votre fichier
with open(filepath, encoding='utf-8') as f:
    french_text = f.read()

print(french_text[:80])

text_vec_layer = tf.keras.layers.TextVectorization(split="character", standardize="lower")
text_vec_layer.adapt([french_text])
encoded = text_vec_layer([french_text])[0]

encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars
dataset_size = len(encoded)  # total number of chars

print(n_tokens)
print(dataset_size)

# Création du dataset
def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)

length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1_000_000], length=length, shuffle=True, seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000], length=length)
test_set = to_dataset(encoded[1_060_000:], length=length)

# Définition et compilation du modèle
tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=32),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.GRU(256, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint("my_french_model.keras", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10, callbacks=[model_ckpt])

# Sauvegarder le modèle
model.save('mon_modele.')

# Télécharger le fichier du modèle
#from google.colab import files
#files.download('mon_modele.h5')
# Création du modèle complet pour la génération de texte
french_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])

# Génération de texte
def next_char(text, temperature=1):
    text = tf.constant([text])  # Convertir la liste en tenseur
    y_proba = french_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]

def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text



tf.random.set_seed(42)
print(extend_text("Comment ", temperature=0.01))
print(extend_text("Pourquoi ", temperature=1))
