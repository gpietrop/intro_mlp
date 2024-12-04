import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Caricamento e preprocessamento del dataset MNIST
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

# One-hot encoding delle etichette
y_train_mnist_enc = tf.keras.utils.to_categorical(y_train_mnist)
y_test_mnist_enc = tf.keras.utils.to_categorical(y_test_mnist)

# Normalizzazione dei pixel delle immagini
x_train_mnist_norm = x_train_mnist.astype(np.float32) / 255.0
x_test_mnist_norm = x_test_mnist.astype(np.float32) / 255.0

# Creazione del modello MLP
mnist_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten per convertire l'immagine in un vettore
    tf.keras.layers.Dense(128, activation='relu'),  # Primo hidden layer
    tf.keras.layers.Dropout(0.2),  # Dropout per regolarizzazione
    tf.keras.layers.Dense(10, activation='softmax')  # Strato di output per classificazione multiclasse
])

# Compilazione del modello
lr = 0.001
loss_fn = tf.keras.losses.CategoricalCrossentropy()
mnist_mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=loss_fn,
    metrics=['accuracy']
)

# Addestramento del modello
n_epochs = 25
mnist_mlp_history = mnist_mlp.fit(
    x_train_mnist_norm, y_train_mnist_enc,
    validation_split=0.2,
    epochs=n_epochs,
    verbose=1
)

# Plot delle perdite di training e validazione
train_loss = mnist_mlp_history.history['loss']
val_loss = mnist_mlp_history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, 'r-', label='Training Loss')
plt.plot(epochs, val_loss, 'b--', label='Validation Loss')
plt.title('Andamento della Loss durante l\'addestramento')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Valutazione del modello sul set di test
mnist_test_loss, mnist_test_accuracy = mnist_mlp.evaluate(x_test_mnist_norm, y_test_mnist_enc, verbose=0)
print('Test accuracy: ', mnist_test_accuracy)

# Predizioni sul test set
y_pred_mnist_prob = mnist_mlp.predict(x_test_mnist_norm)
y_pred_mnist = np.argmax(y_pred_mnist_prob, axis=1)

# Matrice di confusione
conf_matrix = confusion_matrix(y_test_mnist, y_pred_mnist)
ConfusionMatrixDisplay(conf_matrix, display_labels=np.arange(10)).plot(colorbar=True, cmap='viridis')
plt.title("Matrice di Confusione (MNIST)")
plt.show()


# Visualizzazione di un'immagine con le sue probabilità predette
# Plot di un'immagine e la sua probabilità predetta
def plot_dimmagine_prob(prob, y, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    y_pred = np.argmax(prob)
    plt.xlabel('{} ({:.2f}%)'.format(int(y_pred), 100 * np.max(prob)))


# Plot delle probabilità per ogni classe
def plot_prob(prob, y):
    plt.grid(False)
    plt.yticks([])
    plt.xticks(np.arange(10))
    prob_bar = plt.bar(np.arange(10), prob, color='grey')
    plt.ylim([0, 1])
    y_pred = np.argmax(prob)
    prob_bar[y].set_color('red')
    prob_bar[y_pred].set_color('green')


plot_idx = np.random.choice(y_test_mnist.shape[0], 1)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_immagine_prob(
    y_pred_mnist_prob[plot_idx].squeeze(),
    y_test_mnist[plot_idx].squeeze(),
    x_test_mnist[plot_idx].squeeze()
)
plt.subplot(1, 2, 2)
plot_prob(
    y_pred_mnist_prob[plot_idx].squeeze(),
    y_test_mnist[plot_idx].squeeze()
)
plt.show()
