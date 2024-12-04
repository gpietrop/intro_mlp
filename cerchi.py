import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generazione di cerchi non concentrici
x_circles, y_circles = make_circles(n_samples=100, noise=0.1)

# Visualizzazione
plt.scatter(x_circles[np.where(y_circles == 0)[0], 0], x_circles[np.where(y_circles == 0)[0], 1],
            color='red', label='Classe 0')
plt.scatter(x_circles[np.where(y_circles == 1)[0], 0], x_circles[np.where(y_circles == 1)[0], 1],
            color='blue', label='Classe 1')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.tight_layout()
plt.show()

# Divisione in train e test
x_train_circles, x_test_circles, y_train_circles, y_test_circles = train_test_split(
    x_circles, y_circles, test_size=0.2, shuffle=True
)

# Preprocessing dei dati
y_train_circles = y_train_circles.reshape(-1, 1)
y_test_circles = y_test_circles.reshape(-1, 1)

# Creazione del modello
mlp_circles = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),  # Primo hidden layer
    tf.keras.layers.Dense(10, activation='relu'),  # Secondo hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Strato di output per classificazione binaria
])

# Compilazione del modello
lr = 0.01
mlp_circles.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss='binary_crossentropy',  # Funzione di perdita per classificazione binaria
    metrics=['accuracy']
)

# Addestramento del modello
n_epochs = 100
batch_size = 64
history = mlp_circles.fit(
    x_train_circles,
    y_train_circles,
    validation_split=0.2,
    epochs=n_epochs,
    batch_size=batch_size,
    verbose=1
)

# Plot della perdita (Training e Validation)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
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


# Valutazione sul set di test
test_loss, test_accuracy = mlp_circles.evaluate(x_test_circles, y_test_circles, verbose=0)
print(f"Loss sul test set: {test_loss}")
print(f"Accuracy sul test set: {test_accuracy}")

# Predizioni sul test set
y_pred_circles_probabilities = mlp_circles.predict(x_test_circles)
y_pred_circles = (y_pred_circles_probabilities >= 0.5).astype(int)

# Matrice di confusione
conf_matrix = confusion_matrix(y_test_circles, y_pred_circles)
ConfusionMatrixDisplay(conf_matrix, display_labels=['Rosso', 'Blu']).plot(colorbar=False)
plt.title("Matrice di confusione")
plt.show()
