# %%
import tensorflow as tf
# from tensorflow import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10 as cf10

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# %%
# Cargar las imagenes de entrenamiento y prueba
(train_images, train_labels), (test_images, test_labels) = cf10.load_data()

# %%
train_images.shape, train_labels.shape, test_images.shape, test_labels.shape

# %%
def show_images(train_images, class_names, train_labels, nb_samples=12, nb_row=4):
    """
    Muestra una cuadrícula de imágenes de muestra del conjunto de datos de entrenamiento.
    
    Parámetros:
    train_images: Array de imágenes de entrenamiento.
    class_names: Lista de nombres de clases correspondientes a las etiquetas.
    train_labels: Array de etiquetas de entrenamiento.
    nb_samples: Número de imágenes a mostrar.
    nb_row: Número de filas en la cuadrícula.
    """
    plt.figure(figsize=(10, 10))
    for i in range(nb_samples):
        plt.subplot(nb_row, nb_row, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

# %%
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
show_images(train_images, class_names, train_labels)

# %%
# Normalizar las imagenes
max_value = 255

train_images = train_images / max_value
test_images = test_images / max_value

# Codificacion one-hot de las etiquetas
train_labels = tf.keras.utils.to_categorical(train_labels, len(class_names))
test_labels = tf.keras.utils.to_categorical(test_labels, len(class_names))

train_images.shape, test_images.shape, train_labels.shape, test_labels.shape

# %%
# Variables de entrada
INPUT_SHAPE = (32, 32, 3)
FILTER1_SIZE = 32
FILTER2_SIZE = 64
FILTER_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
FULLY_CONNECTED = 128
NUM_CLASSES = len(class_names)

# Crear el modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(FILTER1_SIZE, FILTER_SHAPE, activation='relu', input_shape=INPUT_SHAPE))
model.add(tf.keras.layers.MaxPooling2D(POOL_SHAPE))
model.add(tf.keras.layers.Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(POOL_SHAPE))
model.add(tf.keras.layers.Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(FULLY_CONNECTED, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))


# %%
model.summary()

# %%
BATCH_SIZE = 32
EPOCHS = 20

METRICS = ['accuracy', tf.keras.metrics.Recall(name='precision'), tf.keras.metrics.Precision(name='recall')]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)

training_history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_images, test_labels))

# %%
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_images, test_labels)

print(f'Perdida: {test_loss}')
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')

# %%
def show_performance_curve(training_result, metric, metric_label):
    """
    Muestra las curvas de rendimiento de entrenamiento y validación para un métrico dado.
    
    Parámetros:
    training_result: Objeto History devuelto por model.fit().
    metric: El métrico a graficar (por ejemplo, 'accuracy', 'recall', 'precision').
    metric_label: La etiqueta para el eje y de la gráfica.
    """
    training_performance = training_result.history[str(metric)]
    validation_performance = training_result.history[f'val_{metric}']
    intersection_idx = np.argwhere(np.isclose(training_performance, validation_performance, atol=0.01)).flatten()[0]

    intersection_value = training_performance[intersection_idx]

    plt.plot(training_performance, label=metric_label)
    plt.plot(validation_performance, label=f'val_{metric}')
    plt.axvline(intersection_idx, color='gray', linestyle='--', label=f'Intersection: {intersection_value}')
    plt.annotate(f'Valor Optimo: {intersection_value}', xy=(intersection_idx, intersection_value), xycoords='data', fontsize = 10, ha='center', color='blue')

    plt.xlabel('Epoch')
    plt.ylabel(metric_label)
    plt.legend(loc='lower right')

# %%
show_performance_curve(training_history, 'accuracy', 'Accuracy')

# %%
show_performance_curve(training_history, 'recall', 'Recall')

# %%
show_performance_curve(training_history, 'precision', 'Precision')

# %%
test_prediction = model.predict(test_images)
test_prediction_labels = np.argmax(test_prediction, axis=1)

test_true_labels = np.argmax(test_labels, axis=1)

cf_matrix = confusion_matrix(test_true_labels, test_prediction_labels)

confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=class_names)

confusion_matrix_display.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
plt.show()
