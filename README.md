# Proyecto de Detección de Objetos con PyTorch y VOC Dataset

Este proyecto muestra cómo trabajar con **datasets de detección de objetos**, realizar **visualización de bounding boxes**, aplicar **augmentations** con Albumentations y entrenar un **modelo simple de regresión de bounding boxes** con PyTorch.

---

## Lo aprendido

1. **Carga y exploración de datos**
   - Se descargó el dataset VOC de `torchvision.datasets.VOCDetection`.
   - Se exploraron las anotaciones (`labels`) y se extrajeron las bounding boxes.
   - Se creó la función `get_sample(ix)` para obtener imágenes y sus anotaciones.

2. **Visualización**
   - Se usó `matplotlib` para visualizar imágenes y bounding boxes.
   - Se implementó `plot_anns(img, anns)` para mostrar las cajas delimitadoras junto con las etiquetas.
   - Se aprendió a manejar múltiples objetos por imagen.

3. **Augmentations**
   - Se aplicaron transformaciones con `Albumentations`.
   - Se aprendió a normalizar y desnormalizar bounding boxes con funciones `norm()` y `unnorm()`.
   - Se respetó el formato COCO `[x_min, y_min, width, height]` al hacer augmentations.

4. **Modelado**
   - Se construyó un modelo simple de CNN (`Model`) con bloques convolucionales y lineales.
   - El modelo predice las coordenadas normalizadas de las bounding boxes.
   - Se utilizó L1Loss para la regresión de bounding boxes.

5. **Entrenamiento**
   - Se implementó la función `fit()` para entrenar el modelo con un solo ejemplo.
   - Se manejó el entrenamiento en GPU si está disponible (`device = "cuda"`).
   - Se realizaron predicciones y se visualizó la caja predicha sobre la imagen.

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
