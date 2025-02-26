# Monitoreo de caso real con TensorFlow Extended

El proceso de integración comenzó con el desarrollo de un modelo de predicción de precios de casas utilizando Python. Este modelo fue diseñado para prever valores futuros de propiedades inmobiliarias en función de datos históricos y características relevantes de las viviendas. Una vez que el modelo fue entrenado y probado, se decidió implementar la solución en un entorno de contenedores usando Docker para facilitar su despliegue y ejecución de manera eficiente y escalable.
Se siguieron los siguientes pasos:

## 1. Preparación del Entorno Docker
Para iniciar el proceso de contenedorización, se crearon los archivos necesarios (Dockerfile y otros scripts relacionados) que definen las dependencias, bibliotecas y configuraciones necesarias para ejecutar el modelo de predicción en un contenedor. Docker se utilizó para asegurar que el entorno fuera replicable, portátil y libre de conflictos, lo cual facilita tanto la implementación como la escalabilidad.

## 2. Cargar el Modelo a Docker
Los archivos Python (.py) que contienen el modelo de predicción fueron integrados dentro de la imagen de Docker. Esta imagen fue luego construida y desplegada en el contenedor. El contenedor, a su vez, ejecutó el modelo en el ambiente aislado, garantizando que las dependencias y el entorno fueran consistentes durante la ejecución.

## 3. Exposición de Métricas
Una vez que el modelo estaba funcionando dentro del contenedor, se configuró un sistema para exponer métricas de rendimiento y resultados generados por el modelo, como los precios de las predicciones, el error de las predicciones, y otros indicadores relevantes. Estas métricas fueron expuestas a través de un endpoint o API que puede ser consumido por herramientas de monitoreo, como Grafana.

## 4. Integración con Grafana
Con las métricas expuestas por el modelo, se integró Grafana como plataforma de visualización de datos. Grafana se conectó al endpoint de métricas, recopilando y mostrando visualizaciones en tiempo real. Las métricas relacionadas con el modelo de predicción de precios de casas, tales como los precios predichos, la precisión, y las variaciones de precio, fueron representadas mediante gráficos interactivos y paneles visuales.

## 5. Monitoreo Continuo y Seguimiento de Rendimiento
A través de Grafana, se configuraron paneles de monitoreo que permiten un seguimiento en tiempo real del rendimiento del modelo. Esto incluye análisis sobre la precisión de las predicciones, la variación de precios en el tiempo y otros indicadores clave que son fundamentales para evaluar la eficacia del modelo. Estas métricas proporcionan información valiosa para ajustar y mejorar el modelo de manera continua.

## 6. Optimización y Escalabilidad
El uso de Docker facilita la escalabilidad del modelo, permitiendo que múltiples instancias del contenedor se ejecuten simultáneamente, mejorando la capacidad de procesamiento y respuesta. Grafana, por su parte, puede manejar grandes volúmenes de datos y proporcionar métricas en tiempo real, lo que permite una gestión más eficaz de los recursos y un ajuste rápido en función de las métricas obtenidas.
