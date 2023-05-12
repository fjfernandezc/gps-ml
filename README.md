# Procesamiento de observaciones GNSS de fase en un entorno de machine learning

Esta herramienta escrita en Python permite procesar vectores GNSS utilizando un modelo de simples diferencias de fase utilizando técnicas de machine learning. Actualmente está implementada para el procesamiento de vectores cortos, ya que no se consideran los retrasos atmosféricos en el modelo, y para observaciones de la constelación GPS. Se utiliza la librería TensorFlow y funciones de la librería de procesamiento GNSS [RTKlib](https://github.com/tomojitakasu/RTKLIB).

La versión actual es capaz de:

- Parsear archivos RINEX versión 3, archivos de navegación en formato RINEX 2/3 y archivos de calibración de antenas en formato ANTEX.
- Procesar observaciones de fase L1/L2 de la constelación GPS para estimar las coordenadas de un receptor en modo estático.
- Fijar las ambigüedades de fase de simples diferencias.
- Exportar los resultados a un archivo JSON.


# Uso

Para ejecutar un procesamiento se deben ejecutar el comando:

    python3 posicionamiento_relativo.py -config <archivo de configuración>

El archivo de configuración se estructura de la siguiente manera:

    [rinex-a]
    nombre = <ruta del RINEX del receptor base>
    antena_igs = <código IGS de la antena del receptor base>
    altura_de_antena = <altura vertical de la antena del receptor base>
    x = <coordenada ECEF X del punto base>
    y = <coordenada ECEF Y del punto base>
    z = <coordenada ECEF Z del punto base>
    
    [rinex-b]
    nombre = <ruta del RINEX del receptor móvil>
    antena_igs = <código IGS de la antena del receptor móvil>
    altura_de_antena = <altura vertical de la antena del receptor móvil>
    x = <coordenada ECEF X del punto móvil (solo si se quiere comparar contra una coordenada conocida)>
    y = <coordenada ECEF Y del punto móvil (solo si se quiere comparar contra una coordenada conocida)>
    z = <coordenada ECEF Z del punto móvil (solo si se quiere comparar contra una coordenada conocida)>
    
    [productos]
    rinex-navegacion = <ruta del RINEX de navegación>
    atx = <ruta del archivo ANTEX de calibración de antenas>
    
    [resultados]
    ruta = <ruta donde se guardará el JSON con los resultados>
    prefijo = <nombre del archivo de salida>


# Dependencias

- Python3 (testeado en Python 3.9.2)
- NumPy (testeado en Numpy 1.24.2)
- TensorFlow (testeado en TensorFlow 2.11.0)

## Compilación de la librería RTKLib

Para compilar la librería RTKLib se deben seguir los siguientes pasos:

    cd lib/build
    make
    make install

En la carpeta `lib/rtklib/linux` debe encontrarse el archivo compilado `librtk.so` que contendrá las funciones de la librería.

Se utilizaron rutinas obtenidas de RTKLib, liberadas bajo una licencia BSD-2:
[https://github.com/tomojitakasu/RTKLIB](https://github.com/tomojitakasu/RTKLIB)
[https://github.com/tomojitakasu/PocketSDR](https://github.com/tomojitakasu/PocketSDR)

