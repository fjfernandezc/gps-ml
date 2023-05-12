'''
Parser de RINEX3
2023.02.15
'''

# desactivo las advertencias de tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from constantes import *
from utiles import cod2freq


class RINEX():

    '''
    clase para leer y manipular las observaciones de un
    archivo RINEX en version 3.xx
    '''

    def __init__(self, nombre_rinex):

        self.nombre = nombre_rinex
        
        # si el archivo no existe, termina el proceso
        if not os.path.isfile(nombre_rinex):
            print('No existe el RINEX {}'.format(nombre_rinex))
            return

        # abre el RINEX y lee el contenido del archivo
        with open(nombre_rinex, 'r') as f:
            contenido = f.read()
        
        # separa el contenido en encabezado y observaciones partiendolo
        # donde se especifica que termina el encabezado (END OF HEADER)
        encabezado, eoh, observaciones = contenido.partition('END OF HEADER')
        
        # si no se encuentra la cadena END OF FILE, terimna el proceso
        if not eoh:
            print('El archivo no parece ser un RINEX{}'.format(nombre_rinex))
            return

        self.gnss = []              # lista para las constelaciones gnss
        self.observables = {}       # diccionario para los tipos de observables
        self.epocas = []            # lista para almacenar las epocas
        self.observaciones = {}     # diccionario para almacenar las observaciones
        self.sats_con_datos = {}    # diccionario para almacenar los satelites con datos

        # si puede parsear el encabezado
        if self.parsea_encabezado(encabezado):
            # lee las observaciones
            self.parsea_observaciones(observaciones)

            # remueve los satelites sbas
            if 'S' in self.gnss:
                self.gnss.remove('S')
                self.observables.pop('S')
                self.observaciones.pop('S')
                self.sats_con_datos.pop('S')
            
            # remueve los satelites glonass
            if 'R' in self.gnss:
                self.gnss.remove('R')
                self.observables.pop('R')
                self.observaciones.pop('R')
                self.sats_con_datos.pop('R')
            
            # remueve los satelites beidou
            if 'C' in self.gnss:
                self.gnss.remove('C')
                self.observables.pop('C')
                self.observaciones.pop('C')
                self.sats_con_datos.pop('C')
            
            # remueve los satelites galileo
            if 'E' in self.gnss:
                self.gnss.remove('E')
                self.observables.pop('E')
                self.observaciones.pop('E')
                self.sats_con_datos.pop('E')

    
    def parsea_encabezado(self, encabezado):

        '''
        funcion para parsear los metadatos del rinex. obtiene las coordenadas
        aproximadas, tipos de observables y las constelaciones registradas para
        guardarlas en atributos de la clase
        '''        

        # en primer lugar se obtienen las coordenadas aproximadas
        patron = r'^.+APPROX POSITION XYZ'
        resultados = re.findall(patron, encabezado, flags=re.MULTILINE)
        # si no se encontraron coordenadas aproximadas termina la lectura del rinex
        if not resultados:
            print('Se necesitan coordenadas aproximadas en el encabezado del RINEX')
            return False
        x = float(resultados[0][0 :14])
        y = float(resultados[0][14:28])
        z = float(resultados[0][28:42])
        self.crd = np.array([x, y, z])  # coordenadas aproximadas ecef

        # luego se parsean los tipos de observables y las constelaciones gnss
        patron = r'^[A-Z0-9 ]+SYS / # / OBS TYPES'
        resultados = re.findall(patron, encabezado, flags=re.MULTILINE)
        # si no se encontraron tipos de observables termina la lectura del rinex
        if not resultados:
            return False

        i = 0
        # mientras haya nuevas lineas de etiqueta SYS / # / OBS TYPES
        while i < len(resultados):
            linea = resultados[i]
            # lee de que constelacion son los observables de la linea actual
            # y la agrega a la lista de constelaciones
            constelacion = linea[0]
            self.gnss.append(constelacion)

            # lee todos los observables de dicha constelacion teniendo en cuenta
            # que cada linea puede tener un maximo de 13 observables
            n = int(linea[4:6])     # cantidad de observables
            k = 7                   
            tipos = []
            for _ in range(n):
                if k > 58:
                    # pasa a la siguiente linea
                    i += 1
                    k = 7
                    linea = resultados[i]
                tipos.append(linea[k:k+3])
                k += 4
            # cuando termina de leer todos los observables de la constelacion
            # actual los agrega al diccionario y continua con la proxima constelacion
            self.observables[constelacion] = tipos
            i += 1
        
        self.gnss.sort()

        # finalmente lee los slot numbers de los satelites GLONASS y completa el diccionario de fcn
        patron = r'^[A-Z0-9 \-]+GLONASS SLOT / FRQ #'
        resultados = re.findall(patron, encabezado, flags=re.MULTILINE)
        if not resultados:
            return True

        self.fcn = {}
        i = 0
        linea = resultados[i]
        n = int(linea[0:3])     # cantidad de satelites
        k = 4
        for _ in range(n):
            if k > 53:
                # pasa a la siguiente linea
                i += 1
                k = 4
                linea = resultados[i]
            sat = int(linea[k+1:k+3])
            slot = int(linea[k+3:k+7])
            self.fcn[sat] = slot
            k += 7
        
        return True


    def parsea_observaciones(self, observaciones):

        '''
        funcion para leer las diferentes epocas almacenadas en el rinex.
        lee las epocas y las observaciones de cada satelite y las almacena
        en arrays de numpy de dimension (epocas x satelites x num. de observables)
        que se guardan en atributos de la clase. la funcion almacena todos los
        observables sin descartar datos
        '''
        
        # para cada constelacion crea un array donde se guardaran las observaciones
        for constelacion in self.gnss:
            n = MAX_PRNS[constelacion]
            m = len(self.observables[constelacion])
            self.observaciones[constelacion] = np.zeros((25000, n , m))
            self.sats_con_datos[constelacion] = []
        
        # separa cada linea del rinex en una lista
        observaciones = observaciones.split('\n')[1:]
        i = 0       # contador de lineas
        k = 0       # contador de epocas
        while True:
            # mientras haya lineas por leer
            if i == (len(observaciones) - 1):
                break
            linea = observaciones[i]

            # si no hay una nueva epoca continua con la siguiente linea
            if linea[0] != '>':
                i += 1
                continue

            # parsea el encabezado de la epoca
            try:
                epoca = datetime(int(linea[2:6]), int(linea[7:9]), int(linea[10:12]), int(linea[13:15]), int(linea[16:18]), int(linea[19:21]))
            except ValueError:
                # si no puede parsear la epoca continua con la siguiente linea
                i += 1
                continue
            
            # con esto se fija el intervalo de registro a 30 segundos
            if epoca.second not in [0, 30]:
                i += 1
                continue
            
            self.epocas.append(epoca)
            ns = int(linea[33:35])      # cantidad de satelites en la epoca

            # parsea la linea de cada satelite para esa epoca
            for _ in range(ns):
                i += 1
                linea = observaciones[i]
                constelacion = linea[0]     # constelacion del satelite
                sat = int(linea[1:3])       # numero del satelite

                j = 3
                n = 0
                while j < len(linea):
                    # parsea cada observable de la linea y lo guarda
                    try:
                        obs = float(linea[j:j+14])

                        # paso a metros las fases
                        if self.observables[constelacion][n][0] == 'L':
                            num_freq = int(self.observables[constelacion][n][1])
                            if constelacion == 'R':
                                try:
                                    fcn = self.fcn[sat]
                                except:
                                    fcn = 0
                            else:
                                fcn = None
                            freq = cod2freq(constelacion, num_freq, fcn)
                            longitud_onda = VLUZ / freq
                            self.observaciones[constelacion][k, sat - 1, n] = obs * longitud_onda
                        else:
                            self.observaciones[constelacion][k, sat - 1, n] = obs
                            
                        self.sats_con_datos[constelacion].append(sat - 1)
                    except ValueError:
                        # si no hay valor continua con el siguiente observable
                        pass
                    j += 16
                    n += 1
            k += 1      
            i += 1

        # para cada constelacion
        for constelacion in self.gnss:

            # recorta el array de observaciones a la cantidad de epocas que realmente hay
            self.observaciones[constelacion] = np.copy(self.observaciones[constelacion][:len(self.epocas), :, :])

            # obtiene los satelites que se observaron
            self.sats_con_datos[constelacion] = list(set(self.sats_con_datos[constelacion]))



if __name__ == '__main__':

    # prueba para sacar algunas estadisticas de un rinex
    print('Resumen CORD00ARG_R_20220470000_01D_30S_MO.rnx\n')

    # lee el rinex
    rnx = RINEX('datos/obs/CORD00ARG_R_20220470000_01D_30S_MO.rnx')

    # para cada constelacion
    for c in rnx.gnss:

        # imprime el encabezado con los observables de la constelacion
        print('    ', end='')
        for ot in rnx.observables[c]:
            print('{:>4s} '.format(ot), end='')
        print('')

        # para cada satelite con datos
        for sat in rnx.sats_con_datos[c]:
            print('{}{:02d} '.format(c, sat + 1), end='')
            # para cada tipo de observable
            for ot in rnx.observables[c]:
                # cuenta la cantidad de observaciones y los imprime
                observaciones = tf.math.count_nonzero(rnx.observaciones[c][:, sat, rnx.observables[c].index(ot)])
                print('{:>4d} '.format(observaciones), end='')
            print('')


        print('-----------------')
