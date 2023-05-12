'''
Funciones utiles para el procesamiento GNSS
2022.05.04
'''


import numpy as np
import math

import pyrtklib
import bancroft

from constantes import *


def ecef2neu(pos, ecef):

    '''
    convierte dx dy dz a dn de du para un array de n puntos
    recibe:
        pos         posiciones del punto de comparacion en lat (rad) lon (rad) alt [npuntos x 3]
        ecef        diferencias en x y z [npuntos x 3]
    devuelve:
        neu         diferencias en n e a [npuntos x 3]
    '''

    lat = pos[:, 0]
    lon = pos[:, 1]

    npuntos = ecef.shape[0]

    unit_n = np.zeros((npuntos, 3))
    unit_n[:, 0] = -np.sin(lat) * np.cos(lon)
    unit_n[:, 1] = -np.sin(lat) * np.sin(lon)
    unit_n[:, 2] =  np.cos(lat)

    unit_e = np.zeros((npuntos, 3))
    unit_e[:, 0] = -np.sin(lon)
    unit_e[:, 1] =  np.cos(lon)

    unit_u = np.zeros((npuntos, 3))
    unit_u[:, 0] =  np.cos(lat) * np.cos(lon)
    unit_u[:, 1] =  np.cos(lat) * np.sin(lon)
    unit_u[:, 2] =  np.sin(lat)

    norte  = np.sum(unit_n * ecef, axis=1).reshape(-1, 1)
    este   = np.sum(unit_e * ecef, axis=1).reshape(-1, 1)
    altura = np.sum(unit_u * ecef, axis=1).reshape(-1, 1)

    neu = np.hstack((norte, este, altura))

    return neu


def azimut_elevacion(r_sats, r_recs):

    '''
    funcion para determinar el azimut y elevacion (en radianes) de cada satelite respecto a cada receptor
    recibe:
        r_sats      coordenadas ecef de los satelites [nrec x neps x nsat x nobs x 3]
        r_recs      coordenadas ecef de los receptores [nrec x 3]
    devuelve:
        azel        azimut y elevacion de los satelites respecto a los receptores [nrec x neps x nsat x nobs x 2]
    '''

    # cambia el orden de las dimensiones para poder obtener los vectores satelite-receptor
    r_sats = np.transpose(r_sats, axes=[1, 2, 3, 0, 4])

    # diferencia entre coordenadas de los satelites y de los receptores
    e = r_sats - r_recs

    # norma del vector (distancia satelite-receptor)
    distancia = np.linalg.norm(e, axis=4)

    # vectores unitarios satelite-receptor en ecef
    ecef = np.transpose(np.transpose(e, axes=[4, 0, 1, 2, 3]) / distancia, axes=[4, 1, 2, 3, 0])

    nrec = r_recs.shape[0]
    neps = ecef.shape[1]
    nsat = ecef.shape[2]
    nobs = ecef.shape[3]
    ecef = ecef.reshape((nrec, -1, 3))

    # array para almacenar azimutes y elevaciones
    azel = np.zeros((nrec, neps * nsat * nobs, 2))

    # para cada receptor
    for i in range(nrec):

        # en primer lugar convierte las coordenadas de los receptores a geodesicas
        n = ecef.shape[1]
        pos = np.ones((n, 2)) * pyrtklib.ecef2pos(r_recs[i, :])[0:2]

        # convierte los vectores unitarios de ecef a neu
        neu = ecef2neu(pos, ecef[i, :, :])
        
        # calcula los azimuts
        az = np.arctan2(neu[:, 1], neu[:, 0])
        az[az < 0.0] += 2 * np.pi
        azel[i, :, 0] = az

        # calcula las elevaciones
        el = np.arcsin(neu[:, 2])
        azel[i, :, 1] = el

    # donde la elevacion es menor a 0.0 el satelite se encuentra bajo el horizonte
    mascara = (azel[:, :, 1] < 0.0)
    azel[mascara] = 0.0

    # cambia la forma
    azel = azel.reshape((nrec, neps, nsat, nobs, 2))

    return azel


def bancroft_rinexs(r_sats):

    '''
    funcion para aplicar el algoritmo de bancroft a las observaciones de los rinex
    recibe:     
        r_sats      coordenadas del satelite, error del reloj del satelite y pseudodistancia observada [nrec x neps x nsat x nobs x 5]
    devuelve:
        r_recs      coordenadas determinadas para los receptores [nrec x 3]
        deltas      desfasaje de los relojes de los receptores para cada epoca [nrec x neps]
    '''

    # aplico el algoritmo de bancroft para determinar valores aproximados para los parametros. los
    # valores aproximados los guardo en un array de dimensiones [nrec x neps x 4]. los ultimos 4
    # valores son las coordenadas del receptores y el error de sincronizacion del reloj del receptor

    nrec = r_sats.shape[0]      # numero de receptores
    neps = r_sats.shape[1]      # numero de epocas de todo el procesamiento
    nsat = r_sats.shape[2]      # numero de satelites de la constelacion gps
    nobs = r_sats.shape[3]      # numero de observables con observaciones registradas

    # array para los valores aproximados de bancroft
    resultados = np.zeros((nrec, neps, 4))
    resultados[:, :, 0:3] = np.nan

    # para cada receptor
    for i in range(nrec):

        # para cada epoca
        for j in range(neps):
            # obtengo las coordenadas de los satelites y el error de sincronizacion de sus relojes
            # que ya se determinaron al calcular los inputs. rsats tiene dimensiones [nsat x nobs x 5]
            # donde los satelites que no tienen observaciones en alguno o todos los codigos toman
            # el valor 0.0 para las coordenadas y para el desfasaje del reloj
            r_sat = r_sats[i, j, :, :, 0:4].reshape((nsat * nobs, 4))

            # obtengo las pseudodistancias medidas. en este caso las dimensiones son de
            # [nsat * nobs x 1] donde no hay observaciones registradas hay 0.0
            obs = r_sats[i, j, :, :, 4].reshape((nsat * nobs, 1))

            # concateno ambos arrays sin tener en cuenta el error del reloj
            # del satelite. dimensiones [nsat * nobs x 4]
            data = np.hstack((r_sat[:, 0:3], obs))

            # mascara para los pares de satelites y observables sin datos
            mascara = (data[:, 3] != 0.0)

            # aplica la mascara y si para esa epoca el receptor no registro ningun codigo continua
            data = data[mascara]
            if data.shape[0] < 4:
                # no hay datos
                continue
            
            # corrije el error de sincronizacion del reloj del satelite en cada pseudodistancia
            delta_sat = r_sat[:, 3][mascara]
            data[:, 3] += VLUZ * delta_sat

            # aplica bancroft
            res = bancroft.bancroft(data)

            # los resultados los reemplazo en el array de valores aproximados
            resultados[i, j, :] = res

    # como bancroft da una coordenada por epoca tomo el promedio de todas las epocas para cada receptor. dimension [nrec x 3]
    r_recs = np.nanmean(resultados, axis=1)[:, 0:3]

    deltas = resultados[:, :, 3]

    return r_recs, deltas


def prioridad_de_observables(c, observables, fases=False):

    '''
    funcion para determinar la prioridad de eleccion de observables segun los tipos del estandar RINEX3
    recibe:
        c               constelacion
        observables     lista de todos los observables disponibles
        fases           incluir o no los observables de fase
    devuelve:
        observables_    lista de los observables a utilizar para la constelacion
                        (dos codigos y dos fases en diferentes frecuencias)

    (((por ahora solo implementado para GPS y GALILEO)))
    '''

    if fases:
        tipos_de_observables = ['C', 'L']
    else:
        tipos_de_observables = ['C']
    
    if c == 'G':
        frecuencias = ['1', '2']
    elif c == 'E':
        frecuencias = ['1', '5']
    else:
        return None

    observables_ = []
    for o in tipos_de_observables:
        for f in frecuencias:
            atributos = ATRIBUTOS_OBSERVABLES[c][f].split(' ')
            for a in atributos:
                tipo = '{}{}{}'.format(o, f, a)
                try:
                    observables.index(tipo)
                    observables_.append(tipo)
                    break
                except ValueError:
                    continue

    return observables_


def datos_rinex(rinexs, fases=False):

    '''
    funcion para determinar epocas en las que hay registros, observables y constelaciones
    recibe:
        rinexs              listado de los objetos RINEX
        fases               indicador booleano indicando si se incluyen las fases o no
    devuelve:
        epocas              listado completo de epocas en las que algunos de los receptores tiene observaciones
        constelaciones      listado completo de constelaciones en las que algunos de los receptores tiene observaciones
        sats_con_datos      diccionario cuyas claves son las constelaciones donde almacenan los satelites observados
                            para la constelacion
        observables         diccionario cuyas claves son las constelaciones donde almacenan los observables observados
                            para la constelacion. por cada constelacion habra dos codigos y dos fases segun las prioridades
                            establecidas
    '''

    # obtengo todas las epocas en las que se registraron datos
    # y todas las constelaciones que tiene al menos una observacion
    epocas = []
    constelaciones = []
    for ir, rnx in enumerate(rinexs):
        if ir == 0:
            epocas += rnx.epocas
            constelaciones += rnx.gnss
        else:
            epocas = elementos_comunes(epocas, rnx.epocas)
            constelaciones = elementos_comunes(constelaciones, rnx.gnss)
    epocas = list(set(epocas))
    constelaciones = list(set(constelaciones))
    epocas.sort()
    constelaciones.sort()

    # siempre la primera constelacion tiene que ser GPS, por lo tanto intercambio el orden
    indice_gps = constelaciones.index('G')
    c0 = constelaciones[0]
    constelaciones[indice_gps] = c0
    constelaciones[0] = 'G'

    # para cada constelacion guardo todos los observables que interesan
    # y que tienen al menos una observacion en alguno de los rinex
    # ademas guardo cuales satelites tienen observaciones y cuales no
    observables = {}
    sats_con_datos = {}
    for c in constelaciones:
        tipos = []
        sats = []
        for ir, rnx in enumerate(rinexs):
            if c not in rnx.gnss:
                continue
            if ir == 0:
                tipos += rnx.observables[c]
                sats += rnx.sats_con_datos[c]
            else:
                tipos = elementos_comunes(tipos, rnx.observables[c])
                sats = elementos_comunes(sats, rnx.sats_con_datos[c])
        tipos = list(set(tipos))
        sats = list(set(sats))
        tipos.sort()
        sats.sort()

        if fases:
            # busca dos codigos y dos fases en diferentes frecuencias
            observables[c] = prioridad_de_observables(c, tipos, fases=True)
        else:
            # busca dos codigos en diferentes frecuencias
            observables[c] = prioridad_de_observables(c, tipos)

        sats_con_datos[c] = sats

    return epocas, constelaciones, sats_con_datos, observables


def n_observables(observables):

    '''
    funcion para contar cuantos observables son de codigo y cuandos de fase
    recibe:
        observables     listado de observables
    devuelve:
        ncod            cantidad de codigos
        nfas            cantidad de fases
    '''

    ncod = 0
    nfas = 0
    for obs in observables:
        if obs[0] == 'C':
            ncod += 1
        else:
            nfas += 1

    return ncod, nfas


def elementos_comunes(a, b):

    '''
    funcion para determinar elementos comunes a dos listas a y b
    recibe:
        a, b        dos listas
    devuelve:
        lista de interseccion entre a y b
    '''

    a_set = set(a)
    b_set = set(b)
 
    interseccion = (a_set & b_set)

    if interseccion:
        return list(interseccion)
    else:
        []


def redimensiona_rinex_(rinexs, sats_con_datos, observables):

    '''
    funcion limpiar el rinex dejando unicamiente los observables que se indican en el diccionario
    observables
    esta funcion tiene efectos secundarios: modifica las dimensiones de los arrays de observaciones
    del rinex agregando una dimension por cada observable de los otros rinex que el receptor no
    registro. con esto se unifican las dimensiones de los arrays de observaciones de los rinex en el
    eje del numero de observables
    [neps x nsat x nobs]
    '''

    print('Modificando dimension de los RINEX...')

    for rnx in rinexs:

        # para cada constelacion
        for c in observables.keys():
            
            # si ese rinex no tiene datos de esta constelacion
            # continua con la siguiente
            if c not in rnx.gnss:
                continue

            # creo un nuevo array para guardar las observaciones
            # de codigo de la constelacion, teniendo en cuenta que para
            # todo el procesamiento en conjunto puede haber mas observables
            # que los que registro este receptor (neps x nsat x nobs)
            neps = rnx.observaciones[c].shape[0]
            nsat = len(sats_con_datos[c])
            nobs = len(observables[c])

            observaciones = np.zeros((neps, nsat, nobs))

            for i, tipo in enumerate(observables[c]):
                # si el observable esta en ese rinex lo conserva, si no continua con el proximo
                try:
                    j = rnx.observables[c].index(tipo)
                except ValueError:
                    continue
                
                # el satelite es un indice
                for k, sat in enumerate(sats_con_datos[c]):
                    # si encuentra el satelite en el rinex lo conserva, si no sigue con el proximo
                    if sat not in rnx.sats_con_datos[c]:
                        continue

                    observaciones[:, k, i] = rnx.observaciones[c][:, sat, j]

            # guarda los nuevos observables y satelites
            rnx.observables[c] = observables[c]
            rnx.sats_con_datos[c] = sats_con_datos[c]

            # recorta el array de observaciones quedandose unicamente con los codigos
            rnx.observaciones[c] = observaciones


def cod2freq(const, num_freq, fcn=None):

    '''
    funcion para determinar la frecuencia de una observacion de un satelite GLONASS
    '''

    freq = FREQS[const][num_freq]

    if const == 'R':
        if fcn < -7 or fcn > 6:
            return -1.0
        
        if num_freq == 1:
            freq += DFRQ1_GLO * fcn
        elif num_freq == 2:
            freq += DFRQ2_GLO * fcn

    return freq


def filtro_3sigma(array):

    '''
    funcion para filtrar los elementos de un array con un filtro de 3 sigma
    recibe:
        array       un array de numpy
    devuelve:
        array       el array filtrado
    '''
    
    while True:

        mu = array.mean()
        sigma = array.std()

        m = (array > mu + 3 * sigma) | (array < mu - 3 * sigma)

        array_m = array[m]
        n = np.count_nonzero(array_m)

        if n == 0:
            break

        array = np.copy(array[~m])

    return array
