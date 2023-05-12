'''
Modelado de observaciones GNSS
2023.02.15
'''

# desactivo las advertencias de tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

import efemerides
import utiles
import pyrtklib
from constantes import *



def resolver_sesgos(arr_ambiguedades, arr_lambdas, arr_mascara):

    '''
    funcion para calcular los sesgos de hardware a partir de los valores fraccionales de las ambiguedades
    recibe:
        arr_ambiguedades        array de las ambiguedades en metros [neps x nsat x nfas]
        arr_lambdas             array de las longitudes de onda [neps x nsat x nfas]
        arr_mascara             array de la mascara de las ambiguedades [neps x nsat x nfas]
    devuelve:
        sesgos                  array de sesgos de hardware calculados [nfas]
    '''

    # pasa a ciclos, redondea al entero superior y calcula las partes fraccionales
    arr_ambiguedades_ciclos = arr_ambiguedades / arr_lambdas
    arr_ambiguedades_enteras = np.ceil(arr_ambiguedades_ciclos)
    arr_fraccion = arr_ambiguedades_enteras - arr_ambiguedades_ciclos

    # [nfas]
    sesgos = np.zeros(arr_ambiguedades.shape[2])

    # para cada frecuencia
    for nf in range(arr_ambiguedades.shape[2]):

        # solo se toman los datos de la frecuencia correspondiente
        arr_fraccion_f = arr_fraccion[:, :, nf]
        arr_mascara_f = arr_mascara[:, :, nf]

        # enmascara
        arr_fraccion_f_m = arr_fraccion_f[arr_mascara_f]

        # filtro iterativo de 3sigma
        arr_fraccion_filtrada = utiles.filtro_3sigma(arr_fraccion_f_m)

        # calcula la media y la guarda como el sesgo calculado para esa frecuencia
        mu = arr_fraccion_filtrada.mean()
        sesgos[nf] = mu

    return sesgos


def mod_mediciones(rinexs, nav):

    '''
    funcion para inicializar las mediciones, la mascara y los pesos
    recibe:
        rinexs                  listado de los rinex
        nav                     datos de navegacion (struct de rtklib)
    devuelve:
        ten_mediciones_cod      tensor de mediciones de codigo [neps x nsat x ncod]
        ten_mediciones_fas      tensor de mediciones de fase [neps x nfas x nfas]
        ten_mascara_cod         tensor de mascara de codigos [neps x nfas x ncod]
        ten_mascara_fas         tensor de mascara de fases [neps x nfas x nfas]
        ten_peso                tensor de pesos de las fases [neps x nfas x nfas]
        arr_azel                array de azimutes y elevaciones de los satelites [nrec x neps x nsat x nfas x 2]
        arr_r_movil             array de coordenadas del receptor movil obtenidas con bancroft [1 x 3]
        arr_r_sats              array de coordenadas de los satelites para las observaciones de fase [nrec x neps x nsat x nfas x 3]
        arr_zeta_recs           array de desfasajes de los relojes de los receptores obtenidos con bancroft [nrec x neps]
    '''

    # por ahora solo GPS
    c = 'G'

    epocas, constelaciones, sats_con_datos, observables = utiles.datos_rinex(rinexs, fases=True)

    nrec = len(rinexs)              # numero de receptores
    neps = len(epocas)              # numero de epocas
    nsat = len(sats_con_datos[c])   # numero de satelites
    nobs = len(observables[c])      # numero de observables

    ncod, nfas = utiles.n_observables(observables[c])

    # determina las coordenadas de los satelites
    r_sats, satelites_defectuosos = efemerides.satpos(rinexs, nav, c, epocas, sats_con_datos[c], observables[c])

    # obtiene coordenadas con bancroft para cada estacion con las mediciones de GPS
    arr_r_recs, arr_zeta_recs = utiles.bancroft_rinexs(r_sats[:, :, :, :ncod, :])

    # calculo azimut y elevacion para cada par satelite-receptor [nrec x neps x nsat x nobs x 2]
    azel = utiles.azimut_elevacion(r_sats[:, :, :, :, 0:3], arr_r_recs)

    # arma el array donde se van a guardar las observaciones
    mediciones = np.zeros((neps, nsat, nobs, nrec))

    # para cada receptor
    for irec, rnx in enumerate(rinexs):

        # para cada epoca
        for ieps, epoca in enumerate(epocas):

            # busca el indice de la epoca en el rinex
            try:
                ieps_rnx = rnx.epocas.index(epoca)
            except ValueError:
                # la epoca no esta en el rinex entonces sigue con la proxima
                continue

            # para cada satelite con datos
            for isat, sat in enumerate(sats_con_datos[c]):

                # determina si hay observaciones para el par satelite-receptor en la epoca, si no continua con el proximo
                m = np.nonzero(azel[irec, ieps, isat, :, 0])
                if m[0].size == 0:
                    continue
                
                # aplica mascara de elevacion
                el = azel[irec, ieps, isat, :, :][m][0][1] * R2D
                if el < MASCARA_ELEVACION:
                    continue

                # busca si hay observaciones validas en el rinex para esa epoca y ese satelite
                m = np.nonzero(rnx.observaciones[c][ieps_rnx, isat, :])
                if m[0].size == 0:
                    # si no hay ninguna pseudodistancia registrada para ese receptor, en esa epoca y para ese satelite continua con el proximo
                    continue

                # guarda los resultados en los arrays
                mediciones[ieps, isat, m, irec] = np.array(rnx.observaciones[c][ieps_rnx, isat, :][m])

    # determina la mascara
    mascara = np.not_equal(mediciones, 0.0)

    # simples diferencias para las mediciones y para la mascara
    mediciones_sd = mediciones[:, :, :, 1] - mediciones[:, :, :, 0]
    mascara_sd = np.logical_and(mascara[:, :, :, 1], mascara[:, :, :, 0])

    # elimina satelites defectuosos
    mascara_sd = np.logical_and(mascara_sd, satelites_defectuosos)
    
    # separa mascaras de codigos y fases
    mascara_codigos = mascara_sd[:, :, :ncod]
    mascara_fases   = mascara_sd[:, :, ncod:]

    # calcula promedios elevaciones y azimutes y calcula los pesos
    arr_azel = azel[:, :, :, ncod:, :]
    arr_promedio_azel = ((arr_azel[1, :, :, :, :] + arr_azel[0, :, :, :, :]) / 2)
    pesos = np.power(2 * np.sin(arr_promedio_azel[:, :, :, 1]), 2)
    pesos[arr_promedio_azel[:, :, :, 1] >= (30 * D2R)] = 1.0
    pesos *= (1 / (STDF ** 2))

    # convierte a tensores constantes
    mediciones_sd = tf.constant(mediciones_sd, dtype=tf.float64)
    ten_mediciones_cod = mediciones_sd[:, :, :ncod]
    ten_mediciones_fas = mediciones_sd[:, :, ncod:]
    ten_mascara_cod = tf.constant(mascara_codigos, dtype=tf.bool)
    ten_mascara_fas = tf.constant(mascara_fases, dtype=tf.bool)
    ten_peso = tf.constant(pesos, dtype=tf.float64)

    arr_r_movil = arr_r_recs[1:, :]
    arr_r_sats = r_sats[:, :, :, ncod:, :3]

    return ten_mediciones_cod, ten_mediciones_fas, ten_mascara_cod, ten_mascara_fas, ten_peso, arr_azel, arr_r_movil, arr_r_sats, arr_zeta_recs


def mod_ionosfera(arr_azel, neps, nsat, nfas, arr_mascara):

    '''
    funcion para inicializar los retrasos ionosfericos
    recibe:
        neps                    numero de epocas
        nsat                    numero de satelites
        nfas                    numero de fases
    devuelve:
        fn_ionosfera            funcion que calcula el retraso ionosferico
        var_iono                variable retraso ionosferico (en metros) [neps x nsat]
        (
            ten_factores        tensor de factores para escalar el retraso en cada frecuencia [neps x nsat x nfas]
        )
    '''

    # array con la forma de la salida que contiene los factores para multiplicar por el retraso
    #arr_factores = np.ones((neps, nsat, nfas))

    const = RE_WGS84 / (RE_WGS84 + 350000)

    prom = (arr_azel[0, :, :, :, 1] + arr_azel[1, :, :, :, 1]) / 2
    arr_factores = 1 / (np.sqrt(1 - ((const * np.cos(prom)) ** 2)))


    # planteo para el caso de dos frecuencias
    coef = (FREQS['G'][1] ** 2) / (FREQS['G'][2] ** 2)
    arr_factores[:, :, 1] *= coef

    # convierte a tensor
    ten_factores = tf.constant(arr_factores, dtype=tf.float64)

    # crea la variable de los retrasos
    var_iono = tf.Variable(np.zeros((neps, nsat)), dtype=tf.float64)

    def fn_ionosfera(var_iono, ten_factores):

        '''
        funcion que calcula el retraso ionosferico para cada fase
        recibe:
            var_iono            variable retraso ionosferico (en metros) [neps x nsat]
            ten_factores        tensor de factores para escalar el retraso en cada frecuencia [neps x nsat x nfas]
        devuelve:
            retraso             tensor de retrasos ionosfericos calculados [neps x nsat x nfas]
        '''
        
        # [neps x nsat x nfas] -> [nfas x neps x nsat]
        retraso = tf.transpose(ten_factores, perm=[2, 0, 1]) * var_iono

        # [nfas x neps x nsat] -> [neps x nsat x nfas]
        retraso = -tf.transpose(retraso, perm=[1, 2, 0])

        return retraso
    
    arr_mascara_ionosfera = np.any(arr_mascara, axis=2)
    
    def fn_regularizacion_ionosfera(var_iono):

        reg = tf.boolean_mask(var_iono, arr_mascara_ionosfera)

        reg = tf.reduce_sum(reg ** 2)

        return reg
    

    return fn_ionosfera, var_iono, (ten_factores, ), fn_regularizacion_ionosfera


def mod_ambiguedades(observables, ten_mascara, arr_ambiguedades_aproximadas):

    '''
    funcion para inicializar las ambiguedades
    recibe:
        observables                         lista de observables por constelacion
        ten_mascara                         tensor de mascara de las ambiguedades [neps x nsat x nfas]
        arr_ambiguedades_aproximadas        ambiguedades iniciales [neps x nsat x nfas]
    devuelve:
        fn_ambiguedades                     funcion del modulo de las ambiguedades
        var_ambiguedades                    variable ambiguedades (en metros) [neps x nsat x nfas]
        (
            ten_mascara_s                   tensor de mascara de las ambiguedades [neps x nsat x nfas]
            ten_mascara_n                   tensor de mascara donde no se optimizan las ambiguedades [neps x nsat x nfas]
        )
        fn_regularizacion_ambiguedades      funcion de regularizacion de las ambiguedades
        arr_lambdas                         array de longitudes de onda [neps x nsat x nfas]
    '''

    # por ahora solo GPS
    c = 'G'

    # crea un array con las longitudes de onda correspondientes a cada ambiguedad
    arr_lambdas = np.zeros(arr_ambiguedades_aproximadas.shape)
    io = 0
    for obs in observables[c]:
        if obs[0] != 'L':
            continue
        freq = FREQS[c][int(obs[1])]
        lam = VLUZ / freq
        arr_lambdas[:, :, io] = lam
        io += 1

    # inicializa la variable
    var_ambiguedades = tf.Variable(arr_ambiguedades_aproximadas, dtype=tf.float64)

    # mascara que es 1 donde SI hay que optimizar sobre la ambiguedad y 0 en las demas celdas
    arr_mascara_s = np.ones(ten_mascara.shape)
    arr_mascara_s[~ten_mascara.numpy()] = 0
    ten_mascara_s = tf.constant(arr_mascara_s, dtype=tf.float64)

    # mascara que es 1 donde NO hay que optimizar sobre la ambiguedad y 0 en las demas celdas
    arr_mascara_n = np.ones(ten_mascara.shape)
    arr_mascara_n[ ten_mascara.numpy()] = 0
    ten_mascara_n = tf.constant(arr_mascara_n, dtype=tf.float64)

    @tf.function
    def fn_ambiguedades(var_ambiguedades, ten_mascara_s, ten_mascara_n):

        '''
        funcion que calcula la ambiguedad para cada observacion
        recibe:
            var_ambiguedades        variable de las ambiguedades [neps x nsat x nfas]
            ten_mascara_s           tensor de mascara de las ambiguedades [neps x nsat x nfas]
            ten_mascara_n           tensor de mascara donde no se optimizan las ambiguedades [neps x nsat x nfas]
        devuelve:
            ambs                    salida del modulo [neps x nsat x nfas]
        '''

        ambs = -1 * (tf.stop_gradient(ten_mascara_n * var_ambiguedades) + ten_mascara_s * var_ambiguedades)
        
        return ambs


    @tf.function
    def fn_regularizacion_ambiguedades(var_ambiguedades):

        '''
        funcion para regularizar las ambiguedades
        recibe:
            var_ambiguedades        variable de las ambiguedades [neps x nsat x nfas]
        devuelve:
            regularizacion          regularizacion calculada
        '''

        ambs = tf.stop_gradient(ten_mascara_n * var_ambiguedades) + ten_mascara_s * var_ambiguedades

        # diferencias entre epocas sucesivas
        diferencias = ambs[:-1, :, :] - ambs[1:, :, :]
        mascara_diferencias = tf.logical_and(ten_mascara[:-1, :, :], ten_mascara[1:, :, :])

        diferencias_ = tf.boolean_mask(diferencias, mascara_diferencias)

        return GAMMA * tf.reduce_sum(diferencias_ ** 2)

    return fn_ambiguedades, var_ambiguedades, (ten_mascara_s, ten_mascara_n), fn_regularizacion_ambiguedades, arr_lambdas


def mod_antenas(epocas, pcvs, datos_antena, arr_azel, dimensiones, sats_con_datos):

    '''
    funcion para inicializar las correcciones de antena por calibracion y altura de antena
    recibe:
        epocas                      lista de las epocas comunes
        pcvs                        datos de los offsets de las antenas de satelites y receptores (struct de rtklib)
        datos_antena                modelos y alturas de antena
        arr_azel                    array de azimutes y elevaciones de los satelites [nrec x neps x nsat x nfas x 2]
        dimensiones                 tupla con la forma del tensor de salida
        sats_con_datos              satelites con datos
    devuelve:
        fn_correcciones_antena      funcion que devuelve las correcciones de antena 
        ten_correccion_antena_sd    tensor con las correcciones [neps x nsat x nfas]
        (
            ten_factores            tensor con la forma de la salida [neps x nsat x nfas]
        )
    '''

    # por ahora solo GPS y dos receptores
    c = 'G'
    nrec = 2

    # array para las correcciones
    arr_correccion_antena = np.zeros((*dimensiones, nrec))

    # para cada receptor
    for irec in range(nrec):

        # busco la antena para el receptor
        nombre_antena, altura_de_antena = datos_antena[irec]
        delta = np.array([0.000, 0.000, altura_de_antena])
        pcv_r = pyrtklib.searchpcv(0, nombre_antena, epocas[0], pcvs)
        
        # para cada epoca
        for ieps in range(len(epocas)):

            # para cada satelite
            for isat, sat in enumerate(sats_con_datos[c]):

                # si no hay observaciones para ese par satelite-receptor en esa epoca, continua con el proximo
                m = np.nonzero(arr_azel[irec, ieps, isat, :, 0])
                if m[0].size == 0:
                    continue

                # correcciones para la antena del receptor
                azel_sat = arr_azel[irec, ieps, isat, :, :][m][0][0:2]
                corr = pyrtklib.antmodel(pcv_r, delta, azel_sat, 0)

                arr_correccion_antena[ieps, isat, :, irec] = corr[:2]

    # simple diferencia
    arr_correccion_antena_sd = arr_correccion_antena[:, :, :, 1] - arr_correccion_antena[:, :, :, 0]
    ten_correccion_antena_sd = tf.constant(arr_correccion_antena_sd, dtype=tf.float64)

    # solo definido para seguir la logica de los demas modulos
    ten_factores = tf.ones(dimensiones, dtype=tf.float64)

    def fn_correcciones_antena(ten_correccion_antena_sd, ten_factores):

        '''
        funcion para devolver las correcciones de antena
        recibe:
            ten_correccion_antena_sd    tensor con las correcciones de antena simple diferenciadas [neps x nsat x nfas]
            ten_factores                tensor con la forma de la salida [neps x nsat x nfas]
        devuelve:
            correcciones                tensor con las correcciones de antena simple diferenciadas [neps x nsat x nfas]
        '''

        correcciones = ten_factores * ten_correccion_antena_sd

        return correcciones

    return fn_correcciones_antena, ten_correccion_antena_sd, (ten_factores, )


def mod_sesgos_receptores(dimensiones):

    '''
    funcion para inicializar los sesgos de hardware de los receptores
    recibe:
        dimensiones             tupla con la forma del tensor de salida
    devuelve:
        fn_sesgos_receptores    funcion que devuelve los sesgos de los receptores
        var_sesgos              variable para los sesgos de hardware de los receptores [nfas]
        (
            ten_factores        tensor con la forma de la salida [neps x nsat x nfas]
        )
    '''

    # crea la variable para los sesgos
    var_sesgos = tf.Variable(np.zeros((dimensiones[2])), dtype=tf.float64)

    # tensor con la forma de la salida
    ten_factores = tf.ones(dimensiones, dtype=tf.float64)

    @tf.function
    def fn_sesgos_receptores(var_sesgos, ten_factores):

        '''
        funcion que calcula los sesgos de hardware
        recibe:
            var_sesgos          variable para los sesgos de hardware de los receptores [nfas]
            ten_factores        tensor con la forma de la salida [neps x nsat x nfas]
        devuelve:
            sesgos              simples diferencias de los sesgos de los relojes [neps x nsat x nfas]
        '''

        sesgos = ten_factores * var_sesgos

        return sesgos

    return fn_sesgos_receptores, var_sesgos, (ten_factores, )


def mod_relojes_receptores(arr_zeta_recs, dimensiones):

    '''
    funcion para inicializar los desfasajes de los relojes de los receptores
    recibe:
        arr_zeta_recs           array de desfasajes de los relojes de los receptores obtenidos con bancroft [nrec x neps]
        dimensiones             tupla con la forma del tensor de salida
    devuelve:
        fn_reloj_receptor       funcion que calcula el desfasaje de los relojes
        var_zeta_recs           variable de desfasajes de los relojes de los receptores [neps]
        (
            ten_factores        tensor con la forma de la salida [neps x nsat x nfas]
        )
    '''

    # simples diferencias
    arr_zeta_recs_sd = arr_zeta_recs[1, :] - arr_zeta_recs[0, :]

    # variable para los offset del reloj del receptor. dimension [neps]
    var_zeta_recs = tf.Variable(arr_zeta_recs_sd, dtype=tf.float64)

    # tensor con la forma de la salida
    ten_factores = tf.ones(dimensiones, dtype=tf.float64)

    @tf.function
    def fn_reloj_receptor(var_zeta_recs, ten_factores):

        '''
        funcion que calcula el desfasaje de los relojes de los receptores
        recibe:
            var_zeta_recs           variable de desfasajes de los relojes de los receptores [neps]
            ten_factores            tensor con la forma de la salida [neps x nsat x nfas]
        devuelve:
            var_zeta_recs_          simples diferencias de los desfasajes de los relojes [neps x nsat x nfas] 
        '''

        # [neps x nsat x nfas] -> [nsat x nfas x neps]
        f = tf.transpose(ten_factores, perm=[1, 2, 0])

        # [nsat x nfas x neps] -> [neps x nsat x nfas]
        var_zeta_recs_ = tf.transpose(f * var_zeta_recs, perm=[2, 0, 1])

        return var_zeta_recs_


    return fn_reloj_receptor, var_zeta_recs, (ten_factores, )



def mod_distancia(arr_r_base, arr_r_movil, arr_r_sats):

    '''
    funcion para inicializar las distancias satelite-receptor para cada par
    recibe:
        arr_r_base          array de coordenadas del receptor base  [                        1 x 3]
        arr_r_movil         array de coordenadas del receptor movil [                        1 x 3] 
        arr_r_sats          array de coordenadas de los satelites   [nrec x neps x nsat x nfas x 3]
    devuelve:
        fn_distancia        funcion que calcula las simples diferencias de las distancias entre satelites y receptores
        var_r_movil         variable de coordenadas del receptor movil [1 x 3] 
        (
            ten_r_base          tensor de coordenadas del receptor base [1 x 3] 
            ten_r_sats          tensor de coordenadas de los satelites [nrec x neps x nsat x nfas x 3]
        )
    '''

    # convierte a tensor y variable
    ten_r_base  = tf.constant(arr_r_base, dtype=tf.float64)
    var_r_movil = tf.Variable(arr_r_movil, dtype=tf.float64)
    ten_r_sats  = tf.constant(arr_r_sats, dtype=tf.float64)

    @tf.function
    def fn_distancia(var_r_movil, ten_r_base, ten_r_sats):

        '''
        funcion que calcula la distancia entre satelite y receptor en funcion de las coordenadas de ambos
        recibe:
            var_r_movil         variable de coordenadas del receptor movil [                        1 x 3] 
            ten_r_base          tensor de coordenadas del receptor base    [                        1 x 3]
            ten_r_sats          tensor de coordenadas de los satelites     [nrec x neps x nsat x nfas x 3]
        devuelve:
            r_sd                simples diferencias de distancias satelite-receptor para cada par [neps x nsat x nfas] 
        '''

        # concatena las coordenadas de la base con la variable para las coordenadas del movil
        r_recs = tf.concat([ten_r_base, var_r_movil], axis=0)

        # cambia la forma a [neps x nsat x nfas x nrec x 3]
        ten_r_sats_ = tf.transpose(ten_r_sats, perm=[1, 2, 3, 0, 4])

        # hace las diferencias entre coordenadas de los satelites y de los receptores 
        e = ten_r_sats_ - r_recs

        # norma del vector (distancia satelite-receptor)
        d = tf.norm(e, axis=4)

        # correccion de Sagnac por rotacion de la tierra durante el viaje de la senal. OMGE * (x_s * y_r - y_s * x_r) / VLUZ
        sagnac = OMGE * (ten_r_sats_[:, :, :, :, 0] * r_recs[:, 1] - ten_r_sats_[:, :, :, :, 1] * r_recs[:, 0]) / VLUZ
        
        # [neps x nsat x nfas x nrec]
        r = d + sagnac

        # simple diferencia [neps x nsat x nfas]
        r_sd = r[:, :, :, 1] - r[:, :, :, 0]

        return r_sd


    return fn_distancia, var_r_movil, (ten_r_base, ten_r_sats)

