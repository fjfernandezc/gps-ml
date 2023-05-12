'''
Procesamiento de observaciones GNSS de fase en un entorno de machine learning
2023.02.15
'''

# desactivo las advertencias de tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import argparse
import datetime
import configparser
import json

from constantes import *
import rinex
import pyrtklib
import gnss
import utiles
import lbfgs

import warnings
warnings.filterwarnings("ignore")


def guarda_resultados(datos_resultados, epocas, constelaciones, sats_con_datos, observables, ten_residuos, ten_mascara, arr_azel, arr_lambdas, r_base, parametros, n_fijas, n_observaciones, ecms):

    '''
    Funcion que guarda los resultados del procesamiento en un archivo de formato JSON

    recibe:
        datos_resultados        ruta donde se debe guardar el archivo JSON
        epocas                  listado de epocas
        constelaciones          listado de constelaciones
        sats_con_datos          listado de satelites con datos por constelacion
        observables             listado de observables por constelacion
        ten_residuos            tensor de residuos [neps x nsat x nfas]
        ten_mascara             tensor mascara [neps x nsat x nfas]
        arr_azel                array de azimutes y elevaciones [nrec x neps x nsat x nfas x 2]
        arr_lambdas             array de longitudes de onda [neps x nsat x nfas]
        r_base                  array de coordenadas de la base
        parametros              listado de parametros
        n_fijas                 numero de ambiguedades fijas
        n_observaciones         numero de observaciones originales y numero de observaciones utilizadas
        ecms                    desvio estandar y desvio estandar normalizado de las fases
    '''

    print('Guardando resultados del entrenamiento...')

    contenido = {}

    contenido['coordenadas_base'] = list(r_base[0, :])

    r_movil = parametros[0].numpy()
    contenido['coordenadas_movil'] = list(r_movil[0, :])

    contenido['nfijas'] = n_fijas
    contenido['nobs'] = n_observaciones
    contenido['ecms'] = ecms

    sesgos = parametros[2].numpy()
    contenido['sesgos'] = list(sesgos)

    contenido['constelaciones'] = constelaciones

    arr_mascara = ten_mascara.numpy()
    arr_residuos = ten_residuos.numpy()
    arr_azel_prom = ((arr_azel[1, :, :, :, :] + arr_azel[0, :, :, :, :]) / 2)       # promedio de la simple diferencia
    arr_ambiguedades_ciclos = parametros[-1].numpy() / arr_lambdas

    for c in constelaciones:

        contenido_constelacion = {}
        
        for isat, sat in enumerate(sats_con_datos[c]):
            satid = '{}{:02d}'.format(c, sat+1)
            contenido_constelacion[satid] = {}

            io = 0
            for ot in observables[c]:

                if ot[0] != 'L':
                    continue

                m = arr_mascara[:, isat, io]

                if np.count_nonzero(m) == 0:
                    continue
                
                eps_m = np.array(epocas)[m]
                res_m = arr_residuos[:, isat, io][m]
                az_m = arr_azel_prom[:, isat, io, 0][m] * R2D
                el_m = arr_azel_prom[:, isat, io, 1][m] * R2D   
                amb_m = arr_ambiguedades_ciclos[:, isat, io][m]             

                contenido_constelacion[satid][ot] = {}
                contenido_constelacion[satid][ot]['epocas'] = list(eps_m)
                contenido_constelacion[satid][ot]['res'] = list(res_m)
                contenido_constelacion[satid][ot]['azim'] = list(az_m)
                contenido_constelacion[satid][ot]['elev'] = list(el_m)
                contenido_constelacion[satid][ot]['ambs'] = list(amb_m)

                io += 1

        contenido[c] = contenido_constelacion

    nombre_resultados = datos_resultados[0] + '/' + datos_resultados[1] + '.json'
    if not os.path.exists(datos_resultados[0]):
        os.makedirs(datos_resultados[0])

    objeto_json = json.dumps(contenido, indent=4, default=str)
    with open(nombre_resultados, 'w') as f:
        f.write(objeto_json)
    


@tf.function
def costo_minimos_cuadrados(modelo, mediciones, mascara, peso):

    '''
    funcion para calcular el costo de minimos cuadrados

    recibe:
        modelo          tensor del modelo
        mediciones      tensor de las mediciones
        mascara         tensor de la mascara
        peso            tensor de pesos
    devuelve:
        costo           costo calculado

    observacion: los tensores deben tener la misma forma
    '''

    # calculo de los residuos
    residuos = mediciones - modelo

    # aplica la mascara
    residuos = tf.boolean_mask(residuos, mascara)
    peso     = tf.boolean_mask(peso, mascara)

    # P x v^2
    costo = tf.reduce_sum(peso * tf.square(residuos))

    return costo


@tf.function
def modelo_gnss(modulos, info_conocida):

    '''
    funcion para calcular el modelo a partir de la suma de todos los modulos
    recibe:
        modulos             lista de tuplas, donde cada tupla contiene la funcion del modulo y el parametro correspondiente al mismo
        info_conocida       lista que contiene la informacion conocida necesaria para calcular la salida de cada modulo
    devuelve:
        modelo              tensor con el modelo computado [neps x nsat x nfas]
    '''

    # suma de los modulos
    i = 0
    modelo = None
    for modulo, parametro in modulos:
        if modelo is not None:
            modelo += modulo(parametro, *info_conocida[i])
        else:
            modelo = modulo(parametro, *info_conocida[i])
        i += 1

    return modelo


@tf.function
def regularizacion(regularizaciones):

    '''
    funcion para calcular los terminos de regularizacion y sumarlos
    recibe:
        regularizaciones        lista de tuplas, donde cada tupla contiene una funcion de regularizacion y el parametro correspondiente a la misma
    devuelve:
        regs                    terminos de regularizacion sumados
    '''

    # si no hay regularizacion devuelve 0
    if len(regularizaciones) == 0:
        regs = tf.constant(0.0, dtype=tf.float64)
        return regs

    # suma de las regularizaciones
    regs = None
    for reg, parametro in regularizaciones:
        if regs is not None:
            regs = regs + reg(parametro)
        else:
            regs = reg(parametro)

    return regs


@tf.function
def costo_y_gradientes(modulos, parametros, regularizaciones, info_conocida, ten_mediciones, ten_mascara, ten_peso):

    '''
    funcion para calcular el costo del modelo y los gradientes del costo respecto a los parametros
    recibe:
        modulos                 lista de modulos
        parametros              lista de parametros
        regularizaciones        lista de funciones de regularizacion
        info_conocida           lista de datos conocidos
        ten_mediciones          tensor de mediciones de fase [neps x nsat x nfas]
        ten_mascara             tensor de mascara de las observaciones de fase [neps x nsat x nfas]
        ten_peso                tensor de pesos de las observaciones de fase [neps x nsat x nfas]
    devuelve:
        costo_total             costo del modelo
        gradientes              lista de gradientes del costo respecto a los parametros
    '''

    with tf.GradientTape() as g:

        # calcula el modelo para fases
        ten_modelo = modelo_gnss(modulos, info_conocida)

        # calcula la funcion de costos
        costo_mmcc           = costo_minimos_cuadrados(ten_modelo, ten_mediciones, ten_mascara, ten_peso)
        costo_regularizacion = regularizacion(regularizaciones)
        costo_total          = costo_mmcc + costo_regularizacion
    
    # calcula los gradientes del costo respecto a los parametros
    gradientes = g.gradient(costo_total, parametros)
        
    return costo_total, gradientes


def inicializacion(rinexs, nav, pcvs, arr_r_base, datos_antena):

    '''
    funcion para inicializar los modulos y los parametros
    recibe:
        rinexs                  listado de rinex
        nav                     datos de navegacion (struct de rtklib)
        pcvs                    datos de los offsets de las antenas de satelites y receptores (struct de rtklib)
        arr_r_base              array de coordenadas de la base [1 x 3]
        datos_antena            modelos y alturas de antena
    devuelve:
        modulos                 lista de modulos
        parametros              lista de parametros
        regularizaciones        lista de funciones de regularizacion
        info_conocida           lista de datos conocidos
        ten_mediciones          tensor de mediciones de fase [neps x nsat x nfas]
        ten_mascara_fas         tensor de mascara de las observaciones de fase [neps x nsat x nfas]
        ten_peso                tensor de pesos de las observaciones de fase [neps x nsat x nfas]
        arr_azel                array de azimutes y elevaciones de los satelites [nrec x neps x nsat x nfas x 2]
        arr_lambdas             array de longitudes de onda [neps x nsat x nfas]
    '''

    # obtengo epocas comunes de observacion, constelaciones y observables de codigo y fase
    epocas, constelaciones, sats_con_datos, observables = utiles.datos_rinex(rinexs, fases=True)

    modulos = []            # lista donde se guardan las funciones y parametros de los modulos
    parametros = []         # lista donde se guardan los parametros sobre los que se optimizara
    regularizaciones = []   # lista donde se guardan las funciones de regularizacion y parametros de los modulos
    info_conocida = []      # lista donde se guarda la informacion conocida necesaria para el procesamiento


    print('Inicializando mediciones...')
    ten_mediciones_cod, ten_mediciones_fas, ten_mascara_cod, ten_mascara_fas, ten_peso, arr_azel, arr_r_movil, arr_r_sats, arr_zeta_recs = gnss.mod_mediciones(rinexs, nav)


    print('Inicializando distancias satelite-receptor...')
    fn, a, b = gnss.mod_distancia(arr_r_base, arr_r_movil, arr_r_sats)
    modulos.append((fn, a))
    parametros.append(a)
    info_conocida.append(b)


    print('Inicializando sincronizacion de los relojes de los receptores...')
    fn, a, b = gnss.mod_relojes_receptores(arr_zeta_recs, ten_mediciones_fas.shape)
    modulos.append((fn, a))
    parametros.append(a)
    info_conocida.append(b)


    # en la primera iteracion no optimiza sobre este parametro, por lo que no se agrega a la lista parametros
    print('Inicializando sesgos de los receptores...')
    fn, a, b = gnss.mod_sesgos_receptores(ten_mediciones_fas.shape)
    modulos.append((fn, a))
    info_conocida.append(b)


    print('Inicializando correcciones de calibracion de antenas...')
    fn, a, b = gnss.mod_antenas(epocas, pcvs, datos_antena, arr_azel, ten_mediciones_fas.shape, sats_con_datos)
    modulos.append((fn, a))
    info_conocida.append(b)


    # print('Inicializando modelo de la ionosfera...')
    # fn, a, b, fn_regularizacion_ionosfera = gnss.mod_ionosfera(arr_azel, *ten_mediciones_fas.shape, ten_mascara_fas.numpy())
    # modulos.append((fn, a))
    # parametros.append(a)
    # info_conocida.append(b)
    # regularizaciones.append((fn_regularizacion_ionosfera, a))


    # calcula valores iniciales de las ambiguedades como codigo menos fase
    arr_ambiguedades_iniciales = (tf.repeat(ten_mediciones_cod[:, :, :1], 2, axis=2) - ten_mediciones_fas).numpy()
    arr_ambiguedades_iniciales[~ten_mascara_fas.numpy()] = 0.0

    print('Inicializando ambiguedades...')
    fn, a, b, fn_regularizacion_ambiguedades, arr_lambdas = gnss.mod_ambiguedades(observables, ten_mascara_fas, arr_ambiguedades_iniciales)
    modulos.append((fn, a))
    parametros.append(a)
    info_conocida.append(b)
    regularizaciones.append((fn_regularizacion_ambiguedades, a))


    return modulos, parametros, regularizaciones, info_conocida, ten_mediciones_fas, ten_mascara_fas, ten_peso, arr_azel, arr_lambdas


def lectura_de_datos(archivo_cfg):

    '''
    funcion para parsear el archivo de configuracion que contiene la informacion necesaria para realizar el procesamiento
    recibe:
        archivo_cfg                 nombre del archivo de configuracion
    devuelve:
        rinexs                      listado de los rinexs (clase RINEX). el receptor 0 es la base
        nav                         datos de las orbitas y correcciones de relojes (struct de rtklib)
        pcvs                        datos de los offsets de las antenas de satelites y receptores (struct de rtklib)
        coordenadas_comparacion     coordenadas para comparar los resultados de la optimizacion [nrec x 3]
                                    las coordenadas del receptor 0 son las de la base
        datos_antena                modelos y alturas de antena
        datos_resultados            ruta y prefijo para los archivos de los resultados
    '''

    print('============================= Lectura de los RINEX y productos =============================')

    rinexs = []                     # para guardar los rinex
    coordenadas_comparacion = []    # para guardar las coordenadas para comparar
    datos_antena = []               # para guardar los datos de las antenas

    config = configparser.ConfigParser()
    config.read(archivo_cfg)

    for seccion in config.sections():

        if seccion == 'productos' or seccion == 'resultados':
            continue

        ruta_rinex = config[seccion]['nombre']
        coordenadas_ecef = [float(config[seccion]['x']), float(config[seccion]['y']), float(config[seccion]['z'])]

        nombre_antena = config[seccion]['antena_igs']
        altura_de_antena = float(config[seccion]['altura_de_antena'])

        # parsea el rinex y guarda el objeto en la lista
        print('Leyendo {}'.format(ruta_rinex))
        rinexs.append(rinex.RINEX(ruta_rinex))
        coordenadas_comparacion.append(coordenadas_ecef)
        datos_antena.append((nombre_antena, altura_de_antena))

    coordenadas_comparacion = np.array(coordenadas_comparacion)

    # obtiene epocas comunes de observacion
    epocas = utiles.datos_rinex(rinexs, fases=True)[0]

    # lee el rinex de navegacion gps
    nav = config['productos']['rinex-navegacion']
    print('Leyendo {}'.format(nav))
    _, nav = pyrtklib.readrnx(nav)
    
    # lee el atx
    atx = config['productos']['atx']
    print('Leyendo {}'.format(atx))
    pcvs = pyrtklib.readpcv(atx, nav, epocas[0])

    datos_resultados = (config['resultados']['ruta'], config['resultados']['prefijo'])

    return rinexs, nav, pcvs, coordenadas_comparacion, datos_antena, datos_resultados


def posicionamiento_relativo(rinexs, nav, pcvs, coordenadas_comparacion, datos_antena, datos_resultados):

    '''
    funcion que ejecuta el procesamiento
    recibe:
        rinexs                      listado de los rinexs (clase RINEX). el receptor 0 es la base
        nav                         datos de las orbitas y correcciones de relojes (struct de rtklib)
        pcvs                        datos de los offsets de las antenas de satelites y receptores (struct de rtklib)
        coordenadas_comparacion     coordenadas para comparar los resultados de la optimizacion [nrec x 3]
                                    las coordenadas del receptor 0 son las de la base
        datos_antena                modelos y alturas de antena
        datos_resultados            ruta y prefijo para los archivos de los resultados
    '''

    print('========================== Iniciando el posicionamiento preciso ============================')

    # obtiene epocas comunes de observacion, constelaciones y observables
    epocas, constelaciones, sats_con_datos, observables = utiles.datos_rinex(rinexs, fases=True)

    # modifica las dimensiones de los rinex para unificar la dimension de los observables
    utiles.redimensiona_rinex_(rinexs, sats_con_datos, observables)

    # inicializacion de modulos
    r_base = coordenadas_comparacion[:1, :]
    modulos, parametros, regularizaciones, info_conocida, ten_mediciones, ten_mascara, ten_peso, arr_azel, arr_lambdas = inicializacion(rinexs, nav, pcvs, r_base, datos_antena)

    # ------------------------------------------------------------------------------------
    # imprime algunos datos
    # ------------------------------------------------------------------------------------

    print('=============================== Datos del posicionamiento ==================================')

    print('Cantidad total de epocas:            {:>10d}'.format(len(epocas)))
    for c in constelaciones:
        print('Satelites  a  procesar de la constelacion {}: '.format(c), end='')
        for sat in sats_con_datos[c]:
            print('{}{:02d} '.format(c, sat+1), end='')
        print('\nObservables a procesar de la constelacion {}: '.format(c), end='')
        for obs in observables[c]:
            if obs[0] != 'L':
                continue
            print('{} '.format(obs), end='')

    # cantidad de observaciones
    n_fases = n_fases0 = tf.boolean_mask(ten_mediciones, ten_mascara).shape[0]
    print('\nCantidad de observaciones de fase:   {:>10d}'.format(n_fases))
    print('Duracion de la sesion: {}'.format(epocas[-1] - epocas[0]))

    # ------------------------------------------------------------------------------------
    # terminadas las inicializaciones, comienza la optimizacion
    # ------------------------------------------------------------------------------------

    print('=============================== Iniciando la optimizacion ==================================')

    porcentaje_fijas = 0        # variable que almacena el porcentaje de ambiguedades fijas durante la optimizacion
    arr_mascara_original = ten_mascara.numpy()

    for k in range(MAX_ITER):

        print('        ————————————————————————————————————————————————————————————————————————————')
        print('       | {:10s} | {:^10s} | {:^14s} | {:^14s} | {:^14s} |'.format('Iteracion', 'Sigma [m]', 'X [m]', 'Y [m]', 'Z [m]'))
        print('       |————————————|————————————|————————————————|————————————————|————————————————|')

        # inicializa el optimizador
        opt = lbfgs.LBFGS(parametros, tolerance_change=TOL, history_size=HS, max_iter=MAX_ITER_LBFGS, line_search_fn='wolfe')

        # para cada iteracion
        for iters in range(MAX_ITER_OP):

            # ------------------------------------------------------------------------
            # normalizacion de los residuos
            # ------------------------------------------------------------------------

            ten_modelo = modelo_gnss(modulos, info_conocida)
            ten_residuos = ten_modelo - ten_mediciones

            ten_residuos_normalizados = ten_residuos * (ten_peso ** (1 / 2))
            costo = tf.reduce_sum(tf.boolean_mask(ten_residuos_normalizados, ten_mascara) ** 2)
            ecm_fases_norm = np.sqrt(costo.numpy() / n_fases)

            # ------------------------------------------------------------------------
            # deteccion de residuos groseros
            # ------------------------------------------------------------------------

            arr_mascara_residuos = np.logical_and(np.abs(ten_residuos_normalizados.numpy()) >= FRES * ecm_fases_norm, arr_mascara_original)
            nres_f = np.count_nonzero(arr_mascara_residuos)

            # esto permite que las observaciones vuelvan a ser tenidas en cuenta
            arr_nueva_mascara_fases = np.copy(arr_mascara_original)
            arr_nueva_mascara_fases[arr_mascara_residuos] = False
            ten_mascara = tf.constant(arr_nueva_mascara_fases)
            n_fases = n_fases0 - nres_f

            # ------------------------------------------------------------------------
            # realiza un paso de entrenamiento (MAX_ITER_LBFGS iteraciones)
            # ------------------------------------------------------------------------

            _, convergencia = opt.step(costo_y_gradientes, [modulos, parametros, regularizaciones, info_conocida, ten_mediciones, ten_mascara, ten_peso])

            # -----------------------------------------------------
            # imprime resultados parciales
            # -----------------------------------------------------

            # compara las coordenadas obtenidas con las coordenadas precisas de la estacion
            r_movil = parametros[0].numpy()
            dxyz = r_movil - coordenadas_comparacion[1:, :]     
            
            # calcula el desvio estandar de las fases
            costo = tf.reduce_sum(tf.boolean_mask(ten_residuos, ten_mascara) ** 2)
            ecm_fases = np.sqrt(costo.numpy() / n_fases)

            iteraciones_lbfgs = int(opt.state['n_iter'])
            print('       | {:>10d} | {:>10.4f} | {:>14.4f} | {:>14.4f} | {:>14.4f} |'.format(iteraciones_lbfgs, ecm_fases, *dxyz[0]), end='\r')

            if ecm_fases > 10 ** 15:
                # fin del entrenamiento por divergencia
                print('\nOptimizacion divergente')
                return

            n_observaciones = [n_fases0, n_fases]
            ecms = [ecm_fases, ecm_fases_norm]

            # si el optimizador llego a converger
            if convergencia:
                
                print('       | {:>10d} | {:>10.4f} | {:>14.4f} | {:>14.4f} | {:>14.4f} |'.format(iteraciones_lbfgs, ecm_fases, *dxyz[0]))
                print('        ————————————————————————————————————————————————————————————————————————————')

                arr_ambiguedades = parametros[-1].numpy()
                
                # en la primera itaracion se calculan los sesgos con las partes fraccionales de las ambiguedades y
                # se agrega la variable a la lista de parametros para las proximas iteraciones
                if k == 0:

                    sesgos_calculados = gnss.resolver_sesgos(arr_ambiguedades, arr_lambdas, ten_mascara.numpy())

                    var_sesgos = modulos[2][1]
                    arr_sesgos = var_sesgos.numpy()
                    arr_sesgos[0] = sesgos_calculados[0] * VLUZ / FREQS['G'][1]
                    arr_sesgos[1] = sesgos_calculados[1] * VLUZ / FREQS['G'][2]
                    var_sesgos.assign(arr_sesgos)
                    parametros.insert(2, var_sesgos)

                # ------------------------------------------------------------------------
                # fijado de ambiguedades por redondeo
                # ------------------------------------------------------------------------

                # pasa a ciclos
                arr_ambiguedades_ciclos = arr_ambiguedades / arr_lambdas

                # calcula las ambiguedades enteras al quitar los sesgos
                if k == 0:
                    arr_ambiguedades_ciclos += sesgos_calculados
                
                # redondea y calcula diferencias entre fija y flotante
                arr_ambiguedades_enteras = np.round(arr_ambiguedades_ciclos, 0)
                arr_diferencias = np.abs(arr_ambiguedades_ciclos - arr_ambiguedades_enteras)

                # mascara de las ambiguedades fijas
                arr_mascara_fijas = np.logical_and(arr_diferencias < TOL_REDONDEO_AMBS, ten_mascara.numpy())

                # las que se puedieron fijar se reemplazan en tensor de ambiguedades
                arr_ambiguedades_ciclos[arr_mascara_fijas] = arr_ambiguedades_enteras[arr_mascara_fijas]

                # pasa a metros
                arr_ambiguedades_iniciales = arr_ambiguedades_ciclos * arr_lambdas

                # donde se pudo fijar las ambiguedades no se optimiza mas
                arr_mascara_para_reinicio = ten_mascara.numpy()
                arr_mascara_para_reinicio[arr_mascara_fijas] = False
                ten_mascara_para_reinicio = tf.constant(arr_mascara_para_reinicio)

                # calcula cuantas ambiguedades se pudieron fijar
                n_fijas = np.count_nonzero(arr_mascara_fijas)
                nuevo_porcentaje_fijas = n_fijas / n_fases * 100
                
                print('\nSe fijaron {} ambiguedades en total'.format(n_fijas))
                print('Porcentaje de ambiguedades fijas: {:>.2f}%\n'.format(nuevo_porcentaje_fijas))

                # si no hay mas ambiguedades por fijar, se guardan los resultados y se finaliza la optimizacion
                if porcentaje_fijas == nuevo_porcentaje_fijas:
                    guarda_resultados(datos_resultados, epocas, constelaciones, sats_con_datos, observables, ten_residuos, ten_mascara, arr_azel, arr_lambdas, r_base, parametros, n_fijas, n_observaciones, ecms)   
                    return


                porcentaje_fijas = nuevo_porcentaje_fijas

                # reinicia el modulo de ambiguedades, considerando aquellas que puedieron fijarse
                fn, a, b, fn_regularizacion_ambiguedades, arr_lambdas = gnss.mod_ambiguedades(observables, ten_mascara_para_reinicio, arr_ambiguedades_iniciales)

                modulos[-1]          = (fn, a)
                parametros[-1]       = a
                info_conocida[-1]    = b
                regularizaciones[-1] = (fn_regularizacion_ambiguedades, a)
                
                break


    print('\nSe supero la cantidad maxima de iteraciones')

    # guarda los resultados
    guarda_resultados(datos_resultados, epocas, constelaciones, sats_con_datos, observables, ten_residuos, ten_mascara, arr_azel, arr_lambdas, r_base, parametros, n_fijas, n_observaciones, ecms)   
    
    return
        


def main():

    # ejemplo de ejecucion
    # python3 posicionamiento_relativo.py -config config.cfg
    # nohup bash run_tests.sh > output.log &

    parser = argparse.ArgumentParser(description='Procesamiento GNSS de simples diferencias de fases en un entorno de machine learning')

    parser.add_argument('-config', type=str, help='nombre del archivo de configuracion')
    args = parser.parse_args()

    if args.config is None:
        parser.print_help()
        exit()

    t0 = datetime.datetime.now()

    # lee archivo de configuracion, rinex de observacion, archivo de navegacion, antex, coordenadas de la base y datos de las antenas
    datos = lectura_de_datos(args.config)

    # procesamiento de las observaciones
    posicionamiento_relativo(*datos)

    t1 = datetime.datetime.now()

    print('Tiempo de ejecucion: {:.1f} mins'.format((t1 - t0).total_seconds()/60))


if __name__ == '__main__':
    
    main()
