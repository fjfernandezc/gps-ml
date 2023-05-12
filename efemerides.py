'''
Calculadora de las coordenadas de los satelites y los desfasajes de sus relojes
2022.04.24
'''

import numpy as np
from datetime import timedelta

from constantes import *
import pyrtklib
import utiles


def satpos(rinexs, nav, c, epocas, sats_con_datos, observables):

    '''
    funcion para determinar las coordenadas de los satelites y el desfasaje de sincronizacion de sus relojes
    recibe:
        rinexs                  listado de los rinex
        nav                     datos de navegacion (struct de rtklib)
        c                       constelacion
        epocas                  lista de epocas
        sats_con_datos          lista de satelites con datos para la constelacion c
        observables             lista de observables con datos para la constelacion c
    devuelve:
        r_sats                  array de coordenadas de los satelites para las observaciones de cada receptor [nrec x neps x nsat x nobs x 5]
                                la ultima dimension contiene a [x^s, y^s, z^s, delta^s, P_L1] (coordenadas del satelite, el desfasaje de su reloj y un codigo)
        satelites_defectuosos   array de mascara que indice que satelites no son defectuosos [neps x nsat x nobs]
    '''

    nrec = len(rinexs)              # numero de receptores
    neps = len(epocas)              # numero de epocas
    nsat = len(sats_con_datos)      # numero de satelites
    nobs = len(observables)         # numero de observables
    ncod, nfas = utiles.n_observables(observables)

    # arma el array donde se van a guardar las posiciones de los satelites, las correcciones a
    # sus relojes y la pseudodistancia utilizada para calcular las coordenadas.
    # dimensiones [nrec x neps x nsat x nobs x 5]
    r_sats = np.zeros((nrec, neps, nsat, nobs, 5))

    satelites_defectuosos = np.full((neps, nsat, nobs), True)

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
            for isat, sat in enumerate(sats_con_datos):

                satid = '{}{:02d}'.format(c, sat+1)
                satno = pyrtklib.satid2no(satid)

                # busco si hay observaciones validas en el rinex para esa epoca y ese satelite
                m = np.nonzero(rnx.observaciones[c][ieps_rnx, isat, :ncod])
                if m[0].size == 0:
                    # si no hay ningun codigo registrado para ese receptor, en esa epoca y para ese satelite continua con el proximo
                    continue

                # a los efectos de calcular las coordenadas del satelite usa el primer codigo encontrado
                codigo = rnx.observaciones[c][ieps_rnx, isat, :ncod][m][0]

                # obtiene el tiempo de emision con el codigo y calculo las coordenadas del satelite y el error de su reloj
                t_emision = epoca - timedelta(microseconds=(codigo / VLUZ)*1E6)
                r_sat, delta_sat, _, _ = pyrtklib.satpos(t_emision, t_emision, satno, nav, ephopt=0)

                # resta el desfasaje de sincronizacion del satelite
                t_emision = t_emision - timedelta(microseconds=delta_sat[0]*1E6)
                r_sat, delta_sat, _, svh = pyrtklib.satpos(t_emision, t_emision, satno, nav, ephopt=0)

                # si el satelite es defectuoso continua
                if svh != 0:
                    satelites_defectuosos[ieps, isat, :] = False
                    continue

                # mascara de donde hay observaciones en ese receptor, satelite y epoca
                m = np.nonzero(rnx.observaciones[c][ieps_rnx, isat, :])

                # guarda los resultados en el array
                r_sats[irec, ieps, isat, m, 0:4] = np.array([r_sat[0], r_sat[1], r_sat[2], delta_sat[0]])
                r_sats[irec, ieps, isat, m,   4] = np.array(rnx.observaciones[c][ieps_rnx, isat, :][m])
    
    return r_sats, satelites_defectuosos

