'''
Constantes para el procesamiento GNSS
2022.04.24
'''

# constantes fisicas y matematicas
VLUZ = 299792458.0              # velocidad de la luz (m/s)
PI   = 3.1415926535897932       # pi
OMGE = 7.2921151467E-5          # velocidad angular de la tierra (rad/s)
D2R  = PI / 180.0               # grados a radianes
R2D  = 180.0 / PI               # radianes a grados

# parametros wgs84
RE_WGS84 = 6378137.0
FE_WGS84 = 1.0 / 298.257223563

# frecuencias de las senales gnss (Hz)
FREQS = {
    'G': {
        1: 1.57542E9,
        2: 1.22760E9,
        5: 1.17645E9
    },
    'R': {
        1: 1.60200E9,
        2: 1.24600E9,
        3: 1.202025E9,
        4: 1.600995E9,
        6: 1.248060E9
    },
    'E': {
        1: 1.57542E9,
        5: 1.17645E9,
        6: 1.27875E9,
        7: 1.20714E9,
        8: 1.191795E9
    },
    'C': {
        1: 1.57542E9,
        2: 1.561098E9,
        5: 1.17645E9,
        6: 1.26852E9,
        7: 1.20714E9,
        8: 1.191795E9
    },
    'S': {
        1: 1.57542E9,
        5: 1.17645E9
    }
}

DFRQ1_GLO = 0.56250E6
DFRQ2_GLO = 0.43750E6

# maximo numero de satelites de cada constelacion
MAX_PRNS = {
    'G': 32,    # gps
    'R': 27,    # glonass
    'E': 36,    # galileo
    'C': 67,    # beidou
    'S': 39     # sbas
}

ATRIBUTOS_OBSERVABLES = {
    'G': {
        '1': 'P W C S L X Y M N',
        '2': 'P W C S L X Y M N',
        '5': 'I Q X'
    },
    'E': {
        '1': 'B C X',
        '5': 'I Q X',
        '6': 'B C X',
        '7': 'I Q X',
        '8': 'I Q X'
    }
}

# desvio estandar para las fases
STDF  = 0.007

# maxima cantidad de iteraciones para optimizar
MAX_ITER = 30

# parametros de L-BFGS
TOL            = 1e-7
HS             = 100
MAX_ITER_LBFGS = 50
MAX_ITER_OP    = 100000


# tolerancias para los residuos
FRES = 4

# en grados
MASCARA_ELEVACION = 3.0

# tolerancia de redondeo para fijado de ambiguedades
TOL_REDONDEO_AMBS = 0.3

# hiperparametro para la regularizacion de las ambiguedades
GAMMA = 2 ** 23

