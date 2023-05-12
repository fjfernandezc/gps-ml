'''
Algoritmo para resolver el problema de posicionamiento autonomo y obtener valores
aproximados de las coordenadas y desfasaje del reloj del receptor
2022.04.28

https://gssc.esa.int/navipedia/index.php/Bancroft_Method
'''

import numpy as np


def lorentz(a, b):

    '''
    producto interno de Lorentz
    '''

    if a.shape[0] != b.shape[0]:
        return 0.0

    dim = a.shape[0]

    M = np.eye(dim, dtype=np.float64)
    M[dim-1, dim-1] = -1.0

    return ((a.T).dot(M)).dot(b)


def bancroft(B):

    '''
    funcion para aplicar el algoritmo de Bancroft y obtener coordenadas aproximadas del receptor
    recibe:
        B           [nsat x 4] donde cada fila corresponde a los datos de un satelite
                    [xs, ys, zs, pr] siendo los primeros tres las coordenadas del satelite
                    y el ultimo valor la pseudodistancia medida
    devuelve:
        s           [x, y, z, c delta]
    '''

    # armo las matrices intervinientes en el calculo
    l = np.ones(B.shape[0])

    a = np.zeros(B.shape[0])
    for i in range(B.shape[0]):
        a[i] = lorentz(B[i, :], B[i, :]) / 2

    # invierto la matriz B
    Bi = np.linalg.inv((B.T).dot(B))

    # computo los coeficientes de la ecuacion cuadratica
    a1 = lorentz((Bi.dot(B.T)).dot(l), (Bi.dot(B.T)).dot(l))
    a2 = 2 * (lorentz((Bi.dot(B.T)).dot(l), (Bi.dot(B.T)).dot(a)) - 1)
    a3 = lorentz((Bi.dot(B.T)).dot(a), (Bi.dot(B.T)).dot(a))

    # obtengo ambas soluciones a la ecuacion
    s1 = (-a2 + np.sqrt(np.power(a2, 2) - 4 * a1 * a3)) / (2 * a1)
    s2 = (-a2 - np.sqrt(np.power(a2, 2) - 4 * a1 * a3)) / (2 * a1)

    s1 = (Bi.dot(B.T)).dot(a + s1 * l)
    s2 = (Bi.dot(B.T)).dot(a + s2 * l)

    # decido con que solucion me quedo
    d1 = abs(np.linalg.norm(s1[0:3]) - 6371000)
    d2 = abs(np.linalg.norm(s2[0:3]) - 6371000)

    if d1 < d2:
        s1[3] = -1 * s1[3]
        return s1
    else:
        s2[3] = -1 * s2[3]
        return s2


if __name__ == '__main__':

    data = np.array([[-5682326.748, -25398680.713,  -5036513.376, 22647521.801],
                     [23960000.051,   3753676.521, -11002388.947, 23759712.298],
                     [ 1812577.482, -16573813.240, -20791210.056, 20940360.088],
                     [13040831.648, -13937491.575, -18446218.111, 20375862.900],
                     [-9945878.347, -12615426.593, -21554868.405, 23457646.733],
                     [22477237.491, -14096489.639,   2135405.779, 22487426.405],
                     [18855201.475, -16114457.008,   8980904.180, 23474010.239],
                     [-1116678.208, -26085934.831,   5550349.037, 23657471.306],
                     [ 9484574.719, -21393834.399, -12252632.716, 20027498.757],
                     [19451586.161,  -4203185.417, -17632343.151, 21904509.856]])


    sol = bancroft(data)
    print('x   {:>14.3f}\ny   {:>14.3f}\nz   {:>14.3f}\ncdt {:>14.3f}'.format(*sol))


