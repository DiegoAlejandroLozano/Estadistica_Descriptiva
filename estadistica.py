import pandas as pd
import math
from matplotlib import pyplot as plt


class Estadistica:
    '''Modulo básico de estadística'''    

    def tipificacion(self, x:pd.Series)->pd.Series:
        media = x.mean()
        desviacion = x.std()
        return (x-media)/desviacion

    def fisher(self, xi:pd.Series,ni:pd.Series) -> float:
        '''Función encargada de cálcular el coeficiente
        de Fisher'''
        sumatoria = 0
        x = xi.mean()
        s_cubo = xi.std()**3
        n = xi.count()
        for i in range(n):
            sumatoria += ni.iloc[i] * (xi.iloc[i]-x)**3
        return sumatoria / (n*s_cubo)


    def identificar_outliers(self, dato, serie_datos:pd.Series) -> bool:
        '''Función encargada de indetificar si un dato es un
        outlier'''
        q1 = serie_datos.loc['25%'] #Cuartil 1
        q3 = serie_datos.loc['75%'] #Cuartil 2
        iqr = q3-q1 #Rango intercuartílico
        min = q1-(1.5*iqr)
        max = q3+(1.5*iqr)
        if dato<min or dato>max:
            return True
        else:
            return False


    def curtosis(self, xi:pd.Series, ni:pd.Series)->float:
        '''Función encargada de cálcular el coeficiente
        de curtosis'''
        sumatoria = 0
        x = xi.mean()
        s_4 = xi.std()**4
        n = xi.count()
        for i in range(n):
            sumatoria += (ni.iloc[i]/n)*(xi.iloc[i]-x)**4
        return (sumatoria/s_4)-3


    def graficar_datos(self, datos:pd.Series, ni:pd.Series, ancho:float)->None:
        '''Función encargada de graficar los datos'''
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,8))
        ax[0].bar(datos, ni, width=ancho)
        ax[0].vlines(x=datos.mean(), ymin=0, ymax=ni.max(), color='red', linestyle='--', label='Media ({0:.3f})'.format(datos.mean()))
        ax[0].vlines(x=datos.median(), ymin=0, ymax=ni.max(), color='blue', linestyle='--', label='Mediana ({0:0.3f})'.format(datos.median()))
        ax[0].grid()
        ax[0].legend(loc='upper left')
        ax[1].boxplot(datos, vert=False)
        plt.show()


    def funcion_de_densidad(self, dato:float, datos:pd.Series)->float:
        '''Función encargada de calcularlos datos de la campana de Gauss
        Parámetros de entrada:
        dato --> Hace referencia a un dato de la muestra
        datos --> Todos los datos de la muestra
        Parámetros de salida:
        f --> Valor de densidad calculado para el dato de entrada'''
        u = datos.mean()
        s = datos.std()
        b = 1/(s*math.sqrt(2*math.pi))
        f = b * math.exp((-1*(dato-u)**2)/(2*s**2))
        return f
