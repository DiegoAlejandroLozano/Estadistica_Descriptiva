from numpy import double
import pandas as pd
import math
from matplotlib import pyplot as plt
import statistics

from zmq import XPUB_NODROP


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


    def curtosis(self, xi:pd.Series, ni:pd.Series) -> float:
        '''Función encargada de cálcular el coeficiente
        de curtosis'''
        sumatoria = 0
        x = xi.mean()
        s_4 = xi.std()**4
        n = xi.count()
        for i in range(n):
            sumatoria += (ni.iloc[i]/n)*(xi.iloc[i]-x)**4
        return (sumatoria/s_4)-3


    def graficar_datos(self, datos:pd.Series, ni:pd.Series, ancho:float) -> None:
        '''Función encargada de graficar los datos'''
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,8))
        ax[0].bar(datos, ni, width=ancho)
        ax[0].vlines(x=datos.mean(), ymin=0, ymax=ni.max(), color='red', linestyle='--', label='Media ({0:.3f})'.format(datos.mean()))
        ax[0].vlines(x=datos.median(), ymin=0, ymax=ni.max(), color='blue', linestyle='--', label='Mediana ({0:0.3f})'.format(datos.median()))
        ax[0].grid()
        ax[0].legend(loc='upper left')
        ax[1].boxplot(datos, vert=False)
        plt.show()


    def funcion_de_densidad(self, dato:float, datos:pd.Series) -> float:
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

    #Funciones utilizadas en estadística bidimensional************************


    def tabla_dist_frec_bidi(self, datos:list) -> pd.DataFrame:
        '''Función encargada de encontrar la tabla de
        distribución de frecuencias bidimensional de la variable
        
        Parámetros:
        
        datos --> Lista de tuplas, las cuales contienen los valores de
        la variable bidimensional. [(x1,y1),(xi,yj)...,(xp, yq)]
        
        Valor de retorno:
        
        Se retorna un DataFrame el cual contiene la tabla de distribución
        de frecuencias bidimensional. El nombre de las columnas corresponde
        a los valores de X y los indices corresponden a los valores de Y'''
        x=list()
        y=list()
        for i in datos:
            x.append(i[0])
            y.append(i[1])
        #Valores de x
        valores_x = []
        for val_x in x:
            if not val_x in valores_x:
                valores_x.append(val_x)
        valores_x.sort()
        valores_x
        #Valores de y
        valores_y = []
        for val_y in y:
            if not val_y in valores_y:
                valores_y.append(val_y)
        valores_y.sort()
        valores_y
        #Encontrando las frecuencias absolutas
        datos_tb_f = []
        for y_j in valores_y:
            fila = []
            for x_i in valores_x:
                #Ya forme la pareja (x_j,y_i)
                #comparando cada valor de la variabñe
                suma = 0
                if not (x_i, y_j) in datos:
                    #Si el dato no esta en el conjunto, se evita recorrerlo
                    fila.append(0)
                else:
                    #Si el dato si esta en el conjunto, se recorre para contar
                    #cuantas veces aparece 
                    for dato in datos:
                        if dato == (x_i, y_j):
                            suma += 1
                    fila.append(suma)
            datos_tb_f.append(fila)
        #Creando el dataframe
        tabla_dist_fre_bidime = pd.DataFrame(data=datos_tb_f, columns=valores_x, index=valores_y)
        return tabla_dist_fre_bidime

    
    def frec_abs_variables(self, tabla_dist_frec_bidi:pd.DataFrame) -> dict[str, dict]:
        '''Encuentra las frecuencias absolutas de las variables X y Y
        
        Parámetros:

        tabla_dist_frec_bidi-->DataFrame que representa la distribución de frecuencia
        bidimensional
        
        Valor de retorno:

        Retorna una diccionario de diccionarios, el cual contiene las
        frecuencias absolutas de cada valor de Y y X'''
        frec_abs_x = {}
        for i in tabla_dist_frec_bidi.columns:
            frec_abs_x[i] = tabla_dist_frec_bidi[i].sum()
        frec_abs_y = {}
        for i in tabla_dist_frec_bidi.index:
            frec_abs_y[i] = tabla_dist_frec_bidi.loc[i].sum()
        return {'variable X':frec_abs_x, 'Variable Y':frec_abs_y}


    def covarianza(self, datos:list) -> double:
        '''Calcula la covarianza de la variable aleatoria bidimendional (x,y)
        
        Parámetros:
        
        datos: Datos de la forma [(x1,y1),(x2,y2),...,(xi,yi)]
        
        Valor de retorno:
        
        Retorna el valor de la covarianza calculada'''

        valores_x = []
        valores_y = []
        for dato in datos:
            valores_x.append(dato[0])
            valores_y.append(dato[1])
        mean_x = statistics.mean(valores_x)
        mean_y = statistics.mean(valores_y)
        n = len(valores_x)
        #ecuacion 1
        sumatoria = 0
        for i in range(n):
            sumatoria += (valores_x[i]-mean_x)*(valores_y[i]-mean_y)
        s_xy = (1/n)*sumatoria
        return s_xy


    def recta_regresion(self, valores_x:pd.Series, valores_y:pd.Series, valor_x:double) -> double:
        '''Realiza la regresión linea de los datos y predice el valor Y a partir
        del valor X
        
        Parámetros:
        
        valores_x: Los valores de x de la muestra
        
        valores_y: Los valores de y de la muestra
        
        valor_x: Valor de x para realizar la predicción
        
        Valor de retorno:
        
        Retorna el valor de Y calculado'''
        s_xy = valores_x.cov(valores_y)
        varianza_x = valores_x.var()
        x_mean = valores_x.mean()
        y_mean = valores_y.mean()
        #Ecuación de la recta de regresión
        y = y_mean + (s_xy/varianza_x)*(valor_x-x_mean)
        return y


#Funciones utilizadas en correlaciones***********************************


    def coeficiente_determinacion(self, x:pd.Series, y:pd.Series, y_pre:pd.Series) -> double:
        '''Calcula el coeficiente de determinación (r^2) a partir de los datos suministrados
        en los parámetrods.
        
        Parámetros:
        
        x: Serie de valores de la variable x
        
        y: Serie de valores de la variable y
        
        y_pre: Serie con los valores de la predicción de y con la
        recta de regresión
        
        Valor de retorno:
        
        Retorna el valor calculado de r^2'''

        #Encontrando la varianza residual muestral
        v_res = 0
        suma_1 = 0
        n = x.size
        for j in range(n):
            suma_1 += (y[j]-y_pre[j])**2
        s_ry = (1/n)*suma_1
        #Calculando la varianza de y
        s_y = y.var()
        #Calculando el coeficiente de determinación lineal
        r_cua = 1 - (s_ry/s_y)
        return round(r_cua, 2)

    
    def coef_correl_lineal(self, x:pd.Series, y:pd.Series, y_pre:pd.Series) -> double:
        '''Calcula el coeficiente de correlación lineal a partir de los datos suministrados
        en los parámetrods.
        
        Parámetros:
        
        x: Serie de valores de la variable x
        
        y: Serie de valores de la variable y
        
        y_pre: Serie con los valores de la predicción de y con la
        recta de regresión
        
        Valor de retorno:
        
        Retorna el valor calculado del coeficiente'''
        s_xy = x.cov(y)
        s_x = x.std()
        s_y = y.std()
        r = s_xy/(s_x*s_y)
        return r


#Funciones utilizadas en regresión de modelos no lineales***********************************


    def funcion_logaritmica(self, valores_x:pd.Series, valores_y:pd.Series) -> list:
        '''Función encargada de realizar la predicción de los valores de y de una función
        logaritmica, a través de regresión de modelos no lineales por el método de cambio
        de variable:
        
        y = a + b*log(x)
        
        Parámetros de entrada:
        
        valores_x: Serie de valores de la variables x
        
        valores_y: Serie de valores de la variable y
        
        Valor de retorno:
        
        Se retorna un array con los valores de la predicción de la variable y'''
        t = valores_x.apply(math.log10)
        y_pre = []
        for i in t:
            y = self.recta_regresion(t, valores_y, i)
            y_pre.append(y)
        return y_pre


    def funcion_exponencial(self, valores_x:pd.Series, valores_y:pd.Series) -> list:
        '''Función encargada de realizar la predicción de los valores de y de una función
        exponencial, a través de regresión de modelos no lineales por el método de cambio
        de variable:
        
        y = a*e^(bx)
        
        Parámetros de entrada:
        
        valores_x: Serie de valores de la variables x
        
        valores_y: Serie de valores de la variable y
        
        Valor de retorno:
        
        Se retorna un array con los valores de la predicción de la variable y'''
        y_pri = valores_y.apply(math.log)
        y_predic = []
        for i in valores_x:
            y = self.recta_regresion(valores_x, y_pri, i)
            y_predic.append(math.exp(y))
        return y_predic


    def funcion_potencia(self, valores_x:pd.Series, valores_y:pd.Series) -> list:
        '''Función encargada de realizar la predicción de los valores de y de una función
        potencia, a través de regresión de modelos no lineales por el método de cambio
        de variable:
        
        y = a*x^b
        
        Parámetros de entrada:
        
        valores_x: Serie de valores de la variables x
        
        valores_y: Serie de valores de la variable y
        
        Valor de retorno:
        
        Se retorna un array con los valores de la predicción de la variable y'''
        y_pri = valores_y.apply(math.log10)
        x_pri = valores_x.apply(math.log10)
        y_predi = []
        for i in x_pri:
            y = self.recta_regresion(x_pri, y_pri, i)
            y_predi.append(10**y)
        return y_predi
