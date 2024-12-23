import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def plot_distrib_with_analyt(rand_var, x_axis, w, graf_param: list):
    """
    Функция позволяет отобразить экспериментальную и аналитическую 
    плотности распределения вероятности на одном графике.

    Parametrs
    ---------
    rand_var : array_like
        Массив случайных величин.
    rand_var : [array_like]
        Список массивов случайных величин.
    x_axis : array_like
        Интервал значений, на котором задана плотность распределения вероятности.
    w : array_like
        Массив значений теоретической плотности распеределения вероятности.
    graf_param : [string, string, string]
        Список, в качестве элементов которого передаются:
        - название графика,
        - легенда к гистограмме (возможна в виде списка при передачи списка СВ),
        - название оси Ox.   
    graf_param : [string, string, string, tuple]
        Список, в качестве элементов которого передаются:
        - название графика,
        - легенда к гистограмме (возможна в виде списка при передачи списка СВ),
        - название оси Ox,
        - масштаб отображения по оси Ox.
        
    Returns
    -------
    Функция строит график и ничего не возвращает.
    """
    title = graf_param[0]   # название графика
    label = graf_param[1]   # легенда для гистограммы
    xlabel = graf_param[2]  # переменная для отображения названия осей

    if len(graf_param) == 4:
        scale = graf_param[3]   # масштаб по оси Ox
    else:
        scale = (np.min(x_axis), np.max(x_axis))   # масштаб по оси Ox по умолчанию

    plt.figure()
    plt.plot(x_axis, w, 'g-', label='аналитическая')
    plt.hist(rand_var, density=True, bins=x_axis, label=label)
    plt.legend(loc='best', frameon=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('W(' + xlabel + ')', rotation='horizontal')
    plt.xlim(scale)
    plt.grid()
    plt.show()


def plot_2D_func(f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                 x_axis: np.ndarray):
    """
    Функция позволяет отобразить график функции от одной переменой.

    Parametrs
    ---------
    f : [[array_like, array_like], array_like]
        Функция нелинейного преобразования, заданная как функция одной переменной.
    x_axis : array_like
        Массив значений по оси абсцис.
    
    Returns
    -------
    Функция строит график и ничего не возвращает.
    """
    plt.figure()
    plt.plot(x_axis, f(x_axis))
    plt.title('Функиця нелинейного преобразования')
    plt.xlabel('x')
    plt.ylabel('f(x)', rotation='horizontal')
    plt.grid()
    plt.show()

def plot_3D_func(f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                 x_1_axis: np.ndarray, 
                 x_2_axis: np.ndarray):
    """
    Функция позволяет отобразить график функции от двух переменных.

    Parametrs
    ---------
    f : [[array_like, array_like], array_like]
        Функция нелинейного преобразования, заданная как функция двух переменных.
    x_1_axis : array_like
        Массив значений по первой оси.
    x_2_axis : array_like
        Массив значений по второй оси.
    
    Returns
    -------
    Функция строит график и ничего не возвращает.
    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1, 1, 1, projection = '3d')

    axis_1, axis_2 = np.meshgrid(x_1_axis, x_2_axis)    # создание координатной сетки
    z = f(axis_1, axis_2)   # функция нелинейного преобразования
    
    surf = ax.plot_surface(axis_1, axis_2, z)   # построение графика функции от двух переменных

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('f($x_1$, $x_2$)')
    ax.set_title('Функция нелинейного преобразования')