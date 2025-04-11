import numpy as np
from numpy.fft import fft
from numpy.fft import fftshift
import matplotlib.pyplot as plt

def plot_distrib_with_analyt(proc, x_axis, w, title: str):
    """
    Функция позволяет отобразить экспериментальную и аналитическую 
    плотности распределения вероятности на одном графике.

    Parametrs
    ---------
    proc : array_like
        Отсчёты случайного процесса.
    x_axis : array_like
        Интервал значений, на котором задана плотность распределения вероятности.
    w : array_like
        Массив значений теоретической плотности распеределения вероятности.
    title : string
        Название графика.
        
    Returns
    -------
    Функция строит график и ничего не возвращает.
    """
    plt.figure()
    plt.plot(x_axis, w, 'g-', label='аналитическая')
    plt.hist(proc.flatten(), density=True, bins=x_axis, label='исходная СВ')
    plt.legend(loc='best', frameon=False)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('W(x)', rotation='horizontal')
    plt.grid()
    plt.show()


def plot_distrib(proc_param, title: str):
    """
    Функция позволяет отобразить экспериментальную плотность распределения веротности процессов.
    В случае необходимости вывода нескольких линий на один график, они передаются в качестве второй 
    размерности массива, типа: [[`proc_param` первой линии], [`proc_param` второй линии]].
    Одинаковая размерность не обязательна.

    Parametrs
    ---------
    proc_param : [array_like]
            Многомерный список, в качестве элементов которого передаются
            списки массивов отсчётов случайного процесса.
    proc_param : [array_like, string]
            Многомерный список, в качестве элементов которого передаются списки:
            - массив отсчётов случайного процесса в качестве первого аргумента,
            - название линии (пишется в легенде) в качестве второго аргумента.
    title : string
            Название графика.
                 
    Returns
    -------
    Функция строит график и ничего не возвращает.
    
    Raises
    ------
    Exeption
        Если не корректно задан аргумент proc_param.
    TypeError
        Если в функцию было передано неверное число аргументов параметра proc_param.
    """
    plt.figure()

    # Проверка на корректность введенных данных:
    # isinstance(i, list): функция проверяет, является ли объект i экземпляром типа list (True - является, False - не является).
    # not isinstance(i, list): инвертирутся результат проверки. Если i является списком, выражение вернёт False, а если нет — True.
    # for i in proc_param: это часть генератора, который перебирает каждый элемент i в списке proc_param.
    # all(...): возвращает True, если все элементы в переданном ей итерируемом объекте являются истинными.
    if all(not isinstance(i, list) for i in proc_param):
        raise Exception('Ошибка ввода. Возможно, в качестве аргумета corr_funcs передается не многомерный СПИСОК' + 
                        '(массив отсчётов должен передаваться в виде [[...]] или [[...], ..., [...]])!')

    proc = []   # создание пустого списка для записи всех СП, переданных в функцию
    label = []  # создание пустого списка для записи всех легенд, переданных в функцию
    for i in range(len(proc_param)):    # цикл по числу СП, переданных в функцию
        args_len = len(proc_param[i])   # количество аргументов, переданных для i-ого СП
        
        proc.append(proc_param[i][0].flatten()) # добавляем очередной СП в список и превращаем его в одномерный массив
                                                # (требуется, если СП представляет собой совокупность нескольких реализаций)

        if args_len == 0 or args_len > 2:  # если аргументов оказалось не столько, сколько ожидалось.
            raise TypeError('Ошибка ввода. Неверное число аргументов (допускается 1 или 2 аргумента)') 
        
        # Формирование названий линий
        if args_len == 2:
            label.append(str(proc_param[i][1]))
        else:
            label.append('Безымянная ПР ' + str(i))
        
    plt.hist(proc, bins=128, density=True, label=label) # отображение гистограммы 
    plt.legend(loc='best', frameon=False)
    plt.title(title)
    plt.xlabel('y')
    plt.ylabel('W(y)', rotation='horizontal')
    plt.grid()
    plt.show()


def plot_corr_func(corr_funcs_param, time_window: int | float, n_counts: int, bound_graf: int | float):
    """
    Функция позволяет отобразить график корреляционной функции.
    В случае необходимости вывода нескольких линий на один график, они передаются в качестве второй 
    размерности массива, типа: [[`corr_func_param` первой линии], [`corr_func_param` второй линии]].
    Одинаковая размерность не обязательна.

    Parametrs
    ---------
    corr_funcs_param : [array_like]
                Многомерный список, в качестве элементов которого передаются
                списки массивов отсчётов корреляционных функций.
    corr_funcs_param : [array_like, string]
                Многомерный список, в качестве элементов которого передаются списки:
                - массив отсчётов КФ в качестве первого аргумента,
                - название линии (пишется в легенде) в качестве второго аргумента.
    time_window : int, float
               Временной интервал, на котором рассматривается корреляция, мкс.
    n_counts : int
               Число отсчётов.
    bound_graf : int, float
               Границы построения графика по времени, мкс.
    
    Returns
    -------
    Функция строит график и ничего не возвращает.

    Raises
    ------
    Exeption
        Если не корректно задан аргумент corr_funcs_param.
    TypeError
        Если в функцию было передано неверное число аргументов параметра corr_funcs_param.
    """
    
    shift_axis_us = np.linspace(-time_window, time_window, 2*n_counts-1)  # ось отстройки КФ
    
    plt.figure()

    # Проверка на корректность введенных данных:
    # isinstance(i, list): функция проверяет, является ли объект i экземпляром типа list (True - является, False - не является).
    # not isinstance(i, list): инвертирутся результат проверки. Если i является списком, выражение вернёт False, а если нет — True.
    # for i in corr_funcs_param: это часть генератора, который перебирает каждый элемент i в списке corr_funcs_param.
    # all(...): возвращает True, если все элементы в переданном ей итерируемом объекте являются истинными.
    if all(not isinstance(i, list) for i in corr_funcs_param):
        raise Exception('Ошибка ввода. Возможно, в качестве аргумета corr_funcs_param передается не многомерный СПИСОК' + 
                        '(массив отсчётов должен передаваться в виде [[...]] или [[...], ..., [...]])!')
    
    for i in range(len(corr_funcs_param)):    # цикл по числу КФ, переданных в функцию
        args_len = len(corr_funcs_param[i])   # количество аргументов, переданных для i-ой КФ
        
        if args_len == 0 or args_len > 2:  # если аргументов оказалось не столько, сколько ожидалось.
            raise TypeError('Ошибка ввода. Неверное число аргументов (допускается 1 или 2 аргумента)') 
        
        # Формирование названий линий
        if args_len == 2:
            label = str(corr_funcs_param[i][1])
        else:
            label = 'Безымянная КФ ' + str(i)
        
        plt.plot(shift_axis_us, corr_funcs_param[i][0], label=label)    # построение i-го графика

    
    plt.xlim(-bound_graf, bound_graf)
    plt.title('КФ')
    plt.xlabel(r'$\tau$, мкс') 
    plt.ylabel(r'R($\tau$)', rotation='horizontal')
    plt.legend()
    plt.grid()
    plt.show()


def plot_SPD_DNWGN(r, f_sampl: int | float, n_counts: int):
    """
    Функция позволяет отобразить спектральную плотность мощности дискретного
    белого гауссовского шума.

    Parametrs
    ---------
    r : array_like
        Массив отсчётов КФ ДБГШ.
    f_sampl : int, float
        Частота дискретизации, МГц.
    n_counts : int
        Число отсчётов.

    Returns
    -------
    Функция строит график и ничего не возвращает.
    """
    freqs = np.linspace(-0.5*f_sampl, 0.5*f_sampl - (f_sampl/n_counts), n_counts) # ось частот

    spectr = fft(r, n_counts)    # СПМ исходного процесса (БПФ от КФ)
    spectr = np.abs(fftshift(spectr))   # выравнивание СПМ относительно центра + взятие по модулю

    spec_x_mean = np.repeat(np.mean(spectr), freqs.shape[0])    # усреднение полученной СПМ

    plt.figure()
    plt.plot(freqs, spectr, label='полученное')
    plt.plot(freqs, spec_x_mean, label='среднее')
    plt.legend(loc='best', frameon=False)
    plt.title('СПМ ДБГШ')
    plt.xlabel(r'$f$, МГц'), plt.ylabel(r'S($f$)', rotation='horizontal')
    plt.xlim((-0.5*f_sampl, 0.5*f_sampl))
    plt.grid()
    plt.show()


def plot_SPD(corr_funcs_param, f_sampl: int | float, n_counts: int, bound_graf: int | float):
    """
    Функция позволяет построить спектральную плотность мощности случайного процесса
    по его корреляционной функции (ПФ от корреляционной функции).
    В случае необходимости вывода нескольких линий на один график, они передаются в качестве второй 
    размерности массива, типа: [[`corr_func_param` первой линии], [`corr_func_param` второй линии]].
    Одинаковая размерность не обязательна.

    Parametrs
    ---------
    corr_funcs_param : [array_like]
               Многомерный список, в качестве элементов которого передаются
               списки массивов отсчётов корреляционных функций.
    corr_funcs_param : [array_like, string]
               Многомерный список, в качестве элементов которого передаются списки:
               - массив отсчётов КФ в качестве первого аргумента,
               - название линии (пишется в легенде) в качестве второго аргумента.
    f_sampl : int, float
               Частота дискретизации, МГц.
    n_counts : int
               Число отсчётов.
    bound_graf : int, float
               Границы построения графика по частоте, МГц.
    
    Returns
    -------
    Функция строит график и ничего не возвращает.
    
    Raises
    ------
    Exeption
        Если не корректно задан аргумент corr_funcs_param.
    TypeError
        Если в функцию было передано неверное число аргументов параметра corr_funcs_param.
    """
    freqs = np.linspace(-0.5*f_sampl, 0.5*f_sampl - (f_sampl/n_counts), n_counts) # ось частот

    plt.figure()

    # Проверка на корректность введенных данных:
    # isinstance(i, list): функция проверяет, является ли объект i экземпляром типа list (True - является, False - не является).
    # not isinstance(i, list): инвертирутся результат проверки. Если i является списком, выражение вернёт False, а если нет — True.
    # for i in proc_param: это часть генератора, который перебирает каждый элемент i в списке proc_param.
    # all(...): возвращает True, если все элементы в переданном ей итерируемом объекте являются истинными.
    if all(not isinstance(i, list) for i in corr_funcs_param):
        raise Exception('Ошибка ввода. Возможно, в качестве аргумета corr_funcs_param передается не многомерный СПИСОК' + 
                        '(массив отсчётов должен передаваться в виде [[...]] или [[...], ..., [...]])!')
    
    for i in range(len(corr_funcs_param)):    # цикл по числу КФ, переданных в функцию
        args_len = len(corr_funcs_param[i])   # количество аргументов, переданных для i-ой КФ
        if args_len == 0 or args_len > 2:   # если аргументов оказалось не столько, сколько ожидалось.
            raise TypeError('Ошибка ввода. Неверное число аргументов (допускается 1 или 2 аргумента)')
        
        spectr = fft(corr_funcs_param[i][0], n_counts)
        spectr = abs(fftshift(spectr))

        # Формирование названий линий
        if args_len == 2:
            label = str(corr_funcs_param[i][1])
        else:
            label = 'Безымянная СПМ ' + str(i)
        
        plt.plot(freqs, spectr, label=label)    # отображение очередного графика

    plt.legend(loc='best', frameon=False)
    plt.title('СПМ процесса с заданной КФ')
    plt.xlabel(r'$f$, МГц') 
    plt.ylabel(r'S($f$)', rotation='horizontal')
    plt.xlim((-bound_graf, bound_graf))
    plt.grid()
    plt.show()