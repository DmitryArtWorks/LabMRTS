import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftshift
import pywt

# # # # # # # # # # # # #
# Отображение сигналов  #
# # # # # # # # # # # # #

# Временная область
def plot_signal(signal_args: list[list]) -> None:
    """
    Вывести действительную часть сигнала, в случае необходимости вывода
    нескольких линий на один график, они передаются в качестве второй
    размерности массива, типа: [[`signal_args` первой линии графика], 
    [`signal_args` второй линии]]. Одинаковая размерность не обязательна.
    В зависимости от числа переданных аргументов (размерности `signal_args`), 
    функция распознает их следующим образом:

    Parameters
    ----------
    signal_args : array_like
                  Значения по оси ординат (ось OY). В таком случае по оси абсцис (ось OX)
                  будут отложены номера отсчетов.
    signal_args : [array_like or scalar, array_like]
                  Набор значений оси абсцисс или частота дискретизации 
                  в качестве первого аргумента. Значения амплитуд для 
                  оси ординат - в качестве второго.
    signal_args : [array_like or scalar, array_like, string]
                  Первые два аргумента подразумеваются такими же, что и для случая выше. В
                  качестве третьего аргумента передается название линии (пишется в легенде).
    
    Returns
    -------
    Функция создает графики и ничего не возвращает.

    Raises
    ------
    ValueError:
        Если в функцию было передано неверное число аргументов.
    """

    lines = ["-", "--", "-.", ":"] # Список возможных стилей линий графиков
    leg = []
    plt.figure()
    for i in range (len(signal_args)):  # Цикл по числу линий, переданных в функцию
        
        args_len = len(signal_args[i])  # Определяем, сколько аргументов передано для i-ой линии.
        if (args_len > 3 or args_len < 1): # Если аргументов оказалось не столько, сколько ожидалось.
            raise TypeError('Ошибка ввода. Неверное число аргументов (допускается от 1 до 3 аргументов)')
        if (args_len == 1): # Ветка для случая, когда был передан один аргумент (набор значений оси ординат)
            signal = signal_args[i][0]
            x_axis = range(0, len(signal))
        if (args_len == 2 or args_len == 3):
            signal = signal_args[i][1]
            if (type(signal_args[i][0]) == float): # Если в качестве первого аргумента передан шаг дискретизации
                x_axis = np.arange(0, signal_args[i][0]*(len(signal)), signal_args[i][0])
            else:
                x_axis = signal_args[i][1]                
    
        # Построение i-й линии        
        plt.plot(x_axis*1e6, np.real(signal), linestyle=lines[i])
        
        if (args_len == 3):
            leg.append(signal_args[i][2])
        else:
            leg.append('Unnamed signal ' + str(i))  # Если не нравится, можно заменить 
                                                    # содержимое скобок на " ". Работает
                                                    # тоже красиво
        
    plt.legend(leg)
    plt.title('Исходный сигнал')
    plt.xlabel("t, мкc") # ось абсцисс
    plt.grid()
    plt.show()


# Частотная область
def plot_spectum(signal_args: list[list]) -> None:
    """
    Построить график модуля БПФ сигнала, в случае необходимости вывода 
    нескольких линий на один график, они передаются в качестве второй 
    размерности массива, типа: [[`signal_args` первой линии], [`signal_args` второй линии]].
    Одинаковая размерность не обязательна.
    
    Parameters
    ----------       
    signal_args : [scalar, array_like]
                  Частота дискретизации в качестве первого аргумента. 
                  Отсчеты сигнала во временной области - в качестве второго.
    signal_args : [scalar, array_like, string]
                  Первые два аргумента подразумеваются такими же, что и для случая выше. В
                  качестве третьего аргумента передается название линии (пишется в легенде).
    
    Returns
    -------
    Функция создает графики и ничего не возвращает.

    Raises
    ------
    TypeError:
        Если в функцию было передано неверное число аргументов.
    """

    t_d_us = signal_args[0][0]*1e6 # Шаг дискретизации, переведенный в микросекунды
    plt.figure()

    for i in range(len(signal_args)):   # Цикл по числу линий, переданных в функцию
        
        args_len = len(signal_args[i])  # Определяем, сколько аргументов передано для i-ой линии.
        if (args_len > 3 or args_len < 2):  # Если аргументов оказалось не столько, сколько ожидалось.
            raise TypeError('Ошибка ввода. Неверное число аргументов (допускается 2 или 3 аргумента)')
        
        signal = signal_args[i][1]  # Массив значений оси ординат
        n_signal = signal.size  # Способ получить число отсчетов (по сути то же самое, что и len())
        t_window = n_signal*t_d_us  # Вычисление размера окна
        f_step_mhz = 1/t_window # Шаг по частоте (нужен для формирования отсчетов оси абсцисс)
        
        # Формируем точки оси абсцисс графика. Сначала формируем
        # область положительных частот, затем путем отзеркаливания
        # формируется область отрицательных частот.
        # В зависимости от того, четное ли число точек в исходном
        # сигнале, или нечетное, "зеркалится" на 1 отсчет больше или
        # меньше
        if n_signal % 2 == 0:
            f_axis = np.arange(0, (n_signal+1)*f_step_mhz/2, f_step_mhz)
            tmp_axis = np.flip(f_axis[2:]) * (-1) 
            f_axis = np.append(tmp_axis, f_axis)
        else:
            f_axis = np.arange(0, (n_signal)*f_step_mhz/2, f_step_mhz)
            tmp_axis = np.flip(f_axis[1:]) * (-1)
            f_axis = np.append(tmp_axis, f_axis)
        
        # БПФ входных отсчетов, нормированное на число отсчетов
        signal_spectrum = 1/n_signal*fft(signal, n_signal)
        
        # Формирование названий линий
        if (len(signal_args[i]) > 2):
            line_label = str(signal_args[i][2])
        else:
            line_label = 'Unnamed spectrum ' + str(i)   # Выражение может быть заменено 
                                                        # на: ' '(пробел в кавычках)
            
        plt.plot(f_axis, fftshift(abs(signal_spectrum)), label=line_label)  # построение i-й линии
    plt.title("Спектр") # заголовок
    plt.xlabel("Частота, МГц") # ось абсцисс        
    plt.legend()
    plt.grid()
    plt.show()


def plot_spectrogram(signal: np.ndarray, window_name: str, window_offset_step: int, window_opt_len=8):
    """
    Функция позволяет построить спектрограмму сигнала при различных параметрах окна.

    Parametrs
    ---------
    signal : array_like
             Отсчеты сигнала во временной области.
    window_name : 'Прямоугольное окно', 'окно Хэмминга'
             Название применяемого при построении спектрограммы окна.
    window_offset_step : int
             Шаг смещения окна (задается в отсчетах).
    window_opt_len : int, default: 8
             Ширина окна (задается в отсчетах).

    Returns
    -------
    Функция создает график и ничего не возвращает.

    Raises
    ------
    ValueError
        Если в функцию было передано неверное название окна или размерность шага смещения окна.
    """
    if window_name == 'Прямоугольное окно':
        window = np.ones(window_opt_len)
    elif window_name == 'Окно Хэмминга':
        window = np.hamming(window_opt_len)
    else:
        raise ValueError("Ошибка ввода. Некорректное название применяемого окна " +
                         "(допускается: 'Прямоугольное окно' или 'Окно Хэмминга')!")

    noverlap = window_opt_len - window_offset_step # количество перекрывающихся точек соседних окон
    if noverlap < 0:
        raise ValueError("Ошибка ввода. Ширина окна (window_opt_len) должна быть " + 
                         "меньше шага смещения окна (window_offset_step)!")

    plt.figure()
    plt.subplot(211)
    plt.title(window_name + ' с размером (в отсчетах), равным ' + str(window_opt_len) + '.' +
              '\nШаг смещения окна (в отсчетах) равен ' + str(window_offset_step) + '.')
    # Функция построения спектрограммы
    plt.specgram(signal, 
                 NFFT=window_opt_len,  # количество точек, используемых для вычисления БПФ
                 window=window,  # оконная функция, применяемая к каждому сегменту сигнала перед выполнением БПФ
                 noverlap=noverlap) # количество точек перекрытия между соседними окнами
    plt.xlabel('Sample')
    plt.ylabel('Normalized Frequency')
    plt.show()


def plot_scalogramm(signal: np.ndarray, t_d: float):
    """
    Функция позволяет построить скейлограмму сигнала.

    Parametrs
    ---------
    signal : array_like
          Отсчеты сигнала во временной области.
    t_d : float
          Период дискритизации сигнала, с.

    Returns
    -------
    Функция создает график и ничего не возвращает.
    """
    # Функция вычисления непрерывного вейвлет-преобразования
    # cwtmatr - матрица вейвлет-коэффициентов, полученная в результате преобразования
    cwtmatr, _ = pywt.cwt(signal,   # сигнал, который хотим преобразовать
                          np.arange(1,128), # массив масштабов, используемый для преобразования
                              'morl',  # тип вейвлета, используемый для преобразования (вейвлет Морле)
                              sampling_period=t_d)  # период дискретизации
    plt.figure()
    plt.title('Скалограмма')
    plt.imshow(cwtmatr,  # матрица вейвлет-коэффициентов, полученная в результате преобразования
               extent=[-0, len(signal), 1, 128],  # границы осей для отображаемого изображения
               aspect='auto',  # управление соотношением сторон изображения
               vmax=abs(cwtmatr).max(),  # максимальное значение для нормализации
               vmin=-abs(cwtmatr).max())  # минимальное значение для нормализации
    plt.xlabel('Отсчёты')
    plt.ylabel('Шкала')
    plt.tight_layout() # автоматическая настройка отступов  
    plt.show()