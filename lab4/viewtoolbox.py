import numpy as np
import pywt
from matplotlib import pyplot as plt


def plot_spectrogram(signal, window_name: str, window_offset_step: int, window_opt_len=8):
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


def plot_scalogramm(signal, t_d: float):
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