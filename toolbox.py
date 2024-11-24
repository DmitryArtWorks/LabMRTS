import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.fft import fft, fftshift, ifft
from scipy.signal import butter, lfilter, cheby2, cheby1, freqz
import pywt # С этим модулем могут быть проблемы (может быть не установлен). 
            # Лечение: pip install pywavelets


# Отображение сигналов
def plot_signal(signal_args):
    """
    Вывести действительную часть сигнала, в случае необходимости вывода
    нескольких линий на один график, они передаются в качестве второй
    размерности массива, типа: [[`signal_args` первой линии], 
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
    TypeError
        Если в функцию было передано неверное число аргументов.
    """

    lines = ["-", "--", "-.", ":"] # Список возможных стилей линий графиков
    leg = []
    plt.figure()
    for i in range (len(signal_args)):
        markers = {'-r', '--g', ':b'}
        args_len = len(signal_args[i])
        if (args_len > 3 or args_len < 1):
            raise TypeError('Ошибка ввода. Неверное число аргументов (допускается от 1 до 3 аргументов)')
        if (args_len == 1):
            signal = signal_args[i][0]
            x_axis = range(0, len(signal))
        if (args_len == 2 or args_len == 3):
            signal = signal_args[i][1]
            if (type(signal_args[i][0]) == float):
                x_axis = np.arange(0, signal_args[i][0]*(len(signal)), signal_args[i][0])
            else:
                x_axis = signal_args[i][1]                
    
        # Построение графика
        
        plt.title("Name") # заголовок
        plt.xlabel("t, мкc") # ось абсцисс
        
        plt.plot(x_axis*1e6, np.real(signal), linestyle=lines[i])  # построение графика
        
        if (args_len == 3):
            leg.append(signal_args[i][2])
        else:
            leg.append('Unnamed signal ' + str(i)) # Если не нравится, можно заменить 
                                                # содержимое скобок на " ". Работает
                                                # тоже красиво
        
    plt.legend(leg)
    plt.title('Исходный сигнал')
    plt.grid()
    plt.show()


# Отображение спектр сигнала
def plot_spectum(signal_args):
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
    TypeError
        Если в функцию было передано неверное число аргументов.
    """

    t_d_us = signal_args[0][0]*1e6
    plt.figure()

    for i in range(len(signal_args)):
        args_len = len(signal_args[i])
        if (args_len > 3 or args_len < 2):
            raise TypeError('Ошибка ввода. Неверное число аргументов (допускается 2 или 3 аргумента)')
        signal = signal_args[i][1]
        n_signal = signal.size
        t_window = n_signal*t_d_us
        f_step_mhz = 1/t_window
        # Формируем точки оси абсцисс графика
        if n_signal % 2 == 0:
            f_axis = np.arange(0, (n_signal+1)*f_step_mhz/2, f_step_mhz)
            tmp_axis = np.flip(f_axis[2:]) * (-1) 
            f_axis = np.append(tmp_axis, f_axis)
        else:
            f_axis = np.arange(0, (n_signal)*f_step_mhz/2, f_step_mhz)
            tmp_axis = np.flip(f_axis[1:]) * (-1)
            f_axis = np.append(tmp_axis, f_axis)
        
        # БПФ входных отсчетов
        signal_spectrum = 1/n_signal*fft(signal, n_signal)
        
        # Название линий
        if (len(signal_args[i]) > 2):
            line_label = str(signal_args[i][2])
        else:
            line_label = 'Unnamed spectrum ' + str(i)
            
        plt.plot(f_axis, fftshift(abs(signal_spectrum)), label=line_label)  # построение графика
    plt.title("Спектр") # заголовок
    plt.xlabel("Частота, МГц") # ось абсцисс        
    plt.legend()
    plt.grid()
    plt.show()


# Отображение спектрограммы
def plot_spectrograms(data):
    plt.figure()
    plt.subplot(311)

    #большое количество точек для БПФ
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data, NFFT=1024, Fs=None, Fc=None, detrend=None, window=None, 
             noverlap=None, cmap=None, xextent=None, pad_to=None, sides=None, 
             scale_by_freq=None, mode=None, scale=None, vmin=None, vmax=None)

    plt.xlabel('Sample')
    plt.ylabel('Normalized Frequency')
    plt.subplot(312)

    #стандартный шаг
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data, NFFT=None, Fs=None, Fc=None, detrend=None, window=None, 
             noverlap=None, cmap=None, xextent=None, pad_to=None, sides=None, 
             scale_by_freq=None, mode=None, scale=None, vmin=None, vmax=None)

    plt.xlabel('Sample')
    plt.ylabel('Normalized Frequency')
    plt.subplot(313)

    #маленький шаг
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data, NFFT=None, Fs=None, Fc=None, detrend=None, window=None, 
             noverlap=0, cmap=None, xextent=None, pad_to=None, sides=None, 
             scale_by_freq=None, mode=None, scale=None, vmin=None, vmax=None)

    plt.xlabel('Sample')
    plt.ylabel('Normalized Frequency')
    plt.show()


# Отображение карты вейвлет-коэффициентов
def plot_swt(data, t_d):
    plt.figure()
    coef, freqs = pywt.cwt(data, np.arange(1, 20), 'morl',
                       sampling_period=t_d)
    plt.pcolor(range(1, 6401), freqs, coef)
    plt.show()


# Вывод ИХ фильтра по его коэффициентам
def impz(b, a, name):
    impulse = [0]*100
    impulse[0] =1.
    x = range(100)
    response = lfilter(b, a, impulse)
    plt.stem(x, response)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(name)
    plt.show()


# Вывод АЧХ и ФЧХ фильтра по его коэффициентам
def mfreqz(b, a):
    w,h = freqz(b, a)
    h_dB = 20 * np.log10(abs(h))
    plt.subplot(211)
    plt.plot(w/max(w), h_dB)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')
    plt.grid('on')
    plt.ylim(bottom=-30)
    plt.subplot(212)
    h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
    plt.plot(w/max(w), h_Phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.grid('on')
    plt.subplots_adjust(hspace=1.5)
    plt.show()


# Вывод АЧХ и ФЧХ фильтра по его коэффициентам
def mfreqz3(b, a, names, lims=[0,1]):
    lines = ["-","--","-.",":"]
    plt.subplot(211)
    for i in range(3):
        w, h = freqz(b[i], a[i])
        h_dB = 20 * np.log10(abs(h))
        plt.plot(w/max(w), h_dB, linestyle=lines[i])
        plt.legend(names)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')
    plt.grid('on')
    plt.ylim(top=1, bottom=-30)
    plt.xlim(lims[0], lims[1])
    
    plt.subplot(212)
    for i in range(3):
        w,h = freqz(b[i], a[i])
        h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
        plt.plot(w/max(w), h_Phase, linestyle=lines[i])
        #plt.legend(names)
    plt.xlim(lims[0], lims[1])
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.grid('on')
    plt.subplots_adjust(hspace=0.5)
    plt.show()


""" Генерация импульсов 
Функции генерации последовательности импульсов и одиночного импульса
"""


# Формирование последовательности чипов
def generate_sequence(sig_type, t_d, n_chips, t_imp, f_low):
#Ф-я формирования набора последовательности сигналов перестраиваемых по
#частоте
# Аргументы:
#   sig_type - тип сигнала
#   t_d - интервал дискретизации
#   n_chips - число отсетов
#   f_low - нижняя частота
#   запустить функцию формирования сигнала в зависимости от заданного типа
    res_sig = np.array([0])
    sig = []
    n_cnts_sig = math.floor(t_imp/t_d)
    if (sig_type == 'chirp'): # тип в/импульс
        f_mod = f_low/5
        for i in range(0, n_chips):
            sig = [0] * n_cnts_sig
            sig[:n_cnts_sig] = get_chirp_pulse(t_d, t_imp, 0.96*t_imp, f_low, f_mod)
            res_sig = np.append(res_sig, sig)
    elif (sig_type == 'radio'): # тип р/импульс
        
        # случайная перестройка для р/импульса
        for i in range(0, n_chips):
            sig = [0] * n_cnts_sig
            random_freq = f_low + np.random.randint(5, size=(1))*8/t_imp
            sig[:n_cnts_sig] = get_radio_pulse(t_d, t_imp, 0.96*t_imp, random_freq)
            res_sig = np.append(res_sig, sig)
        
    elif (sig_type == 'AM'):
        f_mod = f_low/10
        for i in range(0, n_chips):
            sig = [0] * n_cnts_sig
            random_freq = f_low + np.random.randint(5, size=(1))*(3*f_mod)
            sig[:n_cnts_sig] = get_AM(t_d, t_imp, random_freq, f_mod, 0.5)
            res_sig = np.append(res_sig, sig)
    res_sig = np.delete(res_sig, 0)
    
    return res_sig


# Формирование сигналов (управляющая функция)
def generate_single_chip(signal_type, *signal_data): # запустить функцию формирования сигнала в зависимости от заданного типа
    if (signal_type == 'video'):    # тип в/импульс
        signal = get_video_pulse(*signal_data)
    elif (signal_type == 'radio'):    # тип р/импульс
        signal = get_radio_pulse(*signal_data)
    elif (signal_type == 'AM'):    # тип АМ сигнал  
        signal = get_AM(*signal_data)
    elif (signal_type == 'chirp'):
        signal = get_chirp_pulse(*signal_data)
    return signal


# Функции формирования чипа
# Формирование в/импульса
def get_video_pulse(t_d, t_window, t_imp):
# Ф-я формировнаия в/импульса
# Аргументы:    
#   % t_d - период дискретизации, с t_window - период рассмотрения, с
#   t_imp - длительность импульса, с
    n_signal = math.floor(t_window/t_d)  # число отсчетов в рассматриваемом интерв.
    n_imp = math.floor(t_imp/t_d)        # число отсчетов импульса
    signal = n_signal * [0]            # сформировать пустую выборку
    for i in range(0, n_imp):     # поставить первые n_imp отсчетов
        signal[i] = 1
    return signal


# Формирование р/импульса
def get_radio_pulse(t_d, t_window, t_imp, f_carrier_hz):
# Ф-я формирования р/импульса
# Аргументы:
#   t_d - интервал дискретизации, с    t_window - интервал рассмотрения, с
#   t_imp - длительность импульса, с   f_carrier_hz - частота несущей, Гц
    n_signal = math.floor(t_window/t_d) # число отсчетов в рассматриваемом интерв.
    n_imp = math.floor(t_imp/t_d)       # число отсчетов импульса
    pulse_amp = np.zeros(n_signal)
    for i in range(1, n_imp):     # сформировать огиб.
        pulse_amp[i] = 1
    t_axis = np.linspace(0, t_window, n_signal)   # формирование временной оси
    carrier = np.sin(2*math.pi*f_carrier_hz*t_axis)    # сформировать несущую
    signal = pulse_amp*carrier    # несущая на амплитудные значения
    return signal


# Формирование АМ/сигнала
def get_AM(t_d, t_window, f_carrire_hz, f_mod_hz, i_mod):
# Ф-я формирования АМ сигнала
#   t_d - интервал дискретизации, с    t_window - интервал рассмотрения, с
#   f_carrier_hz - частота несущей, Гц
#   f_mod_hz - частота модуляции, Гц   i_mod - глубина модуляции (Amin/Amax)
    n_signal = math.floor(t_window/t_d)                 # число отсчетов в интервале рассмот.
    t_axis = np.linspace(0, t_window, n_signal)       # ось времени
    am_mult = 1/(2/i_mod) 
    am_shift = 1-am_mult    # множитель для расчета огибающей АМ
    a_modulation = np.sin(2*math.pi*f_mod_hz*t_axis)*am_mult+am_shift  # огибающая
    signal = np.sin(2*math.pi*f_carrire_hz*t_axis) * a_modulation     # формирование АМ сигнала
    return signal


# Формирование ЛЧМ
def get_chirp_pulse(t_d, t_window, t_imp, f_start_hz, f_chirp_hz):
# Ф-я формирования р/импульса
# Аргументы:
#   t_d - интервал дискретизации, с    t_window - интервал рассмотрения, с
#   t_imp - длительность импульса, с   f_center_hz - серединная частота, Гц
#   f_chirp_hz - ширина полосы, Гц
    n_signal = math.floor(t_window/t_d)     # число отсчетов в рассматриваемом интерв.
    n_imp = math.floor(t_imp/t_d)           # число отсчетов импульса
    t_axis = np.linspace(0, t_imp, n_imp)   # формирование временной оси
    chirp_band = f_chirp_hz/t_imp
    chirp_sin_arg = f_start_hz + chirp_band/2*t_axis
    signal = [0] * n_signal
    signal[1:n_imp+1] = np.sin(2*math.pi*chirp_sin_arg*t_axis)     # сформировать несущую
#     signal(n_imp:end) =  sin(2*pi*f_moment(n_imp).*t_axis(n_imp:end))
    return signal


""" Фильтры 
"""


# Применить к сигналу фильтр с "идеальной" АЧХ
def apply_ideal_filter(t_d, filter_type, f_cut_hz, signal_in):
    """
    Применить к сигналу один из фильтров (ФНЧ, ФВЧ, ЗФ, ПФ) с 
    идеальной АЧХ (1 - в полосе пропускания, 0 - в полосе подавления, 
    переходной полосы нет). Принцип работы следующий: формируется
    идеальная АЧХ фильтра в области положительных частот, а затем
    с помощью `np.flip()` и конкатенации получается АЧХ для всего
    диапазона частот (и положительных, и отрицательных). Для получения
    выходного сигнала производится перемножение АЧХ фильтра с АЧХ входного
    сигнала.
    
    Parameters
    ----------       
    t_d : scalar
        Шаг дискретизации. 
    filter_type : 'LP', 'HP', 'BP', 'BS'
        Тип фильтра: фильтр нижних частот, ФНЧ - `'LP'`; фильтр верхних частот,
        ФВЧ - `'HP'`, полосовой фильтр, ПФ - `'BP'`, заградительный фильтр, ЗФ - 
        `'BS'`.
    f_cut_hz : scalar or list
        Одна частота среза для ФНЧ и ФВЧ или две частоты среза для ПФ и ЗФ. 
        Если передать list с числом элементов более 2, все элементы, кроме
        первых двух, будут проигнорированы.
    signal_in : array_like (real)
        Отсчеты входного действительного сигнала.  
    
    Returns
    -------
    Функция возвращает отсчеты отфильтрованного идеальным `filter_type` 
    фильтром сигнала `signal_in`.

    Raises
    ------
    TypeError
        Если в функцию была передано неправильное название фильтра
        или граничные частоты фильтра. Или выходной сигнал получился
        комплексным.
    
    ValueError
        Если в случае фильтра с двумя частотами среза список частот
        не был упорядочен в порядке возрастания.          
    """

    f_d = 1/t_d    # вычисление частоты дискретизации (обратно пропорционально интервалу дискретизации)
    n_signal = len(signal_in) # определение числа отсчетов
    f_axis = np.linspace(0, f_d/2, int(np.floor(n_signal / 2))) # половина набора отсчетов оси абсцисс АЧХ.
                                                                # Вторая половина будет получена с помощью
                                                                # np.flip().
    filter_f_char = np.ones(int(np.ceil((n_signal+1)/2)))   # половина набора отсчетов оси ординат АЧХ.
                                                            # Вторая половина будет получена с помощью
                                                            # np.flip().
    
    if isinstance(f_cut_hz, float or int): # Если одно значение частоты среза, то это будет или ФНЧ, или ФВЧ.
           
        f_cut_i = np.argmax(f_axis > f_cut_hz)  # Выражение f_axis > f_cut_hz дает логический массив, элементами
                                                # которого являются результат сравнения значения f_axis с f_cut_hz
                                                # np.argmax() находит индекс первого максимального значения, который
                                                # будет соответствовать номеру элемента, в котором расположилась 
                                                # частота среза. Такая, на первый взгляд, сложность обусловлена тем, 
                                                # что число отсчетов входного сигнала не является постоянным, как и
                                                # частота среза.
        if (filter_type == 'HP'): 
            filter_f_char[:f_cut_i] = 0     # Если необходимо сформировать ФВЧ, то все отсчеты АЧХ, сответствующие
                                            # частотам ниже частоты среза должны быть равны нулю.
        elif (filter_type == 'LP'): 
            filter_f_char[f_cut_i:] = 0     # Если необходимо сформировать ФНЧ, то все отсчеты АЧХ, соответствующие  
                                            # частотам выше частоты среза должны быть равны нулю.
        else:   
            raise TypeError('Ошибка вввода: Тип фильтра')   # Вызов исключения, если был передан неизвестный
                                                            # аргумент.

        # формирование АЧХ для двухстороннего спектра со второго с начала отсчета.
        tmp_filter_f_char = np.flip(filter_f_char[1:])  # "Переворот" (но не гражданский) массива АЧХ, поскольку
                                                        # АЧХ в отрицательной области "зеркальна" АЧХ в положительной.
                                                        # Опускаем 0-й элемент, поскольку в противном случае получится
                                                        # два отсчета, отвечающих за значение АЧХ на нулевой частоте.
        if (n_signal % 2 == 0): 
            tmp_filter_f_char = tmp_filter_f_char[:len(tmp_filter_f_char) - 1]  # Если количество отсчетов сигнала четное, то
                                                                                # необходимо исключить последний отсчет АЧХ, т.к
                                                                                # он лишний, а появился из-за выбранного способа
                                                                                # формирования массива filter_f_char (мы округляли)
                                                                                # "вверх" число элементов этого массива.
        filter_f_char = np.append(filter_f_char, tmp_filter_f_char) # Конкатенация отрицательной и положительной 
                                                                    # частей АЧХ в один массив.
    
    elif isinstance(f_cut_hz, list):    # Если два значения частоты среза, то ПФ или ЗФ
        
        if f_cut_hz[0] > f_cut_hz[1]:   # Для корректной работы необходимо, чтобы нулевой 
                                        # элемент массива частот был меньше первого.
            raise ValueError('Ошибка: f_cut_hz[0] должна быть меньше f_cut_hz[1]')
        
        f_cut_0 = np.argmax(f_axis > f_cut_hz[0])   # Принцип тот же, как и в ветке с одной частотой среза. 
        f_cut_1 = np.argmax(f_axis > f_cut_hz[1])   # Однако поскольку у данных фильтров две частоты среза, 
                                                    # необходимо найти индексы, соответствующие этим двум
                                                    # частотам. 
        if (filter_type == 'BP'):
            filter_f_char[:f_cut_0] = 0 # Если необходимо сформировать АЧХ ПФ, то все отсчеты, соответствующие
            filter_f_char[f_cut_1:] = 0 # частотам ниже f_cut_0 и выше f_cut_1 должны равняться нулю.
        elif (filter_type == 'BS'): 
            filter_f_char[f_cut_0:f_cut_1] = 0  # Если необходимо сформировать АЧХ ПФ, то все отсчеты, соответствующие
                                                # интервалу частот f_cut_0...f_cut_1 должны равняться нулю.
        tmp_filter_f_char = np.flip(filter_f_char[1:])  # "Переворот" (но не гражданский) массива АЧХ, поскольку
                                                        # АЧХ в отрицательной области "зеркальна" АЧХ в положительной.
                                                        # Опускаем 0-й элемент, поскольку в противном случае получится
                                                        # два отсчета, отвечающих за значение АЧХ на нулевой частоте.
        if (n_signal % 2 == 0):
            tmp_filter_f_char = tmp_filter_f_char[:len(tmp_filter_f_char) - 1]  # Если количество отсчетов сигнала четное, то
                                                                                # необходимо исключить последний отсчет АЧХ, т.к
                                                                                # он лишний, а появился из-за выбранного способа
                                                                                # формирования массива filter_f_char (мы округляли)
                                                                                # "вверх" число элементов этого массива.
        filter_f_char = np.append(filter_f_char, tmp_filter_f_char) # Конкатенация отрицательной и положительной 
                                                                    # частей АЧХ в один массив.
    else:
        raise TypeError('Ошибка ввода: Граничные частоты фильтра')  # В эту ветку код свалится, если f_cut_hz имело
                                                                    # тип данных, который не ожидался (в идеале ожидается)
                                                                    # либо скаляр (int/float) или list из 2 элементов
    # Фильтрация входного сигнала
    signal_in_sp = fft(signal_in, n_signal) # АФЧХ входного сигнала
    signal_out_sp = signal_in_sp * filter_f_char    # АФЧХ выходного сигнала (произведение)
                                                    # АФЧХ выходного сигнала на АФЧХ фильтра
    signal_out = ifft(signal_out_sp, n_signal)  # Выходной сигнал во временно`й области
    
    # Данная ветка будет вызвана, если по какой-то причине выходной сигнал 
    # оказался комплексным (должен быть действительным, поскольку на входе)
    # он действительный
    if (np.prod(np.iscomplex(signal_out))): 
        raise TypeError('Допущена ошибка при формировании фильтра: сигнал после фильтрации комплексный')  
    return signal_out


# Применить к сигналу фильтр Баттерворта
def apply_butt_filter(t_d, filter_type, f_cut_hz, filter_order, signal_in):
# Ф-я фильтрации фильтром с АЧХ Баттерворта
# Аргументы:
#   t_d - интервал дискретизации,с 
#   filter_type: 'LP' - НЧ, 'HP' - ВЧ, 'BP' - полосовой, 'S' - заградительный
#   f_cut_hz - частота среза, Гц
#   filter_order - порядок фильтра
#   signal_in - входной сигнал (вектор строка)
    b, a = create_butt_filter(t_d, filter_type, f_cut_hz, filter_order)  # формирование коэффициентов фильтра Баттерворта
    
    signal_out = lfilter(b, a, signal_in)     # фильтрация сигнала
    return signal_out


# Применить к сигналу фильтр Чебышёва
def apply_cheb2_filter(t_d, filter_type, f_cut_hz, filter_order, bnd_att_db, signal_in):
# Ф-я фильтрации фильтром с АЧХ Чебышева
# Аргументы:
#   t_d - интервал дискретизации,с 
#   filter_type: 'LP' - НЧ, 'HP' - ВЧ, 'BP' - полосовой, 'S' - заградительный
#   f_cut_hz - частота среза, Гц
#   filter_order - порядок фильтра
#   bnd_att_db - подавление боковых лепестков АЧХ
#   signal_in - входной сигнал (вектор строка)
    b, a = create_cheb2_filter(t_d, filter_type, f_cut_hz, filter_order, bnd_att_db)  # формирование фильтра
    
    signal_out = lfilter(b, a, signal_in)                 # фильтрация сигнала
    return signal_out


# Расчет коэффициентов фильтра Баттерворта
def create_butt_filter(t_d, filter_type, f_cut_hz, filter_order):
    # Ф-я расчета коэффициентов передаточной функции фильтра Баттерворта
    # Аргументы:
    #   t_d - интервал дискретизации,с 
    #   filter_type: 'LP' - НЧ, 'HP' - ВЧ, 'BP' - полосовой, 'S' - заградительный
    #   f_cut_hz - частота среза, Гц
    #   filter_order - порядок фильтра
    #   signal_in - входной сигнал (вектор строка)
    if (isinstance(f_cut_hz, float or int)):      # если одно значение - ВЧ или НЧ фильтр
        if (filter_type == 'LP'):
            f_name = 'lowpass'
        elif (filter_type == 'HP'):
            f_name = 'highpass'
        else:
            print('Ошибка вввода: Тип фильтра')
    elif (isinstance(f_cut_hz, list)):
        if (filter_type == 'BP'):
            f_name = 'bandpass'
        elif (filter_type == 'S'):
            f_name = 'bandstop'
        else:
            print('Ошибка вввода: Тип фильтра')

    else:
        raise TypeError('Ошибка ввода: Граничные частоты фильтра')
    
    f_d = 1/t_d            # частота дискретизации
    w_n = np.array(f_cut_hz)/(f_d/2) # нормированная частота среза
    b, a = butter(filter_order, w_n, f_name)  # формирование коэффициентов фильтра Баттерворта
    return b, a


# Расчет коэффициентов Фильтра Чебышева
def create_cheb2_filter(t_d, filter_type, f_cut_hz, filter_order, bnd_att_db):
# Ф-я расчета коэффициентов передаточной характеристики фильтр с АЧХ Чебышева
# Аргументы:
#   t_d - интервал дискретизации,с 
#   filter_type: 'LP' - НЧ, 'HP' - ВЧ, 'BP' - полосовой, 'S' - заградительный
#   f_cut_hz - частота среза, Гц
#   filter_order - порядок фильтра
#   bnd_att_db - подавление боковых лепестков АЧХ
#   signal_in - входной сигнал (вектор строка)
    if isinstance(f_cut_hz, float or int):      # если передано одно зачение частоты среза
        if (filter_type == 'LP'):
            f_name = 'lowpass'           # НЧ фильтр
        elif (filter_type == 'HP'):
            f_name = 'highpass'          # ВЧ фильтр
        else:
            print('Ошибка вввода: Тип фильтра') # если передано два значения частот среза
    elif (isinstance(f_cut_hz, list)):
        if (filter_type == 'BP'):
            f_name = 'bandpass'                 # полосовой фильтр
        elif (filter_type == 'S'):
            f_name = 'bandstop'                 # заградительный фильтр
        else:
            print('Ошибка вввода: Тип фильтра')

    else:
        raise TypeError('Ошибка ввода: Граничные частоты фильтра')

    f_d = 1/t_d            # частота дискретизации, Гц
    w_n = np.array(f_cut_hz)/(f_d/2) # нормированная частота среза (w/w_d)
    b, a = cheby2(filter_order, bnd_att_db, w_n, f_name)  # формирование фильтра
    return b, a


def create_cheb1_filter(t_d, filter_type, f_cut_hz, filter_order, bnd_att_db):
# Ф-я расчета коэффициентов передаточной характеристики фильтр с АЧХ Чебышева
# Аргументы:
#   t_d - интервал дискретизации,с 
#   filter_type: 'LP' - НЧ, 'HP' - ВЧ, 'BP' - полосовой, 'S' - заградительный
#   f_cut_hz - частота среза, Гц
#   filter_order - порядок фильтра
#   bnd_att_db - подавление боковых лепестков АЧХ
#   signal_in - входной сигнал (вектор строка)
    if isinstance(f_cut_hz, float):      # если передано одно зачение частоты среза
        if (filter_type == 'LP'):
            f_name = 'lowpass'           # НЧ фильтр
        elif (filter_type == 'HP'):
            f_name = 'highpass'          # ВЧ фильтр
        else:
            print('Ошибка вввода: Тип фильтра') # если передано два значения частот среза
    elif (isinstance(f_cut_hz, list)):
        if (filter_type == 'BP'):
            f_name = 'bandpass'                 # полосовой фильтр
        elif (filter_type == 'S'):
            f_name = 'bandstop'                 # заградительный фильтр
        else:
            print('Ошибка вввода: Тип фильтра')

    else:
        raise TypeError('Ошибка ввода: Граничные частоты фильтра')
        
    f_d = 1/t_d            # частота дискретизации, Гц
    w_n = np.array(f_cut_hz)/(f_d/2) # нормированная частота среза (w/w_d)
    b, a = cheby1(filter_order, bnd_att_db, w_n, f_name)  # формирование фильтра
    return b, a

