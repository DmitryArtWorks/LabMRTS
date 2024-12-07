import math
import numpy as np


# # # # # # # # # # # # # # # # # # # # # #
# Формирование последовательности чипов*  #
# (функция-обертка)                       #
# # # # # # # # # # # # # # # # # # # # # #

# *чип - один период модулирующего сигнала (в рамках данной ЛР)


def generate_sequence(sig_type, t_d, n_chips, t_imp, f_low):
    """
    Сформировать отсчеты последовательности одного из следующих сигналов:
    радиоимпульс `radio`, АМ сигнал `AM`, ЛЧМ сигнал `chirp`. Функция 
    представляет собой обертку вокруг функций, которые непосредственно 
    формируют сигналы. В отличие от generate_single_chip() не создает 
    видеоимпульс, обладает фиксированным набором входных параметров и 
    создает отсчеты для нескольких периодов сигнала.
    
    Parameters
    ----------       
    signal_type : 'radio', 'AM', 'chirp'
        Вид сигнала, отсчеты которого необходимо сформировать.
    t_d : scalar
        Интервал дискретизации сигнала.
    n_chips : scalar
        Число периодов сигнала, которые необходимо отобразить.
    t_imp : scalar
        Длительность огибающей импульса.
    f_low : scalar
        Минимальная частотаю Параметр крайне условный, поскольку
        для каждого сигнала этот затем преобразуется по-разному. 

    Returns
    -------
    signal : array_like
        Набор отсчетов сигнала требуемого типа.
    
    Raises
    -------
    ValueError
        Если в аргумент `signal_type `был передан неверный тип сигнала.

    """
    n_cnts_sig = math.floor(t_imp/t_d)  # Определение числа отсчетов, приходящихся
                                        # на один из n_chips периодов сигнала.

    res_sig = np.array([0]) # Инициализация выходного массива

    sig = [0] * n_cnts_sig  # инициализация внутренней переменной, необходимой для
                            # работы выбранного алгоритма формирования последовательности
                            # импульсов.
    
    # Выбор вызываемой функции в зависимости от значения sig_type
    if (sig_type == 'chirp'): # тип ЛЧМ импульс
        f_mod = f_low/5
        for _ in range(0, n_chips): # Цикл по числу периодов
            sig[:n_cnts_sig] = get_chirp_pulse(t_d, t_imp, 0.96*t_imp, f_low, f_mod)    # формирование отсчетов для одного периода сигнала.
                                                                                        # `t_window` = t_imp, `t_imp` = 0.96*t_imp, чтобы 
                                                                                        # импульс был равен ПОЧТИ всему времени наблюдения.
                                                                                        # Это нужно, чтобы импульсы были визуально различимы 
                                                                                        # на графике.
            res_sig = np.append(res_sig, sig)   # конкатенация очередного периода к
                                                # коцу массива res_sig

    elif (sig_type == 'radio'): # тип р/импульс
        for _ in range(0, n_chips): # Цикл по числу периодов
            random_freq = f_low + np.random.randint(5, size=(1))*8/t_imp # случайная частота несущей р/импульса
            sig[:n_cnts_sig] = get_radio_pulse(t_d, t_imp, 0.96*t_imp, random_freq) # формирование отсчетов для одного периода сигнала.
                                                                                    # `t_window` = t_imp, `t_imp` = 0.96*t_imp, чтобы 
                                                                                    # импульс был равен ПОЧТИ всему времени наблюдения.
                                                                                    # Это нужно, чтобы импульсы были визуально различимы 
                                                                                    # на графике.
            res_sig = np.append(res_sig, sig)   # конкатенация очередного периода к
                                                # коцу массива res_sig
        
    elif (sig_type == 'AM'): # АМ сигнал
        f_mod = f_low/10
        for _ in range(0, n_chips): # Цикл по числу периодов
            random_freq = f_low + np.random.randint(5, size=(1))*(3*f_mod) # случайная частота несущей АМ сигнала
            sig[:n_cnts_sig] = get_AM(t_d, t_imp, random_freq, f_mod, 0.5) # формирование отсчетов для одного периода сигнала.
            res_sig = np.append(res_sig, sig)   # конкатенация очередного периода к
                                                # коцу массива res_sig
    else:
        raise ValueError("Введён неверный тип сигнала. Допустимы значения: 'video', 'radio', 'AM', 'chirp'")
    
    res_sig = np.delete(res_sig, 0) # удаление нулевого отсчета сигнала, который использовался для инициализации массива
        
    return res_sig


# # # # # # # # # # # # # # #
# Функции формирования чипа #
# # # # # # # # # # # # # # #

# Формирование одного периода в/импульса
def get_video_pulse(t_d, t_window, t_imp):
    """
    Сформировать отсчеты видеоимпульса единичной амплитуды. 
    Он будет помещен в самом начале "временной" оси.
    
    Parameters
    ----------       
    t_d : scalar
        Интервал дискретизации сигнала.
    t_window : scalar
        Длительность "окна", в котором наблюдается сигнал.
    t_imp : scalar
        Длительность создаваемого импульса.
    
    Returns
    -------
    signal : array_like
        Набор отсчетов видеоимпульса.
    """

    n_signal = math.floor(t_window/t_d)  # число отсчетов в рассматриваемом интерв.
    n_imp = math.floor(t_imp/t_d)        # число отсчетов импульса
    signal = [0] * n_signal              # сформировать пустую выборку
    for i in range(0, n_imp):   # первые n_imp отсчетов имеют единичную амплитуду.
        signal[i] = 1
    return signal


# Формирование одного периода р/импульса
def get_radio_pulse(t_d, t_window, t_imp, f_carrier_hz):
    """
    Сформировать отсчеты радиоимпульса единичной амплитуды. 
    Он будет помещен в самом начале "временной" оси.
    
    Parameters
    ----------       
    t_d : scalar
        Интервал дискретизации сигнала.
    t_window : scalar
        Длительность "окна", в котором наблюдается сигнал.
    t_imp : scalar
        Длительность создаваемого импульса.
    f_carrier_hz : scalar
        Частота несущей, "заполняющей" видеосигнал.

    Returns
    -------
    signal : array_like
        Набор отсчетов радиоимпульса.
    """
    
    n_signal = math.floor(t_window/t_d) # число отсчетов в рассматриваемом интерв.
    n_imp = math.floor(t_imp/t_d)   # число отсчетов импульса
    pulse_amp = [0] * n_signal  # сформировать пустую выборку
    for i in range(1, n_imp):   # сформировать огибающую радиоимпульса
        pulse_amp[i] = 1
    t_axis = np.linspace(0, t_window, n_signal) # формирование временной оси
    carrier = np.sin(2*math.pi*f_carrier_hz*t_axis) # сформировать несущую
    signal = pulse_amp*carrier  # несущая на амплитудные значения
    return signal


# Формирование одного периода АМ-сигнала
def get_AM(t_d, t_window, f_carrier_hz, f_mod_hz, i_mod):
    """
    Сформировать отсчеты АМ-сигнала.
    
    Parameters
    ----------       
    t_d : scalar
        Интервал дискретизации сигнала.
    t_window : scalar
        Длительность "окна", в котором наблюдается сигнал.
    t_imp : scalar
        Длительность создаваемого импульса.
    f_carrier_hz : scalar
        Частота несущей, "заполняющей" видеосигнал.
    f_mod_hz : scalar
        Частота модулирующей синусоиды.
    i_mod : scalar
        Индекс (глубина) модуляции сигнала

    Returns
    -------
    signal : array_like
        Набор отсчетов АМ-сигнала.
    """

    n_signal = math.floor(t_window/t_d) # число отсчетов в интервале рассмот.
    t_axis = np.linspace(0, t_window, n_signal) # ось времени
    am_mult = 1/(2/i_mod) 
    am_shift = 1-am_mult    # множитель для расчета огибающей АМ
    a_modulation = np.sin(2*math.pi*f_mod_hz*t_axis)*am_mult+am_shift   # огибающая
    signal = np.sin(2*math.pi*f_carrier_hz*t_axis) * a_modulation   # формирование АМ-сигнала

    return signal


# Формирование одного периода ЛЧМ-сигнала
def get_chirp_pulse(t_d, t_window, t_imp, f_start_hz, f_chirp_hz):
    """
    Сформировать отсчеты ЛЧМ-сигнала (в разработке).
    
    Parameters
    ----------       
    t_d : scalar
        Интервал дискретизации сигнала.
    t_window : scalar
        Длительность "окна", в котором наблюдается сигнал.
    t_imp : scalar
        Длительность создаваемого импульса.
    f_start_hz : scalar
        Начальная (и по совместительству - несущая) частота сигнала.
    f_chirp_hz : scalar
        Частота девиации.

    Returns
    -------
    signal : array_like
        Набор отсчетов ЛЧМ-сигнала.
    """

    n_signal = math.floor(t_window/t_d) # число отсчетов в рассматриваемом интерв.
    n_imp = math.floor(t_imp/t_d)   # число отсчетов импульса
    t_axis = np.linspace(0, t_imp, n_imp)   # формирование временной оси
    chirp_band = f_chirp_hz/t_imp   # Определение полосы ЛЧМ-сигнала
    chirp_sin_arg = f_start_hz + chirp_band/2*t_axis    # Множитель аргумента в функции
                                                        # синусоиды
    signal = [0] * n_signal # Инициализация массива выходных значений требуемого
                            # размера
    signal[1:n_imp+1] = np.sin(2*math.pi*chirp_sin_arg*t_axis)  # сформировать несущую
    
    return signal