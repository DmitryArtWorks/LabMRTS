import numpy as np
from numpy.random import normal
from numpy.fft import fft
from numpy.fft import ifft


def apply_corr_filter(sig, time_window: int | float, corr_us: int | float, n_var: int):
   """
   Функция позволяет получить коррелированные отсчёты процесса при условии, 
   что на вход подается некоррелированная числовая последовательность.

   Parametrs
   ---------
   sig : array_like
         Отсчёты некоррелированного процесса.
   time_window : int, float
         Длительность числовой последовательности, мкс.
   corr_us : int, float
         Длительность интервала корреляции, мкс.
   n_var : int
         Номер варианта.

   Returns
   -------
   Функция возвращает отсчёты коррелированного случайного процесса.

   Raises
   ------
   ValueError
         Если в функцию было передано неверное значение варианта.
   """
   signal_length = sig.shape[1]
   x = np.linspace(-time_window/2, time_window/2, signal_length)

   # Целевая корреляционная функция
   if n_var == 0:
      corr_func = np.exp(-abs(x) / corr_us) ** 2
   elif n_var == 1:
      corr_func = np.exp(-(x/corr_us)**2)
   elif n_var == 2:
      corr_func = (1 - abs(x)/corr_us) * (np.abs(x) <= corr_us)
   elif n_var == 3:
      corr_func = np.cos(10*x/corr_us) * np.exp(-1/2*(x/corr_us)**2)
   elif n_var == 4:
      corr_func = 1 / (1 + (x/corr_us)**2)
   else:
      raise ValueError("Неправильно задан номер варианта!!!")

   filter_freq_resp = np.sqrt(np.fft.fft(corr_func, signal_length))    # АЧХ фильтра (вычисляется с помощью БПФ от КФ)
   spectrum_corr_sig = fft(sig, signal_length) * filter_freq_resp    # спектр процесса на выходе фильтра
   # Получение отсчетов коррелированого процесса (с помощью ОБПФ)
   corr_sig = np.real(np.sqrt(2)*ifft(spectrum_corr_sig, signal_length))

   return corr_sig


def calculate_correlate(proc, n_counts: int, n_samples: int):
    """
    Функция позволяет рассчитать усредненную нормированную корреляционную функцию по заданному количеству реализаций.

    Parametrs
    ---------
    proc : array_like
        Отсчёты случайного процесса.
    n_counts : int
        Число отсчётов.
    n_samples : int
        Количество усредняемых реализаций.

    Returns
    -------
    tuple : Кортеж, содержащий массив отсчетов КФ одной реализации (r_sample) 
            и массив отсчетов КФ, усредненной по заданному количеству реализаций (r_est).
    """
    # Расчет нормированной КФ одной реализации
    r_sample = np.correlate(proc[1,:], proc[1,:], mode='full') / np.var(proc[1,:]) / n_counts

    # Расчет усредненной нормированной КФ одной реализации
    r_est = np.zeros((n_samples, 2*n_counts-1))
    for i in range(n_samples):
        r_est[i,:] = np.correlate(proc[i,:], proc[i,:], mode='full') / np.var(proc[i,:]) / n_counts
    r_est = np.mean(r_est, axis=0)    # усреднение КФ по множеству реализаций

    return r_sample, r_est


def get_corr_func(corr_int: int | float, time_window: int | float, n_counts: int, n_var: int):
    """
    Функция позволяет получить отсчёты теоретической корреляционной функции.

    Parametrs
    ---------
    corr_int : int, float
            Длительность интервала корреляции, мкс.
    time_window : int, float
            Временной интервал, на котором рассматривается корреляция, мкс.
    n_counts : int
            Число отсчётов.
    n_var : int
            Номер варианта.

    Returns
    -------
    Функция возвращает отсчёты теоретической коррелированной функции.

    Raises
    ------
    ValueError
            Если в функцию было передано неверное значение варианта.
    """
    shift_axis_us = np.linspace(-time_window, time_window, 2*n_counts-1)    # ось отстройки КФ

    # Теоретическая КФ в зависимости от варинта
    if n_var == 0:
        r_fun = np.exp(-abs(shift_axis_us) / (corr_int / 2))
    elif n_var == 1:
        r_fun = np.exp(-(shift_axis_us/corr_int)**2)
    elif n_var == 2:
        r_fun = (1 - abs(shift_axis_us)/corr_int) * (np.abs(shift_axis_us) <= corr_int)
    elif n_var == 3:
        r_fun = np.cos(10*shift_axis_us/corr_int) * np.exp(-1/2*(shift_axis_us/corr_int)**2)
    elif n_var == 4:
        r_fun = 1 / (1 + (shift_axis_us/corr_int)**2)
    else:
      raise ValueError("Неправильно задан номер варианта!!!")
    
    return r_fun