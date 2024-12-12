import numpy as np
from numpy.fft import fft
from numpy.fft import ifft
from typing import Callable


def apply_corr_filter(sig: np.ndarray, 
                      time_window: int | float, 
                      r_fun: Callable[[np.ndarray, int | float], np.ndarray], 
                      corr_us: int | float) -> np.ndarray:
    """
    Функция позволяет получить коррелированные отсчёты процесса при условии, 
    что на вход подается некоррелированная числовая последовательность.

    Parametrs
    ---------
    sig : array_like
            Отсчёты некоррелированного процесса.
    time_window : int, float
            Длительность числовой последовательности, мкс.
    r_fun : [[array_like, int | float], array_like]
            Корреляционная функция, заданная как функция от двух переменных.
    corr_us : int, float
            Длительность интервала корреляции, мкс.

    Returns
    -------
    Функция возвращает отсчёты коррелированного случайного процесса.
    """
    signal_length = sig.shape[1]
    x = np.linspace(-time_window/2, time_window/2, signal_length)

    corr_func_counts = r_fun(x, corr_us)    # отсчеты КФ

    filter_freq_resp = np.sqrt(fft(corr_func_counts, signal_length))    # АЧХ фильтра (вычисляется с помощью БПФ от КФ)
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
        r_est[i,:] = np.correlate(proc[i,:], proc[i,:], mode='full')
        r_est[i,:] /= np.max(r_est[i,:])    # нормировка КФ
    r_est = np.mean(r_est, axis=0)    # усреднение КФ по множеству реализаций

    return r_sample, r_est


def get_corr_func(r_fun: Callable[[np.ndarray, int | float], np.ndarray], 
                  corr_int: int | float, 
                  time_window: int | float, 
                  n_counts: int) -> np.ndarray:
    """
    Функция позволяет получить отсчёты теоретической корреляционной функции.

    Parametrs
    ---------
    r_fun : [[array_like, int | float], array_like]
            Корреляционная функция, заданная как функция от двух переменных.
    corr_int : int, float
            Длительность интервала корреляции, мкс.
    time_window : int, float
            Временной интервал, на котором рассматривается корреляция, мкс.
    n_counts : int
            Число отсчётов.

    Returns
    -------
    Функция возвращает отсчёты теоретической коррелированной функции.
    """
    shift_axis_us = np.linspace(-time_window, time_window, 2*n_counts-1)    # ось отстройки КФ

    corr_func_counts = r_fun(shift_axis_us, corr_int)   # отсчеты КФ
    
    return corr_func_counts