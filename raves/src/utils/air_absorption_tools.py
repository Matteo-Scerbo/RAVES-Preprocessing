import numpy as np
from typing import Union


def air_absorption_db(frequency: Union[np.ndarray, float],
                      humidity: float, temperature: float, pressure: float = 100.) -> Union[np.ndarray, float]:
    """
    Compute air absorption in dB per meter.

    Parameters
    ----------
    frequency : float or ndarray
        Frequency in Hz. Can be a scalar or a vector (process multiple at once).
    humidity : float
        Ambient relative humidity (%).
    temperature : float
        Ambient temperature (°C).
    pressure : float, default: 100.0
        Ambient pressure (kPa).

    Returns
    -------
    float or ndarray
        Attenuation in dB per meter at the given frequency or frequencies.

    Notes
    -----
    Formulas are adapted from sengpielaudio.com, originally from ISO 9613 Part 1.
    """
    T = 273.15 + temperature
    f = frequency
    pa = pressure
    hr = humidity

    To = 293.15
    Tow = 273.15
    pr = 101.325

    psat = pr * (10**(-6.8346 * ((Tow/T)**1.261) + 4.6151))
    h = hr * (psat / pa)
    frO = (pa / pr) * (24 + 4.04e4 * h * ((0.02 + h) / (0.391 + h)))
    frN = (pa / pr) * ((T / To)**(-1/2)) * (9 + 280 * h * np.exp(-4.170 * (((T / To)**(-1/3))-1)))

    x = 1.84e-11 * ((pa / pr)**-1) * ((T / To)**(1/2))
    y = 0.01275 * np.exp(-2239.1 / T) * ((frO + ((f**2) / frO))**-1)
    z = 0.1068 * np.exp(-3352 / T) * ((frN + (f**2) / frN)**-1)

    return 8.686 * (f**2) * (x + ((T / To)**(-5/2)) * (y + z))


def gain_from_dbm(dbm: Union[np.ndarray, float], distance: float = 1.) -> Union[np.ndarray, float]:
    """
    Convert per-meter attenuation (dB/m) to linear pressure gain over a distance.

    Parameters
    ----------
    dbm : float or ndarray
        Attenuation in dB per meter.
    distance : float, default 1.0
        Propagation distance in meters.

    Returns
    -------
    float or ndarray
        Linear pressure gain (amplitude scale factor).
    """

    return 10 ** (-dbm * distance / 20)


def air_absorption_linear(frequency: Union[np.ndarray, float], distance: float,
                          humidity: float, temperature: float, pressure: float = 100.) -> Union[np.ndarray, float]:
    """
    Compute linear pressure gain due to air absorption over a distance.

    Parameters
    ----------
    frequency : float or ndarray
        Frequency in Hz. Can be a scalar or a vector (process multiple at once).
    distance : float
        Propagation distance in meters.
    humidity : float
        Ambient relative humidity (%).
    temperature : float
        Ambient temperature (°C).
    pressure : float, default: 100.0
        Ambient pressure (kPa).

    Returns
    -------
    float or ndarray
        Linear pressure gain (amplitude scale factor) at the given frequency or frequencies.
    """
    return gain_from_dbm(air_absorption_db(frequency, humidity, temperature, pressure), distance)


def air_absorption_in_band(fc: float, fd: float, distance: float,
                           humidity: float, temperature: float, pressure: float = 100.,
                           num_samples: int = 1000) -> float:
    """
    Compute band-average linear pressure gain via RMS over a fractional band.

    The band is [fc/fd, fc*fd] and the response is averaged over linear frequency.

    Parameters
    ----------
    fc : float
        Band center frequency in Hz.
    fd : float
        Band width factor (e.g., sqrt(2) for full octave band).
    distance : float
        Propagation distance in meters.
    humidity : float
        Ambient relative humidity (%).
    temperature : float
        Ambient temperature (°C).
    pressure : float, default: 100.0
        Ambient pressure (kPa).
    num_samples : int, default 1000
        Number of frequency samples used inside the band.

    Returns
    -------
    float
        RMS linear pressure gain for the band.
    """
    return np.sqrt(np.mean(air_absorption_linear(np.linspace(fc/fd, fc*fd, num_samples),
                                                 distance, humidity, temperature, pressure)**2))


def air_absorption_in_bands(band_centers: np.ndarray, fd: float, distance: float,
                            humidity: float, temperature: float, pressure: float = 100,
                            num_samples: int = 1000) -> np.ndarray:
    """
    Compute band-average linear pressure gain for multiple band centers.

    Parameters
    ----------
    band_centers : ndarray
        Array of band center frequencies in Hz.
    fd : float
        Band width factor (e.g., sqrt(2) for full octave band).
    distance : float
        Propagation distance in meters.
    humidity : float
        Ambient relative humidity (%).
    temperature : float
        Ambient temperature (°C).
    pressure : float, default: 100.0
        Ambient pressure (kPa).
    num_samples : int, default 1000
        Number of frequency samples used inside each band.

    Returns
    -------
    ndarray
        Linear pressure gain per band center, same length as band_centers.
    """
    return np.array([air_absorption_in_band(fc, fd, distance, humidity, temperature, pressure, num_samples)
                     for fc in band_centers])


def sound_speed(humidity: float, temperature: float, pressure: float = 100.) -> float:
    """
    Compute the speed of sound in air.

    Parameters
    ----------
    humidity : float
        Ambient relative humidity (%).
    temperature : float
        Ambient temperature (°C).
    pressure : float, default: 100.0
        Ambient pressure (kPa).

    Returns
    -------
    float
        Speed of sound in m/s.

    Notes
    -----
    Formulas are adapted from resource.npl.co.uk, originally from Cramer (J. Acoust. Soc. Am., 93, p2510, 1993),
    "with saturation vapour pressure taken from Davis, Metrologia, 29, p67, 1992, and a mole fraction of carbon dioxide of 0.0004."
    """
    T = temperature
    Rh = humidity
    P = pressure * 1e3

    T_kel = 273.15 + T

    ENH = 3.14e-8 * P + 1.00062 + (T * T) * 5.6e-7

    PSV1 = T_kel * (T_kel * 1.2378847e-5 - 1.9121316e-2)
    PSV2 = 33.93711047 - 6.3431645e3 / T_kel
    PSV = np.exp(PSV1 + PSV2)

    H = Rh * ENH * PSV / P
    Xw = H / 100.0
    Xc = 400.0e-6

    C1 = (
        0.603055 * T
        + 331.5024
        - (T * T) * 5.28e-4
        + (0.1495874 * T + 51.471935 - (T * T) * 7.82e-4) * Xw
    )
    C2 = (
        (-1.82e-7 + 3.73e-8 * T - (T * T) * 2.93e-10) * P
        + (-85.20931 - 0.228525 * T + (T * T) * 5.91e-5) * Xc
    )
    C3 = (
        (Xw * Xw) * 2.835149
        + (P * P) * 2.15e-13
        - (Xc * Xc) * 29.179762
        - 4.86e-4 * Xw * P * Xc
    )

    return C1 + C2 - C3


if __name__ == "__main__":
    # Test (visualization) code
    import matplotlib.pyplot as plt

    distance = 1.
    temperature = 19.5
    pressure = 100
    humidity = 21.7

    freqs = np.logspace(np.log10(3e1), np.log10(3e4), int(1e3))
    db_over_meter = air_absorption_db(frequency=freqs, humidity=humidity, temperature=temperature, pressure=pressure)

    band_centers = np.array([125., 250., 500., 1e3, 2e3, 4e3, 8e3, 16e3])
    fd = np.sqrt(2)
    band_absorptions = air_absorption_in_bands(band_centers, fd,
                                               distance=1,
                                               humidity=humidity,
                                               temperature=temperature)

    fig, ax = plt.subplots(2, dpi=200, figsize=(8, 6))

    ax[0].plot(freqs, db_over_meter)
    ax[0].set_xscale('log')
    ax[0].set_xlim(3e1, 3e4)
    # ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('Absorption [dB/m]')
    ax[0].set_title('Air absorption in dB per meter.')

    # ax[1].plot(freqs, gain_from_dbm(db_over_meter),
    #            label='Continuous')
    # ax[1].plot(sum([[fc/fd, fc*fd]
    #                 for fc in band_centers],
    #                []),
    #            sum([[band_absorptions[fi], band_absorptions[fi]]
    #                 for fi in range(len(band_centers))],
    #                []),
    #            ls=':', c='black', marker='x', label='Octave bands')
    # ax[1].legend()
    # ax[1].set_xscale('log')
    # ax[1].set_xlim(3e1, 3e4)
    # # ax[1].set_xlabel('Frequency [Hz]')
    # ax[1].set_ylabel('Pressure amplitude gain')
    # ax[1].set_title('Air absorption as pressure amplitude gain (over one meter).')

    ax[1].plot(freqs, gain_from_dbm(db_over_meter)**2,
               label='Continuous')
    ax[1].plot(sum([[fc/fd, fc*fd]
                    for fc in band_centers],
                   []),
               sum([[band_absorptions[fi]**2, band_absorptions[fi]**2]
                    for fi in range(len(band_centers))],
                   []),
               ls=':', c='black', marker='x', label='Octave bands')
    ax[1].legend()
    ax[1].set_xscale('log')
    ax[1].set_xlim(3e1, 3e4)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Energy gain')
    ax[1].set_title('Air absorption as energy gain (over one meter).')

    plt.tight_layout()
    plt.show()
    plt.close()
