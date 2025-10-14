import numpy as np
from typing import Union


def air_absorption_db(frequency: Union[np.ndarray, float],
                      humidity: float, temperature: float, pressure: float = 100) -> Union[np.ndarray, float]:
    """
    Formulas adapted from sengpielaudio.com, originally from ISO 9613 Part 1
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


def gain_from_dbm(dbm: Union[np.ndarray, float], distance: float = 1) -> Union[np.ndarray, float]:
    return 10 ** (-dbm * distance / 20)


def air_absorption_linear(frequency: Union[np.ndarray, float], distance: float,
                          humidity: float, temperature: float, pressure: float = 100) -> Union[np.ndarray, float]:
    return gain_from_dbm(air_absorption_db(frequency, humidity, temperature, pressure), distance)


def air_absorption_in_band(fc: float, fd: float, distance: float,
                           humidity: float, temperature: float, pressure: float = 100,
                           num_samples: int = 1000) -> float:
    """
    The band level is the root-mean-square of the response within the band (integral over linear frequency).
    """
    return np.sqrt(np.mean(air_absorption_linear(np.linspace(fc/fd, fc*fd, num_samples),
                                                 distance, humidity, temperature, pressure)**2))


def air_absorption_in_bands(band_centers: np.ndarray, fd: float, distance: float,
                            humidity: float, temperature: float, pressure: float = 100) -> np.ndarray:
    return np.array([air_absorption_in_band(fc, fd, distance, humidity, temperature, pressure)
                     for fc in band_centers])


def sound_speed(temperature: float) -> float:
    T = 273.15 + temperature
    R = 287.05
    return np.sqrt(1.4 * T * R)


if __name__ == "__main__":
    # Test visualization code
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
