
from transmitter.resource_grid import build_preamble_grid
from transmitter.ofdm_modulator import ofdm_modulate


def generate_preamble() :
    """
    Generate one known OFDM synchronization symbol.
    """
    grid = build_preamble_grid()

    return ofdm_modulate(grid)
