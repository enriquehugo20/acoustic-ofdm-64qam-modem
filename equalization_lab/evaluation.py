import csv
import numpy as np

from config import (
    ESTIMATION_SYMBOLS,
    VALIDATION_SYMBOLS,
    OUTPUT_DIR,
)

from equalizers import (
    zero_forcing,
    mmse_equalizer,
)

from channel_loader import (
    generate_known_training_symbols,
)


def calculate_evm(reference, estimated):
    error = estimated - reference

    signal_power = np.mean(np.abs(reference) ** 2)
    error_power = np.mean(np.abs(error) ** 2)

    if signal_power <= 0:
        return float("inf")

    return float(
        100 * np.sqrt(error_power / signal_power)
    )


def calculate_qpsk_ber(reference_phases, estimated_symbols):
    """
    Decisión QPSK por cuadrante.
    """

    estimated_phases = np.mod(
        np.rint(
            np.angle(estimated_symbols) / (np.pi / 2)
        ).astype(int),
        4,
    )

    reference_bits = np.stack(
        [
            (reference_phases >> 1) & 1,
            reference_phases & 1,
        ],
        axis=-1,
    )

    estimated_bits = np.stack(
        [
            (estimated_phases >> 1) & 1,
            estimated_phases & 1,
        ],
        axis=-1,
    )

    errors = np.count_nonzero(
        reference_bits != estimated_bits
    )

    total_bits = reference_bits.size

    return errors / total_bits


def estimate_noise_variance(training_estimates, channel):
    """
    Estimación aproximada de variación residual por subportadora.

    No separa perfectamente ruido y variación temporal del canal.
    """

    residual = training_estimates - channel[np.newaxis, :]

    variance = np.mean(
        np.abs(residual) ** 2,
        axis=0,
    )

    return np.maximum(variance, 1e-10)


def evaluate_capture(capture_estimates, known_symbols, phases):
    """
    Evalúa una captura utilizando separación temporal:

    Primeros 16 símbolos: estimación.
    Últimos 16 símbolos: validación.
    """

    training = capture_estimates[:ESTIMATION_SYMBOLS]

    validation = capture_estimates[
        ESTIMATION_SYMBOLS:
        ESTIMATION_SYMBOLS + VALIDATION_SYMBOLS
    ]

    if len(training) != ESTIMATION_SYMBOLS:
        raise ValueError("Entrenamiento incompleto.")

    if len(validation) != VALIDATION_SYMBOLS:
        raise ValueError("Validación incompleta.")

    # Estimación del canal usando exclusivamente los primeros 16 símbolos.
    channel = np.mean(training, axis=0)

    # Variación residual estimada durante el entrenamiento.
    noise_variance = estimate_noise_variance(
        training,
        channel,
    )

    # Los símbolos recibidos se reconstruyen a partir de las
    # estimaciones LS guardadas en el NPZ:
    # H_est[m,k] = Y[m,k] / X[k]
    #
    # Por tanto, Y reconstruido = H_est * X.
    received = validation * known_symbols[np.newaxis, :]

    reference = np.broadcast_to(
        known_symbols,
        received.shape,
    )

    # Ecualización.
    zf_output = zero_forcing(
        received,
        channel,
    )

    mmse_output = mmse_equalizer(
        received,
        channel,
        noise_variance,
    )

    results = {
        "zf_evm_percent": calculate_evm(
            reference,
            zf_output,
        ),
        "mmse_evm_percent": calculate_evm(
            reference,
            mmse_output,
        ),
        "zf_ber": calculate_qpsk_ber(
            np.broadcast_to(phases, reference.shape),
            zf_output,
        ),
        "mmse_ber": calculate_qpsk_ber(
            np.broadcast_to(phases, reference.shape),
            mmse_output,
        ),
        "noise_variance_mean": float(
            np.mean(noise_variance)
        ),
        "weakest_channel_db": float(
            20 * np.log10(
                max(np.min(np.abs(channel)), 1e-12)
            )
        ),
    }

    return results


def run_evaluation(channel_data):
    estimates = channel_data["symbol_channel_estimates"]

    known_symbols, phases = generate_known_training_symbols(
        estimates.shape[2]
    )

    all_results = []

    print("\nZF vs MMSE EQUALIZATION")
    print("=" * 75)

    for index in range(estimates.shape[0]):
        result = evaluate_capture(
            estimates[index],
            known_symbols,
            phases,
        )

        result["capture"] = index + 1
        all_results.append(result)

        print(f"\nCapture {index + 1}")
        print("-" * 50)

        print(
            f"ZF   EVM: {result['zf_evm_percent']:.4f}%"
        )
        print(
            f"MMSE EVM: {result['mmse_evm_percent']:.4f}%"
        )

        print(
            f"ZF   BER: {result['zf_ber']:.6f}"
        )
        print(
            f"MMSE BER: {result['mmse_ber']:.6f}"
        )

        print(
            "Weakest estimated subcarrier: "
            f"{result['weakest_channel_db']:.2f} dB"
        )

    save_results(all_results)
    print_final_decision(all_results)

    return all_results


def save_results(results):
    output_file = OUTPUT_DIR / "equalization_results.csv"

    fieldnames = list(results[0].keys())

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nResults saved: {output_file}"
    )


def print_final_decision(results):
    zf_evm = np.median(
        [item["zf_evm_percent"] for item in results]
    )

    mmse_evm = np.median(
        [item["mmse_evm_percent"] for item in results]
    )

    zf_ber = np.mean(
        [item["zf_ber"] for item in results]
    )

    mmse_ber = np.mean(
        [item["mmse_ber"] for item in results]
    )

    print("\nFINAL COMPARISON")
    print("=" * 75)

    print(f"Median ZF EVM:   {zf_evm:.4f}%")
    print(f"Median MMSE EVM: {mmse_evm:.4f}%")

    print(f"Mean ZF BER:     {zf_ber:.6f}")
    print(f"Mean MMSE BER:   {mmse_ber:.6f}")

    print("\nPRELIMINARY RECOMMENDATION")
    print("-" * 75)

    if mmse_ber < zf_ber:
        recommendation = "MMSE"
    elif zf_ber < mmse_ber:
        recommendation = "ZF"
    elif mmse_evm < zf_evm:
        recommendation = "MMSE"
    else:
        recommendation = "ZF"

    print(f"Best candidate based on this evaluation: {recommendation}")

    print(
        "\nIMPORTANT: This comparison uses reconstructed received "
        "symbols derived from saved channel estimates."
    )

    print(
        "It is a channel-estimate consistency evaluation, "
        "not an independent physical BER measurement."
    )

    print(
        "Final integration must be validated using actual "
        "received resource grids."
    )