import argparse
import json

from valor_intrinseco_dcf import intrinsic_value, load_inputs


def parse_range(value: str) -> list[float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Usa el formato inicio,fin,paso. Ejemplo: 0.08,0.12,0.01")

    start, end, step = parts
    values = []
    current = start
    while current <= end + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera una tabla de sensibilidad DCF.")
    parser.add_argument("input_file", help="Ruta al archivo JSON con los supuestos base.")
    parser.add_argument(
        "--discount-range",
        default="0.08,0.12,0.01",
        help="Rango para tasa de descuento: inicio,fin,paso",
    )
    parser.add_argument(
        "--terminal-range",
        default="0.01,0.04,0.01",
        help="Rango para crecimiento terminal: inicio,fin,paso",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    inputs = load_inputs(args.input_file)
    discount_values = parse_range(args.discount_range)
    terminal_values = parse_range(args.terminal_range)

    print("SENSIBILIDAD VALOR INTRINSECO POR ACCION")
    print("filas = crecimiento terminal, columnas = tasa de descuento")
    print()

    header = ["g \\ r"] + [f"{rate:.2%}" for rate in discount_values]
    print(" | ".join(f"{item:>12}" for item in header))
    print("-" * (15 * len(header)))

    for terminal_growth in terminal_values:
        row = [f"{terminal_growth:.2%}"]
        for discount_rate in discount_values:
            if discount_rate <= terminal_growth:
                row.append("N/A")
                continue
            inputs.discount_rate = discount_rate
            inputs.terminal_growth = terminal_growth
            result = intrinsic_value(inputs)
            row.append(f"{result['value_per_share']:,.2f}")
        print(" | ".join(f"{item:>12}" for item in row))


if __name__ == "__main__":
    main()
