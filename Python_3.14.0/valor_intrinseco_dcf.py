import argparse
import json
from dataclasses import dataclass


@dataclass
class ValuationInputs:
    revenue: float
    operating_margin: float
    tax_rate: float
    depreciation_pct: float
    capex_pct: float
    nwc_pct: float
    revenue_growth: list[float]
    discount_rate: float
    terminal_growth: float
    net_debt: float
    shares_outstanding: float
    current_price: float | None = None


def load_inputs(path: str) -> ValuationInputs:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return ValuationInputs(**data)


def project_free_cash_flows(inputs: ValuationInputs) -> list[dict]:
    projections = []
    revenue = inputs.revenue
    previous_nwc = revenue * inputs.nwc_pct

    for year, growth in enumerate(inputs.revenue_growth, start=1):
        revenue *= 1 + growth
        ebit = revenue * inputs.operating_margin
        nopat = ebit * (1 - inputs.tax_rate)
        depreciation = revenue * inputs.depreciation_pct
        capex = revenue * inputs.capex_pct
        current_nwc = revenue * inputs.nwc_pct
        delta_nwc = current_nwc - previous_nwc
        free_cash_flow = nopat + depreciation - capex - delta_nwc

        projections.append(
            {
                "year": year,
                "revenue": revenue,
                "ebit": ebit,
                "nopat": nopat,
                "depreciation": depreciation,
                "capex": capex,
                "delta_nwc": delta_nwc,
                "free_cash_flow": free_cash_flow,
            }
        )
        previous_nwc = current_nwc

    return projections


def intrinsic_value(inputs: ValuationInputs) -> dict:
    if inputs.discount_rate <= inputs.terminal_growth:
        raise ValueError("La tasa de descuento debe ser mayor que el crecimiento terminal.")
    if inputs.shares_outstanding <= 0:
        raise ValueError("El numero de acciones debe ser mayor que cero.")

    projections = project_free_cash_flows(inputs)
    present_value_fcf = 0.0

    for projection in projections:
        year = projection["year"]
        present_value_fcf += projection["free_cash_flow"] / ((1 + inputs.discount_rate) ** year)

    terminal_fcf = projections[-1]["free_cash_flow"] * (1 + inputs.terminal_growth)
    terminal_value = terminal_fcf / (inputs.discount_rate - inputs.terminal_growth)
    terminal_value_present = terminal_value / ((1 + inputs.discount_rate) ** projections[-1]["year"])

    enterprise_value = present_value_fcf + terminal_value_present
    equity_value = enterprise_value - inputs.net_debt
    value_per_share = equity_value / inputs.shares_outstanding
    margin_of_safety = None

    if inputs.current_price is not None and inputs.current_price > 0:
        margin_of_safety = (value_per_share / inputs.current_price) - 1

    return {
        "projections": projections,
        "present_value_fcf": present_value_fcf,
        "terminal_value": terminal_value,
        "terminal_value_present": terminal_value_present,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": value_per_share,
        "margin_of_safety": margin_of_safety,
    }


def print_report(inputs: ValuationInputs, result: dict) -> None:
    print("DCF FAIR VALUE")
    print("-" * 60)
    print(f"Revenue:                  {inputs.revenue:,.2f}")
    print(f"WACC:                     {inputs.discount_rate:.2%}")
    print(f"Long Term Earnings Growth Rate: {inputs.terminal_growth:.2%}")
    print(f"Net Debt:                 {inputs.net_debt:,.2f}")
    print(f"Shares Outstanding:       {inputs.shares_outstanding:,.2f}")
    print("-" * 60)
    print("Projected Free Cash Flow")

    for projection in result["projections"]:
        print(
            f"Year {projection['year']}: revenue={projection['revenue']:,.2f} "
            f"fcf={projection['free_cash_flow']:,.2f}"
        )

    print("-" * 60)
    print(f"PV of Explicit FCF:       {result['present_value_fcf']:,.2f}")
    print(f"PV of Terminal Value:     {result['terminal_value_present']:,.2f}")
    print(f"Enterprise Value:         {result['enterprise_value']:,.2f}")
    print(f"Equity Value:             {result['equity_value']:,.2f}")
    print(f"Fair Value Per Share:     {result['value_per_share']:,.2f}")

    if result["margin_of_safety"] is not None:
        print(f"Margin of Safety:         {result['margin_of_safety']:.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate a company's fair value using a DCF model.")
    parser.add_argument(
        "input_file",
        help="Path to a JSON file with valuation assumptions.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    inputs = load_inputs(args.input_file)
    result = intrinsic_value(inputs)
    print_report(inputs, result)


if __name__ == "__main__":
    main()
