from valor_intrinseco_dcf import ValuationInputs, intrinsic_value, print_report


def ask_float(message: str, allow_empty: bool = False) -> float | None:
    while True:
        raw_value = input(message).strip()
        if allow_empty and raw_value == "":
            return None
        try:
            return float(raw_value)
        except ValueError:
            print("Enter a valid number.")


def ask_percent(message: str, allow_empty: bool = False) -> float | None:
    value = ask_float(message, allow_empty=allow_empty)
    if value is None:
        return None
    return value / 100


def ask_growth_list() -> list[float]:
    while True:
        raw_value = input(
            "Revenue CAGR (5y) or Avg EPS Growth (5y), comma separated (example 10,9,8,7,6): "
        ).strip()
        try:
            values = [float(item.strip()) / 100 for item in raw_value.split(",") if item.strip()]
        except ValueError:
            print("Enter only numbers separated by commas.")
            continue

        if len(values) != 5:
            print("Enter exactly 5 values.")
            continue
        return values


def collect_inputs() -> ValuationInputs:
    print("DCF FAIR VALUE")
    print("Enter percentages as standard numbers. Example: 10 for 10%")
    print()

    revenue = ask_float("Revenue: ")
    operating_margin = ask_percent("Operating Income Margin (%): ")
    tax_rate = ask_percent("Effective Tax Rate (%): ")
    depreciation_pct = ask_percent("Depreciation & Amortization Margin (%): ")
    capex_pct = ask_percent("Capital Expenditures Margin (%): ")
    nwc_pct = ask_percent("Net Working Capital Margin (%): ")
    revenue_growth = ask_growth_list()
    discount_rate = ask_percent("WACC (%): ")
    terminal_growth = ask_percent("Long Term Earnings Growth Rate (%): ")
    net_debt = ask_float("Net Debt: ")
    shares_outstanding = ask_float("Shares Outstanding: ")
    current_price = ask_float("Price, Current (press Enter if not available): ", allow_empty=True)

    return ValuationInputs(
        revenue=revenue,
        operating_margin=operating_margin,
        tax_rate=tax_rate,
        depreciation_pct=depreciation_pct,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        revenue_growth=revenue_growth,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        net_debt=net_debt,
        shares_outstanding=shares_outstanding,
        current_price=current_price,
    )


def main() -> None:
    inputs = collect_inputs()
    result = intrinsic_value(inputs)
    print()
    print_report(inputs, result)


if __name__ == "__main__":
    main()
