import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Optional


# -----------------------------
# Two-Stage DCF (UFCF)
# -----------------------------
@dataclass
class DCFInputs:
    unlevered_free_cash_flow: float      # "Unlevered Free Cash Flow" (base year / LTM)
    wacc: float                          # WACC as decimal (e.g., 0.093)
    net_debt: float                      # "Net Debt" (can be negative if net cash)
    shares_outstanding: float            # "Shares Outstanding"
    price_current: float                 # "Price, Current"

    projection_years: int                # typically 5
    stage1_growth_cagr: float            # use "Unlevered FCF Forecast CAGR (5y)" as decimal
    terminal_growth_rate: float          # perpetual growth as decimal (e.g., 0.025)


@dataclass
class DCFOutputs:
    enterprise_value: float
    equity_value: float
    intrinsic_value_per_share: float
    upside_downside_pct: float
    pv_stage_1: float
    pv_terminal_value: float


def validate_inputs(i: DCFInputs) -> None:
    if i.unlevered_free_cash_flow <= 0:
        raise ValueError("Unlevered Free Cash Flow must be > 0.")
    if i.wacc <= 0:
        raise ValueError("WACC must be > 0.")
    if i.shares_outstanding <= 0:
        raise ValueError("Shares Outstanding must be > 0.")
    if i.price_current <= 0:
        raise ValueError("Price, Current must be > 0.")
    if i.projection_years <= 0:
        raise ValueError("Projection Years must be > 0.")
    if i.terminal_growth_rate < 0:
        raise ValueError("Terminal Growth Rate must be >= 0.")
    if i.wacc <= i.terminal_growth_rate:
        raise ValueError("WACC must be greater than Terminal Growth Rate (otherwise terminal value explodes).")


def compute_two_stage_dcf(i: DCFInputs) -> DCFOutputs:
    validate_inputs(i)

    # Stage 1: project UFCF using a single CAGR for each year
    pv_stage_1 = 0.0
    fcf = i.unlevered_free_cash_flow
    for t in range(1, i.projection_years + 1):
        fcf *= (1.0 + i.stage1_growth_cagr)
        pv_stage_1 += fcf / ((1.0 + i.wacc) ** t)

    # Terminal Value (Gordon Growth) based on last projected FCF
    terminal_value = (fcf * (1.0 + i.terminal_growth_rate)) / (i.wacc - i.terminal_growth_rate)
    pv_terminal_value = terminal_value / ((1.0 + i.wacc) ** i.projection_years)

    enterprise_value = pv_stage_1 + pv_terminal_value

    # Equity Value = Enterprise Value - Net Debt
    equity_value = enterprise_value - i.net_debt

    intrinsic_value_per_share = equity_value / i.shares_outstanding
    upside_downside_pct = (intrinsic_value_per_share / i.price_current - 1.0) * 100.0

    return DCFOutputs(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        intrinsic_value_per_share=intrinsic_value_per_share,
        upside_downside_pct=upside_downside_pct,
        pv_stage_1=pv_stage_1,
        pv_terminal_value=pv_terminal_value
    )


# -----------------------------
# Tkinter App (simple, button visible)
# -----------------------------
class DCFFairValueAppV2(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DCF Fair Value (Two-Stage) - Unlevered Free Cash Flow")
        self.geometry("900x620")
        self.resizable(True, True)

        self._build_ui()

    @staticmethod
    def _parse_float(s: str) -> float:
        s = s.strip().replace(",", "")
        if s == "":
            raise ValueError("Empty input.")
        return float(s)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        header = ttk.Frame(self)
        header.pack(fill="x", padx=12, pady=10)

        ttk.Label(
            header,
            text="Enter inputs (InvestingPro+ naming). Click Compute to get Intrinsic Value Per Share.",
            font=("Segoe UI", 10, "bold")
        ).pack(side="left")

        ttk.Button(header, text="Compute DCF Fair Value", command=self._on_compute).pack(side="right", padx=6)

        core = ttk.LabelFrame(self, text="Core Inputs (search these in InvestingPro+)")
        core.pack(fill="x", padx=12, pady=8)

        self.var_ufcf = tk.StringVar()
        self.var_wacc = tk.StringVar()
        self.var_net_debt = tk.StringVar()
        self.var_shares = tk.StringVar()
        self.var_price = tk.StringVar()

        def row(frame, r, label, var, hint):
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(frame, textvariable=var, width=28).grid(row=r, column=1, sticky="w", **pad)
            ttk.Label(frame, text=hint, foreground="#666").grid(row=r, column=2, sticky="w", **pad)

        row(core, 0, "Unlevered Free Cash Flow", self.var_ufcf, "Base year / LTM (currency)")
        row(core, 1, "WACC (%)", self.var_wacc, "e.g., 9.3 means 9.3%")
        row(core, 2, "Net Debt", self.var_net_debt, "Use negative if net cash")
        row(core, 3, "Shares Outstanding", self.var_shares, "Total shares (not %) ")
        row(core, 4, "Price, Current", self.var_price, "Current share price")

        fc = ttk.LabelFrame(self, text="Forecast Assumptions (Two-Stage)")
        fc.pack(fill="x", padx=12, pady=8)

        self.var_years = tk.StringVar(value="5")
        self.var_stage1 = tk.StringVar()     # Unlevered FCF Forecast CAGR (5y)
        self.var_terminal = tk.StringVar(value="2.5")

        row(fc, 0, "Projection Years", self.var_years, "Typically 5")
        row(fc, 1, "Unlevered FCF Forecast CAGR (5y) (%)", self.var_stage1, "Use InvestingPro+ CAGR (Stage 1)")
        row(fc, 2, "Terminal Growth Rate (%)", self.var_terminal, "Perpetual growth (typically 2.0–3.0%)")

        expl = ttk.LabelFrame(self, text="What is Terminal Growth Rate?")
        expl.pack(fill="x", padx=12, pady=8)
        msg = (
            "Terminal Growth Rate (%) is the long-run perpetual growth rate after the explicit forecast period.\n"
            "It should be conservative (often ~2–3% for USD markets). It is NOT a metric from GAAP statements;\n"
            "it is a modeling assumption. Keep it below WACC."
        )
        ttk.Label(expl, text=msg, foreground="#333").pack(anchor="w", padx=10, pady=8)

        out = ttk.LabelFrame(self, text="Results")
        out.pack(fill="both", expand=True, padx=12, pady=10)

        self.txt = tk.Text(out, height=12, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt.insert("1.0", "Ready. Fill inputs and click 'Compute DCF Fair Value'.\n")

    def _on_compute(self):
        try:
            i = DCFInputs(
                unlevered_free_cash_flow=self._parse_float(self.var_ufcf.get()),
                wacc=self._parse_float(self.var_wacc.get()) / 100.0,
                net_debt=self._parse_float(self.var_net_debt.get()),
                shares_outstanding=self._parse_float(self.var_shares.get()),
                price_current=self._parse_float(self.var_price.get()),
                projection_years=int(self.var_years.get().strip()),
                stage1_growth_cagr=self._parse_float(self.var_stage1.get()) / 100.0,
                terminal_growth_rate=self._parse_float(self.var_terminal.get()) / 100.0,
            )

            o = compute_two_stage_dcf(i)

            def fmt(x: float) -> str:
                return f"{x:,.2f}"

            self.txt.delete("1.0", "end")
            self.txt.insert("1.0", "\n".join([
                "DCF Fair Value (Two-Stage) - Unlevered Free Cash Flow\n",
                "Inputs",
                f"- Unlevered Free Cash Flow (Base): {fmt(i.unlevered_free_cash_flow)}",
                f"- WACC: {i.wacc*100:.2f}%",
                f"- Net Debt: {fmt(i.net_debt)}",
                f"- Projection Years: {i.projection_years}",
                f"- Stage 1 Growth (Unlevered FCF Forecast CAGR (5y)): {i.stage1_growth_cagr*100:.2f}%",
                f"- Terminal Growth Rate: {i.terminal_growth_rate*100:.2f}%",
                f"- Shares Outstanding: {fmt(i.shares_outstanding)}",
                f"- Price, Current: {fmt(i.price_current)}\n",
                "Results",
                f"- PV of Stage 1: {fmt(o.pv_stage_1)}",
                f"- PV of Terminal Value: {fmt(o.pv_terminal_value)}",
                f"- Enterprise Value (DCF): {fmt(o.enterprise_value)}",
                f"- Equity Value (DCF): {fmt(o.equity_value)}",
                f"- Intrinsic Value Per Share (DCF Fair Value): {fmt(o.intrinsic_value_per_share)}",
                f"- Upside/Downside vs Price, Current: {o.upside_downside_pct:.2f}%",
            ]))

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = DCFFairValueAppV2()
    app.mainloop()
