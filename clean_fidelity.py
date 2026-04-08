"""Convert a raw Fidelity activity CSV into the cleaned dashboard schema.

The dashboard (app.py) reads a CSV with a fixed 15-column schema. This module
is the only place that knows about Fidelity's raw export format — to add
another broker later, drop in a sibling module (e.g. clean_schwab.py) that
exposes the same `clean_*_csv` signature.
"""

import hashlib
from io import BytesIO, StringIO

import pandas as pd

BROKER = "Fidelity"

# Money market funds — distributions are always ordinary income.
# Backstop for cases where the description text wouldn't match the keyword list.
MONEY_MARKET_SYMBOLS = {
    "SPAXX", "FDRXX", "FZDXX", "FZFXX", "SPRXX", "FCASH",
    "FNSXX", "FGRXX", "FGXXX", "FMPXX", "FTEXX",
}

# Substrings in the Account column that mark a tax-advantaged account.
# Dividends in IRAs, 401ks, HSAs, 529s, etc. are tax-free (Roth) or
# tax-deferred (Traditional) and shouldn't have estimated tax applied.
TAX_ADVANTAGED_ACCOUNT_KEYWORDS = (
    "IRA",
    "ROTH",
    "401",
    "403",
    "HSA",
    "SEP",
    "SIMPLE",
    "529",
)

# Substrings in the security description that indicate ordinary-rate dividends:
# bond funds, money market, REITs, floating-rate funds, etc. Matched
# case-insensitively against the Description column.
ORDINARY_DESC_KEYWORDS = (
    "MONEY MARKET",
    "FLOATING RATE",
    "BOND",
    "TREASURY",
    "REIT",
    "MORTGAGE",
    "HIGH YIELD",
    "MUNI",
    "MUNICIPAL",
    "INCOME FUND",
)

CLEANED_COLUMNS = [
    "event_id",
    "event_date",
    "year_month",
    "symbol",
    "security_name",
    "event_type",
    "payout_method",
    "gross_dividend_amount",
    "reinvestment_amount",
    "cash_paid_amount",
    "tax_classification",
    "estimated_tax_rate",
    "estimated_tax_amount",
    "estimated_after_tax_amount",
    "source_broker",
]


def classify(symbol: str, description: str) -> str:
    """Return 'ordinary' or 'qualified' based on symbol + description text."""
    sym = (symbol or "").strip().upper()
    desc = (description or "").upper()
    if sym in MONEY_MARKET_SYMBOLS:
        return "ordinary"
    if any(kw in desc for kw in ORDINARY_DESC_KEYWORDS):
        return "ordinary"
    return "qualified"


def _event_id(date_str: str, account: str, symbol: str, amount: float) -> str:
    key = f"{date_str}|{account}|{symbol}|{amount:.2f}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def clean_fidelity_csv(
    raw_bytes: bytes,
    ordinary_rate: float,
    qualified_rate: float,
) -> bytes:
    """Parse a raw Fidelity activity export and emit cleaned-schema CSV bytes.

    Args:
        raw_bytes: contents of the user-uploaded Fidelity CSV.
        ordinary_rate: tax rate for ordinary dividends as a decimal (e.g. 0.24).
        qualified_rate: tax rate for qualified dividends as a decimal (e.g. 0.15).

    Returns:
        UTF-8 encoded CSV bytes ready to feed straight to load_cleaned_csv().
    """
    # Fidelity prepends a variable number of blank rows before the header,
    # so locate the header line by content rather than fixed offset.
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Run Date,")),
        None,
    )
    if header_idx is None:
        raise ValueError(
            "This doesn't look like a Fidelity activity export "
            "(couldn't find the 'Run Date' header row)."
        )

    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    df.columns = [c.strip() for c in df.columns]

    required = {"Run Date", "Action", "Symbol", "Description", "Amount ($)", "Account Number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fidelity CSV is missing expected columns: {sorted(missing)}")

    # Drop the trailing disclaimer/blank rows by requiring a parseable date.
    df["Run Date"] = pd.to_datetime(df["Run Date"], errors="coerce")
    df = df.dropna(subset=["Run Date"])

    # Keep only dividend payouts. Reinvestment rows are intentionally ignored —
    # they're the cash being redeployed, not a separate dividend event.
    action = df["Action"].fillna("").str.upper()
    div = df[action.str.startswith("DIVIDEND RECEIVED")].copy()

    div["Amount ($)"] = pd.to_numeric(div["Amount ($)"], errors="coerce").fillna(0)
    div = div[div["Amount ($)"] > 0].reset_index(drop=True)

    if div.empty:
        raise ValueError("No dividend events found in this file.")

    out = pd.DataFrame({
        "event_date": div["Run Date"],
        "year_month": div["Run Date"].dt.strftime("%Y-%m"),
        "symbol": div["Symbol"].astype(str).str.strip().str.upper(),
        "security_name": div["Description"].astype(str).str.strip(),
        "event_type": "dividend",
        "payout_method": "cash",
        "gross_dividend_amount": div["Amount ($)"].round(2),
        "reinvestment_amount": 0.0,
    })
    out["cash_paid_amount"] = out["gross_dividend_amount"]
    out["tax_classification"] = [
        classify(s, n) for s, n in zip(out["symbol"], out["security_name"])
    ]

    # Tax-advantaged accounts (IRA / 401k / HSA / 529 / etc.) owe no tax on
    # dividends. Detect by scanning the raw Account column for keywords; the
    # security's classification (qualified/ordinary) stays meaningful but the
    # rate is zeroed for these rows.
    account_names = div["Account"].fillna("").astype(str).str.upper()
    is_tax_advantaged = account_names.apply(
        lambda name: any(kw in name for kw in TAX_ADVANTAGED_ACCOUNT_KEYWORDS)
    ).tolist()

    out["estimated_tax_rate"] = [
        0.0 if adv else (qualified_rate if cls == "qualified" else ordinary_rate)
        for cls, adv in zip(out["tax_classification"], is_tax_advantaged)
    ]
    out["estimated_tax_amount"] = (
        out["gross_dividend_amount"] * out["estimated_tax_rate"]
    ).round(2)
    out["estimated_after_tax_amount"] = (
        out["gross_dividend_amount"] - out["estimated_tax_amount"]
    ).round(2)
    out["source_broker"] = BROKER
    out["event_id"] = [
        _event_id(d.strftime("%Y-%m-%d"), str(acct), s, a)
        for d, acct, s, a in zip(
            out["event_date"],
            div["Account Number"],
            out["symbol"],
            out["gross_dividend_amount"],
        )
    ]

    out = out[CLEANED_COLUMNS].sort_values("event_date").reset_index(drop=True)

    buf = BytesIO()
    out.to_csv(buf, index=False)
    return buf.getvalue()
