import time
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st

from clean_fidelity import clean_fidelity_csv


@st.cache_data(show_spinner=False)
def load_cleaned_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parse the cleaned dividend CSV. Handles BOM and whitespace-padded columns."""
    df = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    numeric_cols = [
        "gross_dividend_amount",
        "reinvestment_amount",
        "cash_paid_amount",
        "estimated_tax_rate",
        "estimated_tax_amount",
        "estimated_after_tax_amount",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "tax_advantaged" in df.columns:
        df["tax_advantaged"] = df["tax_advantaged"].astype(str).str.lower().isin(
            ("true", "1", "yes")
        )
    else:
        df["tax_advantaged"] = False
    return df


def money(value: float) -> str:
    return f"${value:,.2f}"


def build_monthly_frame(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Aggregate to one row per month in [start, end], filling gaps with 0."""
    if df.empty:
        monthly = pd.DataFrame(
            columns=["gross", "tax", "after_tax", "payouts"]
        )
    else:
        monthly = (
            df.assign(month=df["event_date"].dt.to_period("M"))
            .groupby("month")
            .agg(
                gross=("gross_dividend_amount", "sum"),
                tax=("estimated_tax_amount", "sum"),
                after_tax=("estimated_after_tax_amount", "sum"),
                payouts=("event_id", "count"),
            )
        )
    full_range = pd.period_range(
        pd.Timestamp(start).to_period("M"),
        pd.Timestamp(end).to_period("M"),
        freq="M",
    )
    monthly = monthly.reindex(full_range, fill_value=0).rename_axis("month").reset_index()
    monthly["month_ts"] = monthly["month"].dt.to_timestamp()
    monthly["month_label"] = monthly["month_ts"].dt.strftime("%b %Y")
    return monthly


def get_filter_bounds(
    df: pd.DataFrame,
    mode: str,
    custom_start=None,
    custom_end=None,
):
    max_date = df["event_date"].max()
    if mode == "YTD":
        return pd.Timestamp(year=max_date.year, month=1, day=1), max_date
    if mode == "Last 12 months":
        return max_date - pd.DateOffset(months=12) + pd.Timedelta(days=1), max_date
    if mode == "All time":
        return df["event_date"].min(), max_date
    if mode == "Custom":
        return pd.Timestamp(custom_start), pd.Timestamp(custom_end)
    return df["event_date"].min(), max_date


def format_filter_label(mode: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    if mode == "YTD":
        return f"YTD {start.year}"
    if mode == "Last 12 months":
        return "Last 12 months"
    if mode == "All time":
        return f"All time ({start.strftime('%b %Y')} – {end.strftime('%b %Y')})"
    return f"{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"


def render_report_view() -> None:
    # Back button (subtle, left-aligned)
    back_col, _ = st.columns([1, 5])
    with back_col:
        if st.button("← Back", type="tertiary"):
            st.session_state.view = "upload"
            st.rerun()

    st.title("Your Dividend Summary")

    df = load_cleaned_csv(st.session_state.csv_bytes)

    # Filter chips
    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = "YTD"

    filter_mode = st.segmented_control(
        "Time range",
        options=["YTD", "Last 12 months", "All time", "Custom"],
        default=st.session_state.filter_mode,
        label_visibility="collapsed",
        key="filter_mode_control",
    )
    if filter_mode is None:
        filter_mode = st.session_state.filter_mode
    st.session_state.filter_mode = filter_mode

    custom_start = custom_end = None
    if filter_mode == "Custom":
        file_min = df["event_date"].min().date()
        file_max = df["event_date"].max().date()
        c1, c2 = st.columns(2)
        with c1:
            custom_start = st.date_input("Start", value=file_min)
        with c2:
            custom_end = st.date_input("End", value=file_max)

    start_date, end_date = get_filter_bounds(df, filter_mode, custom_start, custom_end)
    df_filtered = df[
        (df["event_date"] >= start_date) & (df["event_date"] <= end_date)
    ]

    gen_ord, gen_qual = st.session_state.generated_rates
    filter_label = format_filter_label(filter_mode, start_date, end_date)
    st.caption(
        f"{filter_label} · {len(df_filtered)} events · "
        f"using **{gen_ord}** ordinary / **{gen_qual}** qualified"
    )

    if df_filtered.empty:
        st.info("No dividend events in this date range. Try a different filter.")
        return

    total_dividends = df_filtered["gross_dividend_amount"].sum()
    estimated_taxes = df_filtered["estimated_tax_amount"].sum()
    after_tax = df_filtered["estimated_after_tax_amount"].sum()

    tab_overview, tab_holdings, tab_events = st.tabs(
        ["Overview", "By Holding", "Events"]
    )

    with tab_overview:
        _render_overview_tab(
            df_filtered,
            start_date,
            end_date,
            filter_mode,
            total_dividends,
            estimated_taxes,
            after_tax,
        )

    with tab_holdings:
        _render_holdings_tab(df_filtered, after_tax)

    with tab_events:
        _render_events_tab(df_filtered)


def _render_overview_tab(
    df_filtered,
    start_date,
    end_date,
    filter_mode,
    total_dividends,
    estimated_taxes,
    after_tax,
) -> None:
    # Monthly average (works for any mode)
    range_days = (end_date - start_date).days + 1
    months_in_range = max(range_days / 30.44, 1)
    monthly_avg_after_tax = after_tax / months_in_range

    projection_line = ""
    if filter_mode == "YTD":
        projected_annual = monthly_avg_after_tax * 12
        projection_line = (
            f"<div style='margin-top:14px; font-size:15px; color:#2e7d32;'>"
            f"At this pace: <b>{money(projected_annual)}</b> for the year</div>"
        )

    st.markdown(
        f"""
        <div style="
            background-color: #eaf5ea;
            border: 1px solid #c8e6c9;
            border-radius: 12px;
            padding: 24px 28px;
            margin: 18px 0 20px 0;
        ">
            <div style="font-size:14px; color:#4e6e55; font-weight:500; letter-spacing:0.3px;">
                YOU'RE EARNING
            </div>
            <div style="font-size:44px; font-weight:700; color:#1b5e20; line-height:1.1; margin-top:6px;">
                {money(monthly_avg_after_tax)}<span style="font-size:22px; font-weight:500; color:#4e6e55;"> / month</span>
            </div>
            <div style="font-size:14px; color:#6b7f70; margin-top:4px;">
                after taxes
            </div>
            {projection_line}
        </div>
        """,
        unsafe_allow_html=True,
    )

    estimate_help = (
        "Estimate using your selected federal rates. Each holding is classified "
        "as qualified or ordinary from its description — unfamiliar securities "
        "default to qualified. Tax-advantaged accounts (IRA, 401k, HSA) are not "
        "taxed. Gross dividend totals are always exact."
    )

    row = st.columns(3)
    row[0].metric("Total Dividends", money(total_dividends))
    row[1].metric("Estimated Taxes", money(estimated_taxes), help=estimate_help)
    row[2].metric("After-Tax Dividends", money(after_tax), help=estimate_help)

    st.divider()

    monthly = build_monthly_frame(df_filtered, start_date, end_date)
    avg_monthly_chart = monthly["after_tax"].mean()

    st.subheader("Monthly After-Tax Income")
    st.markdown(
        f"<div style='color:#2E7D32; font-size:15px; margin-top:-8px; margin-bottom:12px;'>"
        f"Avg <b>{money(avg_monthly_chart)}</b>/month</div>",
        unsafe_allow_html=True,
    )

    bars = (
        alt.Chart(monthly)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#2E7D32")
        .encode(
            x=alt.X(
                "month_label:N",
                sort=list(monthly["month_label"]),
                title=None,
                axis=alt.Axis(labelAngle=0, labelPadding=6),
            ),
            y=alt.Y(
                "after_tax:Q",
                title=None,
                axis=alt.Axis(format="$,.0f", grid=False, tickCount=4),
            ),
            tooltip=[
                alt.Tooltip("month_label:N", title="Month"),
                alt.Tooltip("after_tax:Q", title="After-tax", format="$,.2f"),
                alt.Tooltip("gross:Q", title="Gross", format="$,.2f"),
                alt.Tooltip("payouts:Q", title="Payouts"),
            ],
        )
    )

    avg_line = (
        alt.Chart(pd.DataFrame({"avg": [avg_monthly_chart]}))
        .mark_rule(strokeDash=[4, 4], color="#999")
        .encode(y="avg:Q")
    )

    chart = (bars + avg_line).properties(height=280).configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)


def _render_holdings_tab(df_filtered, after_tax_total) -> None:
    holdings = (
        df_filtered.groupby("symbol")
        .agg(
            name=("security_name", "first"),
            gross=("gross_dividend_amount", "sum"),
            tax=("estimated_tax_amount", "sum"),
            after_tax=("estimated_after_tax_amount", "sum"),
            payouts=("event_id", "count"),
            classification=("tax_classification", "first"),
            all_tax_advantaged=("tax_advantaged", "all"),
        )
        .reset_index()
        .sort_values("after_tax", ascending=False)
    )

    if holdings.empty or after_tax_total <= 0:
        st.info("No holdings to display for this range.")
        return

    holdings["share"] = holdings["after_tax"] / after_tax_total
    holdings["tax_tag"] = [
        "TF" if adv else ("Q" if cls.lower() == "qualified" else "O")
        for cls, adv in zip(holdings["classification"], holdings["all_tax_advantaged"])
    ]

    top = holdings.iloc[0]
    st.markdown(
        f"<div style='background:#eaf5ea; border:1px solid #c8e6c9; border-radius:8px; "
        f"padding:12px 16px; margin:14px 0 10px 0; font-size:14px; color:#1b5e20;'>"
        f"💡 <b>{top['symbol']}</b> is your biggest earner — "
        f"<b>{money(top['after_tax'])}</b> after-tax</div>",
        unsafe_allow_html=True,
    )

    # Concentration insight
    top_n = min(3, len(holdings))
    top_share = holdings["after_tax"].head(top_n).sum() / after_tax_total
    if top_share >= 0.60:
        conc_bg, conc_border, conc_color, conc_icon = (
            "#fff4e5",
            "#ffd9a8",
            "#8a4b00",
            "⚠️",
        )
        conc_text = (
            f"Top {top_n} holdings = <b>{top_share:.0%}</b> of your dividend income — "
            f"fairly concentrated"
        )
    else:
        conc_bg, conc_border, conc_color, conc_icon = (
            "#eaf5ea",
            "#c8e6c9",
            "#1b5e20",
            "✅",
        )
        conc_text = (
            f"Well diversified — top {top_n} holdings are only "
            f"<b>{top_share:.0%}</b> of your dividend income"
        )

    st.markdown(
        f"<div style='background:{conc_bg}; border:1px solid {conc_border}; "
        f"border-radius:8px; padding:12px 16px; margin:0 0 18px 0; "
        f"font-size:14px; color:{conc_color};'>"
        f"{conc_icon} {conc_text}</div>",
        unsafe_allow_html=True,
    )

    display = holdings.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "share": "Share",
            "after_tax": "After-Tax",
            "tax_tag": "Tax",
        }
    )[["Symbol", "Name", "Share", "After-Tax", "Tax"]]

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Symbol": st.column_config.TextColumn(width="small"),
            "Name": st.column_config.TextColumn(),
            "Share": st.column_config.ProgressColumn(
                "Share",
                format="%.0f%%",
                min_value=0,
                max_value=1,
            ),
            "After-Tax": st.column_config.NumberColumn(
                "After-Tax",
                format="$%.2f",
            ),
            "Tax": st.column_config.TextColumn(
                "Tax",
                width="small",
                help="Q = Qualified · O = Ordinary · TF = Tax-free (held in IRA / 401k / HSA)",
            ),
        },
    )

    st.caption(
        f"{len(holdings)} holdings · sorted by after-tax contribution"
    )


def _render_events_tab(df_filtered) -> None:
    events = df_filtered.sort_values("event_date", ascending=False).copy()
    events["event_date"] = events["event_date"].dt.strftime("%b %d, %Y")
    events["tax_classification"] = [
        "TF" if adv else ("Q" if cls == "qualified" else "O")
        for cls, adv in zip(
            events["tax_classification"].str.lower(),
            events["tax_advantaged"],
        )
    ]
    display = events.rename(
        columns={
            "event_date": "Date",
            "symbol": "Symbol",
            "security_name": "Security",
            "gross_dividend_amount": "Gross",
            "estimated_tax_amount": "Tax",
            "estimated_after_tax_amount": "After-Tax",
            "tax_classification": "Type",
        }
    )[["Date", "Symbol", "Security", "Gross", "Tax", "After-Tax", "Type"]]

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Gross": st.column_config.NumberColumn("Gross", format="$%.2f"),
            "Tax": st.column_config.NumberColumn("Tax", format="$%.2f"),
            "After-Tax": st.column_config.NumberColumn("After-Tax", format="$%.2f"),
            "Type": st.column_config.TextColumn(
                "Type",
                width="small",
                help="Q = Qualified · O = Ordinary · TF = Tax-free (held in IRA / 401k / HSA)",
            ),
        },
    )
    st.caption(f"{len(events)} events · sorted newest first")


def _card_header(step_label: str, title: str) -> None:
    st.markdown(
        f'<div style="font-size:11px; font-weight:600; letter-spacing:1.4px; '
        f'color:#6b7280; text-transform:uppercase;">{step_label}</div>'
        f'<div style="font-size:16px; font-weight:600; color:#0f172a; '
        f'margin-top:4px; margin-bottom:10px;">{title}</div>',
        unsafe_allow_html=True,
    )


def render_upload_view() -> None:
    # Transition animations for the staged reveal (TurboTax-style cascade).
    st.markdown(
        """
        <style>
        @keyframes divFadeUp {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .st-key-upload_chip,
        .st-key-step2_card,
        .st-key-cta_block {
            animation: divFadeUp 0.45s ease-out both;
        }
        .st-key-step2_card { animation-delay: 0.08s; }
        .st-key-cta_block  { animation-delay: 0.16s; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin: 8px 0 4px 0;">
            <div style="font-size:12px; font-weight:600; letter-spacing:1.6px;
                        color:#6b7280; text-transform:uppercase;">
                Dividend Income
            </div>
            <div style="font-size:34px; font-weight:700; color:#0f172a;
                        line-height:1.15; margin-top:6px;">
                See what your dividends pay you each month.
            </div>
            <div style="font-size:15px; color:#475569; margin-top:10px;
                        line-height:1.5;">
                Upload your Fidelity activity export for a clear monthly income view — with taxes accounted for.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_upload = "upload_csv_bytes" in st.session_state

    # ---------- State A: pending upload ----------
    if not has_upload:
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            _card_header("Step 1 of 2", "Upload activity CSV")
            uploaded_file = st.file_uploader(
                "Upload your Fidelity CSV",
                type=["csv"],
                label_visibility="collapsed",
                key="csv_uploader",
            )

            with st.expander("How do I get this CSV from Fidelity?"):
                st.markdown(
                    "**1.** Log in to your Fidelity account.\n\n"
                    "**2.** Go to your portfolio and open the **Activity & Orders** tab.\n\n"
                    "**3.** Click **More filters**.\n\n"
                    "**4.** Under transaction type, check only **Dividends/Interest** "
                    "and click **Apply**.\n\n"
                    "**5.** Choose your time period. Fidelity places limits on the date "
                    "ranges you can export, so for the best results we recommend selecting "
                    "**Q1 (Jan–Mar)** — it gives you a clean snapshot of the first three "
                    "months of 2026.\n\n"
                    "**6.** Click the **download icon** in the top-right corner of the "
                    "activity table — this saves the CSV.\n\n"
                    "Then upload that file above."
                )

        if uploaded_file is not None:
            st.session_state.upload_csv_bytes = uploaded_file.getvalue()
            st.session_state.upload_csv_name = uploaded_file.name
            st.session_state.upload_csv_size = uploaded_file.size
            st.rerun()

        st.markdown(
            '<div style="text-align:center; font-size:12px; color:#64748b; '
            'margin-top:18px;">🔒 Your CSV is processed in memory and never stored. Nothing is saved on the server.</div>',
            unsafe_allow_html=True,
        )
        return

    # ---------- State B: uploaded ----------
    csv_name = st.session_state.upload_csv_name
    size_kb = st.session_state.upload_csv_size / 1024

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    # Collapsed Step 1 chip
    with st.container(border=True, key="upload_chip"):
        chip_left, chip_right = st.columns([5, 1])
        with chip_left:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="
                        width: 28px; height: 28px; border-radius: 50%;
                        background-color: #ecfdf5; color: #0f766e;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 15px; font-weight: 700; flex-shrink: 0;
                    ">✓</div>
                    <div style="min-width:0;">
                        <div style="font-weight: 600; color: #0f172a; font-size: 14px;
                                    overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                            {csv_name}
                        </div>
                        <div style="font-size: 12px; color: #64748b; margin-top: 1px;">
                            {size_kb:,.1f} KB &middot; uploaded
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with chip_right:
            if st.button("Change file", type="tertiary", key="change_file"):
                for k in ("upload_csv_bytes", "upload_csv_name", "upload_csv_size", "csv_uploader"):
                    st.session_state.pop(k, None)
                st.rerun()

    # Step 2 card
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    with st.container(border=True, key="step2_card"):
        _card_header("Step 2 of 2", "Set your tax rates")

        col1, col2 = st.columns(2)

        with col1:
            ordinary_rate = st.selectbox(
                "Ordinary Income Tax Rate",
                options=["10%", "12%", "22%", "24%", "32%", "35%", "37%"],
                index=None,
                placeholder="Select a rate",
            )

        with col2:
            qualified_rate = st.selectbox(
                "Qualified Dividend Tax Rate",
                options=["0%", "15%", "20%"],
                index=None,
                placeholder="Select a rate",
                help="The lower preferential rate for most US stock and ETF dividends. Most investors are at 15%.",
            )

        current_rates = (ordinary_rate, qualified_rate)
        both_selected = ordinary_rate is not None and qualified_rate is not None

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        clicked = st.button(
            "Generate report",
            type="primary",
            use_container_width=True,
            disabled=not both_selected,
        )

        if not both_selected:
            st.markdown(
                '<div style="text-align:center; font-size:12px; color:#64748b; '
                'margin-top:6px;">Select both tax rates to continue</div>',
                unsafe_allow_html=True,
            )

        with st.expander("How do I pick my rates?"):
            st.markdown(
                "**Ordinary income rate** — your normal federal tax bracket "
                "(2025, single filer):\n\n"
                "| Rate | Taxable income |\n"
                "|---|---|\n"
                "| 10–12% | under \$48K |\n"
                "| 22% | \$48K–\$103K |\n"
                "| 24% | \$103K–\$197K |\n"
                "| 32–35% | \$197K–\$626K |\n"
                "| 37% | above \$626K |\n\n"
                "_Married filing jointly? Roughly double those thresholds._\n\n"
                "---\n\n"
                "**Qualified dividend rate** — a lower preferential rate:\n\n"
                "| Rate | Taxable income (single) |\n"
                "|---|---|\n"
                "| 0% | under \$48K |\n"
                "| 15% | \$48K–\$518K |\n"
                "| 20% | above \$518K |\n\n"
                "Most investors are at **15%**.\n\n"
                "---\n\n"
                "**Why two rates?** Most dividends from US stocks and ETFs are "
                "_qualified_ and get the lower rate. Money market funds (like "
                "**SPAXX**), REITs, and short-held stocks are _ordinary_ and taxed "
                "at your regular income rate. Your CSV already tags each payout, "
                "so the dashboard applies the right one automatically."
            )

    st.markdown(
        '<div style="text-align:center; font-size:12px; color:#64748b; '
        'margin-top:14px;">🔒 Your CSV is processed in memory and never stored. Nothing is saved on the server.</div>',
        unsafe_allow_html=True,
    )

    if clicked:
        try:
            with st.spinner("Processing your dividend data…"):
                start = time.time()
                ord_decimal = int(ordinary_rate.rstrip("%")) / 100
                qual_decimal = int(qualified_rate.rstrip("%")) / 100
                cleaned_bytes = clean_fidelity_csv(
                    st.session_state.upload_csv_bytes,
                    ordinary_rate=ord_decimal,
                    qualified_rate=qual_decimal,
                )
                elapsed = time.time() - start
                if elapsed < 0.8:
                    time.sleep(0.8 - elapsed)
        except Exception as e:
            st.error(f"Couldn't read this file: {e}")
            return
        st.session_state.csv_bytes = cleaned_bytes
        st.session_state.generated_rates = current_rates
        st.session_state.view = "report"
        st.rerun()


st.set_page_config(page_title="Dividend Income Dashboard", layout="centered")


def check_password() -> bool:
    """Gate the app behind a single shared password stored in st.secrets.

    If no password is configured (e.g. running locally without a secrets file),
    the app is open — this preserves the local dev workflow.
    """
    try:
        expected = st.secrets["app_password"]
    except Exception:
        expected = ""
    if not expected:
        return True
    if st.session_state.get("password_correct"):
        return True
    pw = st.text_input("Enter access password", type="password")
    if pw:
        if pw == expected:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


if not check_password():
    st.stop()

if "view" not in st.session_state:
    st.session_state.view = "upload"

if st.session_state.view == "report":
    render_report_view()
    st.stop()

render_upload_view()
