from __future__ import annotations

from datetime import datetime
from html import escape
from random import shuffle

import streamlit as st

from src.document_pipeline import load_document_chunks
from src.fund_data import export_canonical_fund_dataset, load_fund_schemes
from src.rag_engine import (
    MODE_ADVISOR,
    MODE_AUDITOR,
    MODE_RESEARCHER,
    build_grounded_answer,
    sample_rag_questions,
)
from src.recommendation_engine import (
    InvestorProfile,
    action_checklist,
    investor_persona,
    recommend_schemes,
    risk_bucket,
    strategy_blueprint,
)
from src.vector_store import load_or_build_chunk_vector_index


st.set_page_config(
    page_title="Mutual Fund Planner",
    page_icon="MF",
    layout="wide",
)


FLASH_QUOTES = (
    "Diversify before you optimise.",
    "A lower cost today gives compounding more room tomorrow.",
    "Time horizon decides whether volatility is noise or real stress.",
    "Liquidity belongs near goals that cannot wait.",
    "A shortlist becomes investable only after one last factsheet check.",
    "Good allocation usually matters more than chasing the hottest recent return.",
)

FLASH_QUOTE_BAG: list[str] = []


PROFILE_DEFAULTS = {
    "profile_age_input": "",
    "profile_goal": None,
    "profile_risk_appetite": None,
    "profile_horizon_input": "",
    "profile_investment_mode": None,
    "profile_amount_input": "",
    "profile_needs_tax_saving_choice": None,
}

APP_DEFAULTS = {
    "risk_notice_acknowledged": False,
}

RISK_NOTICE_POINTS = (
    "This app is an educational recommendation assistant, not personalized financial advice.",
    "Mutual funds are subject to market risk, and returns are never guaranteed.",
    "Please read the latest factsheet, scheme information document, riskometer, cost details, and lock-in conditions before investing.",
    "Use the shortlist as a starting point for research, not as an automatic buy signal.",
)

GOAL_OPTIONS = (
    "Wealth Creation",
    "Emergency Fund",
    "Capital Preservation",
    "Income Stability",
    "Retirement",
    "Tax Saving",
)

RISK_APPETITE_OPTIONS = ("Low", "Moderate", "High")
INVESTMENT_MODE_OPTIONS = ("SIP", "Lump Sum")
YES_NO_OPTIONS = ("Yes", "No")

PROFILE_EXAMPLES = {
    "age": "28",
    "goal": "Wealth Creation",
    "risk_appetite": "Moderate",
    "horizon_years": "7",
    "investment_mode": "SIP",
    "amount": "10,000",
    "needs_tax_saving": "No",
}


PROFILE_PRESETS = (
    {
        "label": "Safe Parking",
        "description": "Lower-risk cash and short-duration mindset.",
        "values": {
            "profile_age_input": "32",
            "profile_goal": "Emergency Fund",
            "profile_risk_appetite": "Low",
            "profile_horizon_input": "2",
            "profile_investment_mode": "SIP",
            "profile_amount_input": "5000",
            "profile_needs_tax_saving_choice": "No",
        },
    },
    {
        "label": "Balanced Builder",
        "description": "Steady growth with controlled volatility.",
        "values": {
            "profile_age_input": "29",
            "profile_goal": "Wealth Creation",
            "profile_risk_appetite": "Moderate",
            "profile_horizon_input": "7",
            "profile_investment_mode": "SIP",
            "profile_amount_input": "12000",
            "profile_needs_tax_saving_choice": "No",
        },
    },
    {
        "label": "Long Run",
        "description": "Long-horizon equity-led compounding.",
        "values": {
            "profile_age_input": "34",
            "profile_goal": "Retirement",
            "profile_risk_appetite": "High",
            "profile_horizon_input": "15",
            "profile_investment_mode": "SIP",
            "profile_amount_input": "20000",
            "profile_needs_tax_saving_choice": "No",
        },
    },
    {
        "label": "Tax Focus",
        "description": "Tax-aware shortlist starting point.",
        "values": {
            "profile_age_input": "31",
            "profile_goal": "Tax Saving",
            "profile_risk_appetite": "Moderate",
            "profile_horizon_input": "5",
            "profile_investment_mode": "Lump Sum",
            "profile_amount_input": "25000",
            "profile_needs_tax_saving_choice": "Yes",
        },
    },
)


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def format_rupees(value: float | None) -> str:
    if value is None or value <= 0:
        return "N/A"
    return f"Rs. {value:,.0f}"


def format_decimal(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def format_date_label(value: str | None) -> str:
    if not value:
        return "N/A"
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        return value


def format_crores(value: float | None) -> str:
    if value is None or value <= 0:
        return "N/A"
    return f"Rs. {value:,.0f} Cr."


def prettify_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def clean_frontend_text(text: str) -> str:
    replacements = {
        "Master NAV or AUM data is not yet linked for this scheme.": (
            "Some supporting details are still limited, so confirm the latest official factsheet before investing."
        ),
        "Displayed NAV is a synthetic peer-based estimate used to keep dataset coverage complete.": (
            "Some displayed values may be estimated for continuity, so verify the latest official fund information before investing."
        ),
        "No major red flag from the current profile, but scheme documents should still be checked.": (
            "No major profile mismatch stands out, but the latest scheme factsheet should still be checked."
        ),
        "scheme documents": "scheme factsheet",
        "project documentation": "supporting research",
        "local knowledge corpus": "supporting research",
        "managed knowledge corpus": "trusted supporting research",
        "structured scheme data": "scheme-level details",
        "structured scheme metrics": "scheme metrics",
        "synthetic NAV": "estimated NAV",
        "master-data": "supporting-data",
        "master link": "supporting-data match",
        "NAV source": "NAV status",
        "data quality": "information quality",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return " ".join(cleaned.split())


def render_markdown_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        st.write("No comparison data available.")
        return

    headers = list(rows[0].keys())
    head_html = "".join(
        f"<div class='comparison-cell comparison-head'>{escape(header)}</div>"
        for header in headers
    )

    row_html_parts: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cell_html_parts: list[str] = []
        for header_index, header in enumerate(headers):
            value = str(row.get(header, ""))
            cell_classes = ["comparison-cell"]
            if header_index == 0:
                cell_classes.append("lead")
            if header.lower() == "fit":
                cell_classes.append("signal")
            if any(token in header.lower() for token in ("return", "expense", "minimum")):
                cell_classes.append("metric")

            cell_html_parts.append(
                (
                    f"<div class='{' '.join(cell_classes)}'>"
                    f"<div class='comparison-text'>{escape(value)}</div>"
                    "</div>"
                )
            )

        row_html_parts.append(
            (
                f"<div class='comparison-row' style='--row-index:{row_index}; "
                f"--comparison-columns:{len(headers)};'>"
                + "".join(cell_html_parts)
                + "</div>"
            )
        )

    st.markdown(
        (
            "<section class='comparison-stage'>"
            "<div class='comparison-orbit comparison-orbit-a'></div>"
            "<div class='comparison-orbit comparison-orbit-b'></div>"
            "<div class='comparison-board-shell'>"
            f"<div class='comparison-row comparison-row-head' style='--row-index:0; --comparison-columns:{len(headers)};'>{head_html}</div>"
            + "".join(row_html_parts)
            + "</div></section>"
        ),
        unsafe_allow_html=True,
    )


def theme_html(items: list[str]) -> str:
    return "".join(f"<span class='hero-chip'>{escape(item)}</span>" for item in items)


def parse_optional_int(value: str | None) -> int | None:
    cleaned = (value or "").replace(",", "").strip()
    if not cleaned:
        return None
    if not cleaned.isdigit():
        raise ValueError("Use digits only.")
    return int(cleaned)


def build_profile_from_inputs(
    age_text: str,
    goal: str | None,
    risk_appetite: str | None,
    horizon_text: str,
    investment_mode: str | None,
    amount_text: str,
    needs_tax_saving_choice: str | None,
) -> tuple[InvestorProfile | None, list[str], list[str]]:
    missing_fields: list[str] = []
    issues: list[str] = []

    if not age_text.strip():
        missing_fields.append("Age")
    if goal is None:
        missing_fields.append("Primary Goal")
    if risk_appetite is None:
        missing_fields.append("Risk Appetite")
    if not horizon_text.strip():
        missing_fields.append("Investment Horizon")
    if investment_mode is None:
        missing_fields.append("Investment Mode")
    if not amount_text.strip():
        missing_fields.append("Planned Investment Amount")
    if needs_tax_saving_choice is None:
        missing_fields.append("Tax Saving Preference")

    age: int | None = None
    horizon_years: int | None = None
    amount: int | None = None

    try:
        age = parse_optional_int(age_text)
    except ValueError:
        issues.append("Age must be entered using digits only, for example `28`.")
    else:
        if age is not None and not 18 <= age <= 70:
            issues.append("Age must be between 18 and 70.")

    try:
        horizon_years = parse_optional_int(horizon_text)
    except ValueError:
        issues.append("Investment horizon must use digits only, for example `7`.")
    else:
        if horizon_years is not None and not 1 <= horizon_years <= 20:
            issues.append("Investment horizon must be between 1 and 20 years.")

    try:
        amount = parse_optional_int(amount_text)
    except ValueError:
        issues.append("Planned investment amount must use digits only, for example `10000`.")
    else:
        if amount is not None and amount < 1000:
            issues.append("Planned investment amount must be at least Rs. 1,000.")

    if missing_fields or issues:
        return None, missing_fields, issues

    return (
        InvestorProfile(
            age=age or 0,
            goal=goal or "",
            risk_appetite=risk_appetite or "",
            horizon_years=horizon_years or 0,
            investment_mode=investment_mode or "",
            amount=amount or 0,
            needs_tax_saving=needs_tax_saving_choice == "Yes",
        ),
        [],
        [],
    )


def ensure_profile_state() -> None:
    for key, value in PROFILE_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    if "flash_quote" not in st.session_state:
        st.session_state["flash_quote"] = pick_flash_quote()


def pick_flash_quote() -> str:
    global FLASH_QUOTE_BAG
    if not FLASH_QUOTE_BAG:
        FLASH_QUOTE_BAG = list(FLASH_QUOTES)
        shuffle(FLASH_QUOTE_BAG)
    return FLASH_QUOTE_BAG.pop()


def ensure_app_state() -> None:
    for key, value in APP_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def apply_profile_preset(values: dict[str, object]) -> None:
    for key, value in values.items():
        st.session_state[key] = value
    st.rerun()


@st.dialog("Important Notice", width="large")
def show_risk_notice_dialog() -> None:
    st.markdown(
        """
        <div class="risk-notice-shell">
            <div class="risk-notice-kicker">Please read before you continue</div>
            <h3 class="risk-notice-title">This tool helps you explore mutual fund options, but it is not a substitute for professional financial advice.</h3>
            <p class="risk-notice-copy">
                Treat every result here as decision-support guidance. Always verify the latest official fund documents,
                suitability details, and disclosures before making any investment decision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for item in RISK_NOTICE_POINTS:
        st.markdown(f"- {item}")

    st.warning("Mutual funds are subject to market risk. Read all scheme-related documents carefully before investing.")
    if st.button(
        "I Understand, Continue",
        key="dismiss_risk_notice",
        use_container_width=True,
    ):
        st.session_state["risk_notice_acknowledged"] = True
        st.rerun()


def render_metric_grid(items: list[tuple[str, str]], columns: int, compact: bool = False) -> None:
    card_html = "".join(
        (
            "<div class='metric-card'>"
            f"<div class='metric-card-label'>{escape(label)}</div>"
            f"<div class='metric-card-value{' compact' if compact else ''}'>{escape(value)}</div>"
            "</div>"
        )
        for label, value in items
    )
    st.markdown(
        (
            f"<div class='metric-grid' "
            f"style='grid-template-columns: repeat({columns}, minmax(150px, 1fr));'>"
            f"{card_html}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_note_grid(items: list[str] | tuple[str, ...], tone: str = "positive", columns: int = 2) -> None:
    if not items:
        return

    card_html = "".join(
        (
            f"<article class='note-card {tone}'>"
            f"<div class='note-badge'>{index:02d}</div>"
            f"<div class='note-text'>{escape(clean_frontend_text(item))}</div>"
            "</article>"
        )
        for index, item in enumerate(items, start=1)
    )
    st.markdown(
        (
            f"<div class='note-grid' "
            f"style='grid-template-columns: repeat({columns}, minmax(220px, 1fr));'>"
            f"{card_html}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_flash_quotes() -> None:
    quote = st.session_state.get("flash_quote", FLASH_QUOTES[0])
    st.markdown(
        f"""
        <section class="flash-shell">
            <div class="flash-viewport">
                <div class="flash-runner">
                    <span class="flash-pill">"{escape(quote)}"</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_profile_presets() -> None:
    st.markdown("**Quick Modes**")
    preset_cols = st.columns(len(PROFILE_PRESETS), gap="small")
    for index, (column, preset) in enumerate(zip(preset_cols, PROFILE_PRESETS), start=1):
        with column:
            if st.button(preset["label"], key=f"profile_preset_{index}", use_container_width=True):
                apply_profile_preset(preset["values"])
            st.caption(preset["description"])


def render_advice_panel(
    kicker: str,
    title: str,
    subtitle: str,
    items: list[str] | tuple[str, ...],
    tone: str,
    columns: int,
) -> str:
    cards_html = "".join(
        (
            f"<article class='advice-card {tone}'>"
            f"<div class='advice-index'>{index:02d}</div>"
            f"<div class='advice-text'>{escape(clean_frontend_text(item))}</div>"
            "</article>"
        )
        for index, item in enumerate(items, start=1)
    )
    return (
        f"<section class='advice-panel {tone}'>"
        f"<div class='advice-head'>"
        f"<div class='advice-kicker'>{escape(kicker)}</div>"
        f"<h4 class='advice-title'>{escape(title)}</h4>"
        f"<p class='advice-copy'>{escape(subtitle)}</p>"
        f"</div>"
        f"<div class='advice-grid' style='grid-template-columns: repeat({columns}, minmax(220px, 1fr));'>"
        f"{cards_html}"
        f"</div>"
        f"</section>"
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-top: #fbf7ef;
            --bg-bottom: #eee7db;
            --ink: #14213d;
            --muted: #5b6575;
            --panel: rgba(255, 255, 255, 0.74);
            --line: rgba(20, 33, 61, 0.10);
            --accent: #d68c45;
            --accent-soft: rgba(214, 140, 69, 0.18);
            --teal: #0f766e;
            --display: "Baskerville", "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
            --serif: "Baskerville", "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
            --sans: "Avenir Next", "Gill Sans", "Trebuchet MS", "Segoe UI", sans-serif;
        }

        html, body, [class*="css"] {
            font-family: var(--sans);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 0% 0%, rgba(214, 140, 69, 0.16), transparent 30%),
                radial-gradient(circle at 100% 0%, rgba(15, 118, 110, 0.12), transparent 26%),
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .block-container,
        [data-testid="stMainBlockContainer"] {
            max-width: 1450px !important;
            margin: 0 auto;
            padding-top: 0.85rem;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-bottom: 2.7rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        h1, h2 {
            font-family: var(--serif);
        }

        p, label, [data-testid="stMarkdownContainer"], .stCaption {
            color: #2a3443;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.85rem 1rem;
            box-shadow: 0 18px 36px rgba(20, 33, 61, 0.06);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-family: var(--serif);
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.6rem;
            padding-bottom: 1rem;
        }

        [data-testid="stTabs"] [role="tab"] {
            background: rgba(255, 255, 255, 0.70);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.7rem 1rem;
            box-shadow: 0 12px 24px rgba(20, 33, 61, 0.05);
            color: var(--ink) !important;
            font-weight: 800;
            transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease, color 160ms ease;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: linear-gradient(135deg, #14213d 0%, #1f3159 100%);
            color: #fff8eb !important;
            border-color: rgba(20, 33, 61, 0.18);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.14),
                0 16px 28px rgba(20, 33, 61, 0.16);
        }

        [data-testid="stTabs"] [aria-selected="true"] p,
        [data-testid="stTabs"] [aria-selected="true"] span,
        [data-testid="stTabs"] [aria-selected="true"] div {
            color: #fff8eb !important;
        }

        [data-testid="stTabs"] [role="tab"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 28px rgba(20, 33, 61, 0.10);
        }

        [data-testid="stTabs"] [role="tab"]:focus-visible {
            outline: 2px solid rgba(214, 140, 69, 0.65);
            outline-offset: 2px;
        }

        [data-testid="stTextInputRootElement"] > div,
        [data-testid="stNumberInputContainer"] > div,
        div[data-baseweb="select"] > div,
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.82);
            border-radius: 16px;
            border: 1px solid rgba(20, 33, 61, 0.12);
        }

        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid rgba(20, 33, 61, 0.08);
            border-radius: 24px;
            box-shadow: 0 18px 40px rgba(20, 33, 61, 0.06);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-4px);
            border-color: rgba(214, 140, 69, 0.22);
            box-shadow: 0 24px 44px rgba(20, 33, 61, 0.10);
        }

        [data-testid="stButton"] > button {
            min-height: 3.35rem;
            border: 1px solid rgba(20, 33, 61, 0.14);
            border-radius: 18px;
            background: linear-gradient(180deg, #fffdf8 0%, #f0e6d6 100%);
            color: var(--ink);
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow:
                0 10px 0 rgba(20, 33, 61, 0.14),
                0 24px 32px rgba(20, 33, 61, 0.10);
            transform: translateY(0) rotateX(0deg);
            transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
        }

        [data-testid="stButton"] > button:hover {
            background: linear-gradient(180deg, #fffaf0 0%, #eed9bc 100%);
            transform: translateY(-4px) rotateX(10deg);
            box-shadow:
                0 14px 0 rgba(20, 33, 61, 0.12),
                0 30px 38px rgba(20, 33, 61, 0.14);
        }

        [data-testid="stButton"] > button:active {
            transform: translateY(6px);
            box-shadow:
                0 4px 0 rgba(20, 33, 61, 0.16),
                0 14px 18px rgba(20, 33, 61, 0.08);
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            margin-bottom: 1rem;
            padding: 1.8rem 1.7rem 2rem 1.7rem;
            border: 1px solid rgba(20, 33, 61, 0.10);
            border-radius: 30px;
            background:
                linear-gradient(135deg, rgba(20, 33, 61, 0.96) 0%, rgba(31, 49, 89, 0.94) 58%, rgba(214, 140, 69, 0.86) 100%);
            box-shadow: 0 28px 60px rgba(20, 33, 61, 0.18);
            color: #f9f4ec;
        }

        .hero-shell::before {
            content: "";
            position: absolute;
            inset: -10% auto auto 55%;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.18) 0%, transparent 68%);
            animation: heroFloat 8s ease-in-out infinite;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -80px -80px auto;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
            filter: blur(2px);
            animation: heroPulse 10s ease-in-out infinite;
        }

        .eyebrow {
            margin-bottom: 0.6rem;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(249, 244, 236, 0.78);
        }

        .hero-title {
            max-width: 1080px;
            margin: 0;
            font-family: var(--serif);
            font-size: clamp(2.4rem, 4.2vw, 4.4rem);
            line-height: 1.02;
            letter-spacing: -0.04em;
        }

        .hero-copy {
            max-width: 960px;
            margin-top: 0.9rem;
            margin-bottom: 0;
            font-size: 1.08rem;
            line-height: 1.6;
            color: rgba(249, 244, 236, 0.88);
        }

        .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.15rem;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.48rem 0.85rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.16);
            font-size: 0.88rem;
            color: #fbf7ef;
            transform: translateY(0);
            transition: transform 160ms ease, background 160ms ease;
        }

        .hero-chip:hover {
            transform: translateY(-3px);
            background: rgba(255, 255, 255, 0.18);
        }

        [data-testid="stDialog"] > div {
            background: linear-gradient(180deg, rgba(255, 250, 244, 0.98) 0%, rgba(250, 242, 232, 0.98) 100%);
            border: 1px solid rgba(20, 33, 61, 0.08);
            box-shadow: 0 28px 72px rgba(20, 33, 61, 0.18);
        }

        .risk-notice-shell {
            border: 1px solid rgba(214, 140, 69, 0.22);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(245, 232, 214, 0.92));
            border-radius: 24px;
            padding: 1.2rem 1.25rem 1rem 1.25rem;
            margin-bottom: 0.45rem;
        }

        .risk-notice-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--teal);
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .risk-notice-title {
            margin: 0;
            font-family: var(--serif);
            font-size: 1.45rem;
            line-height: 1.18;
            color: var(--ink);
        }

        .risk-notice-copy {
            margin: 0.7rem 0 0 0;
            color: var(--muted);
            line-height: 1.72;
            font-size: 0.98rem;
        }

        .panel-card {
            padding: 1.1rem 1.15rem;
            border-radius: 24px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 18px 36px rgba(20, 33, 61, 0.05);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .panel-card:hover,
        .support-card:hover,
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 44px rgba(20, 33, 61, 0.12);
        }

        .panel-kicker {
            margin-bottom: 0.45rem;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--teal);
        }

        .panel-title {
            margin: 0;
            font-family: var(--serif);
            font-size: 1.75rem;
            line-height: 1.06;
            color: var(--ink);
        }

        .panel-copy {
            margin-top: 0.7rem;
            margin-bottom: 0;
            color: var(--muted);
            line-height: 1.6;
        }

        .recommend-heading {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.7rem;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 118, 110, 0.14);
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.14) 0%, rgba(255, 255, 255, 0.92) 100%);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.72),
                0 12px 24px rgba(20, 33, 61, 0.07);
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--teal);
        }

        .recommend-title {
            margin: 0;
            font-family: var(--display);
            font-size: clamp(2rem, 2.2vw, 2.65rem);
            line-height: 1.02;
            letter-spacing: -0.035em;
            color: var(--ink);
        }

        .recommend-meta {
            margin-top: 0.55rem;
            color: #687285;
            font-size: 1rem;
            line-height: 1.55;
        }

        .section-note {
            margin-top: 0.55rem;
            padding-left: 0.95rem;
            border-left: 4px solid var(--accent-soft);
            color: var(--muted);
            line-height: 1.55;
        }

        .support-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.66);
            box-shadow: 0 16px 34px rgba(20, 33, 61, 0.05);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .support-title {
            margin: 0;
            font-weight: 800;
            color: var(--ink);
        }

        .support-meta {
            margin-top: 0.35rem;
            margin-bottom: 0.6rem;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .hint-strip {
            margin-top: 0.75rem;
            padding: 0.85rem 1rem;
            border-radius: 18px;
            background: rgba(214, 140, 69, 0.10);
            border: 1px solid rgba(214, 140, 69, 0.16);
            color: #5d4631;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
        }

        .metric-grid {
            display: grid;
            gap: 0.95rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            min-height: 118px;
            padding: 0.95rem 1rem 1rem;
            border-radius: 22px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 16px 34px rgba(20, 33, 61, 0.05);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .metric-card-label {
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--muted);
        }

        .metric-card-value {
            margin-top: 0.5rem;
            font-family: var(--serif);
            font-size: clamp(1.5rem, 1.8vw, 2.25rem);
            line-height: 1.08;
            color: var(--ink);
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .metric-card-value.compact {
            font-size: clamp(1.15rem, 1.35vw, 1.55rem);
        }

        .flash-shell {
            position: relative;
            overflow: hidden;
            margin-bottom: 1.2rem;
            padding: 1rem 0.95rem;
            border-radius: 24px;
            border: 1px solid rgba(20, 33, 61, 0.10);
            background:
                radial-gradient(circle at 100% 0%, rgba(214, 140, 69, 0.12), transparent 28%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(247, 241, 230, 0.88) 100%);
            box-shadow: 0 18px 40px rgba(20, 33, 61, 0.06);
        }

        .flash-shell::after {
            content: "";
            position: absolute;
            inset: auto -30px -40px auto;
            width: 160px;
            height: 160px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(20, 33, 61, 0.10) 0%, transparent 72%);
        }

        .flash-viewport {
            position: relative;
            z-index: 1;
            overflow: hidden;
            padding: 0.15rem 0;
        }

        .flash-runner {
            display: inline-flex;
            align-items: center;
            padding-left: 100%;
            width: max-content;
            animation: flashScroll 22s linear infinite;
        }

        .flash-shell:hover .flash-runner {
            animation-play-state: paused;
        }

        .flash-pill {
            display: inline-flex;
            align-items: center;
            min-height: 3.6rem;
            padding: 0.9rem 1.45rem;
            border-radius: 999px;
            border: 1px solid rgba(20, 33, 61, 0.08);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(246, 237, 223, 0.94) 100%);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.7),
                0 12px 24px rgba(20, 33, 61, 0.08);
            font-family: "Didot", "Cochin", "Iowan Old Style", "Book Antiqua", var(--display);
            font-size: clamp(1.16rem, 1.65vw, 1.42rem);
            font-style: italic;
            font-weight: 500;
            line-height: 1.35;
            letter-spacing: 0.01em;
            color: var(--ink);
            white-space: nowrap;
            transform-origin: center;
            transition: transform 180ms ease, box-shadow 180ms ease;
            transform: rotate(-1.4deg);
        }

        .flash-pill:hover {
            transform: translateY(-3px) scale(1.02) rotate(0deg);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.74),
                0 18px 28px rgba(20, 33, 61, 0.12);
        }

        .section-banner {
            position: relative;
            overflow: hidden;
            margin-bottom: 0.9rem;
            padding: 1rem 1.05rem 1.05rem;
            border-radius: 24px;
            border: 1px solid rgba(20, 33, 61, 0.08);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.80) 0%, rgba(246, 239, 228, 0.74) 100%);
            box-shadow: 0 18px 34px rgba(20, 33, 61, 0.06);
        }

        .section-banner::after {
            content: "";
            position: absolute;
            inset: auto -24px -42px auto;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(214, 140, 69, 0.16) 0%, transparent 72%);
        }

        .section-banner.strategy {
            background:
                radial-gradient(circle at 100% 0%, rgba(15, 118, 110, 0.14), transparent 34%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.84) 0%, rgba(235, 246, 242, 0.80) 100%);
        }

        .section-banner.assistant {
            background:
                radial-gradient(circle at 100% 0%, rgba(59, 130, 246, 0.12), transparent 34%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.84) 0%, rgba(239, 243, 252, 0.82) 100%);
        }

        .section-banner-kicker {
            position: relative;
            z-index: 1;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--teal);
        }

        .section-banner-title {
            position: relative;
            z-index: 1;
            margin: 0.28rem 0 0;
            font-family: var(--display);
            font-size: clamp(1.7rem, 2vw, 2.2rem);
            line-height: 1.06;
            color: var(--ink);
        }

        .section-banner-copy {
            position: relative;
            z-index: 1;
            margin: 0.55rem 0 0;
            max-width: 72ch;
            color: #5d6878;
            line-height: 1.65;
        }

        .comparison-stage {
            position: relative;
            overflow: hidden;
            margin-top: 1rem;
            padding: 1.2rem;
            border-radius: 30px;
            border: 1px solid rgba(20, 33, 61, 0.10);
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(243, 236, 224, 0.90) 100%);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.68),
                0 24px 46px rgba(20, 33, 61, 0.10);
            perspective: 1800px;
        }

        .comparison-stage::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(115deg, transparent 14%, rgba(255, 255, 255, 0.34) 30%, transparent 46%);
            transform: translateX(-120%);
            animation: comparisonSweep 7s ease-in-out infinite;
        }

        .comparison-orbit {
            position: absolute;
            border-radius: 50%;
            filter: blur(4px);
            opacity: 0.65;
            animation: comparisonFloat 10s ease-in-out infinite;
        }

        .comparison-orbit-a {
            top: -50px;
            right: 90px;
            width: 170px;
            height: 170px;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.16) 0%, transparent 70%);
        }

        .comparison-orbit-b {
            right: -30px;
            bottom: -55px;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(214, 140, 69, 0.18) 0%, transparent 72%);
            animation-delay: -3s;
        }

        .comparison-board-shell {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            overflow-x: auto;
            padding-bottom: 0.25rem;
        }

        .comparison-row {
            display: grid;
            grid-template-columns: repeat(var(--comparison-columns), minmax(150px, 1fr));
            gap: 0.8rem;
            min-width: 1220px;
            transform-style: preserve-3d;
            animation: comparisonRise 620ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
            animation-delay: calc(var(--row-index) * 70ms);
        }

        .comparison-row-head {
            min-width: 1220px;
        }

        .comparison-row:not(.comparison-row-head) {
            padding: 0.22rem;
            border-radius: 26px;
            background: rgba(255, 255, 255, 0.30);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
            transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease;
        }

        .comparison-row:not(.comparison-row-head):hover {
            transform: translateY(-6px) rotateX(4deg) rotateY(-2deg);
            background: rgba(255, 255, 255, 0.42);
            box-shadow:
                0 24px 40px rgba(20, 33, 61, 0.12),
                inset 0 1px 0 rgba(255, 255, 255, 0.72);
        }

        .comparison-cell {
            min-height: 106px;
            padding: 0.92rem 0.95rem 0.9rem;
            border-radius: 22px;
            border: 1px solid rgba(20, 33, 61, 0.08);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 243, 234, 0.93) 100%);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.72),
                0 14px 28px rgba(20, 33, 61, 0.08);
        }

        .comparison-head {
            min-height: auto;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.8rem 0.9rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #14213d 0%, #32466f 100%);
            color: #f7f0e6;
            font-size: 0.78rem;
            font-weight: 500;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            text-align: center;
            box-shadow: 0 16px 26px rgba(20, 33, 61, 0.16);
        }

        .comparison-text {
            display: flex;
            align-items: center;
            min-height: 100%;
            font-family: var(--sans);
            font-size: 1.02rem;
            font-weight: 400;
            line-height: 1.45;
            letter-spacing: -0.01em;
            color: #223247;
        }

        .comparison-cell.lead .comparison-text {
            font-family: var(--display);
            font-size: 1.2rem;
            font-weight: 400;
            line-height: 1.25;
            color: var(--ink);
        }

        .comparison-cell.signal .comparison-text {
            color: #0f766e;
        }

        .comparison-cell.metric .comparison-text {
            font-family: var(--serif);
            font-size: 1.12rem;
            font-weight: 400;
            color: #2f3f56;
        }

        .strategy-stage {
            position: relative;
            overflow: hidden;
            margin-top: 1rem;
            padding: 1.15rem;
            border-radius: 30px;
            border: 1px solid rgba(20, 33, 61, 0.08);
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(237, 248, 244, 0.90) 100%);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.7),
                0 24px 44px rgba(20, 33, 61, 0.08);
        }

        .strategy-stage::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, transparent 12%, rgba(255, 255, 255, 0.26) 30%, transparent 52%);
            transform: translateX(-120%);
            animation: strategySweep 8s ease-in-out infinite;
        }

        .strategy-stage-orbit {
            position: absolute;
            border-radius: 50%;
            filter: blur(5px);
            opacity: 0.7;
            animation: strategyFloat 9s ease-in-out infinite;
        }

        .strategy-stage-orbit-a {
            top: -40px;
            right: 70px;
            width: 170px;
            height: 170px;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.16) 0%, transparent 72%);
        }

        .strategy-stage-orbit-b {
            left: -35px;
            bottom: -60px;
            width: 210px;
            height: 210px;
            background: radial-gradient(circle, rgba(214, 140, 69, 0.16) 0%, transparent 72%);
            animation-delay: -3s;
        }

        .strategy-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1rem;
        }

        .strategy-signal-card {
            position: relative;
            overflow: hidden;
            padding: 1.15rem 1.15rem 1.2rem;
            border-radius: 28px;
            border: 1px solid rgba(20, 33, 61, 0.08);
            background:
                radial-gradient(circle at 100% 0%, rgba(15, 118, 110, 0.16), transparent 26%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(237, 247, 242, 0.94) 100%);
            box-shadow: 0 20px 38px rgba(20, 33, 61, 0.08);
            transform-style: preserve-3d;
            animation: strategyRise 620ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
            animation-delay: var(--strategy-delay);
            transition: transform 180ms ease, box-shadow 180ms ease;
        }

        .strategy-signal-card:hover {
            transform: translateY(-6px) rotateX(4deg) rotateY(-2deg);
            box-shadow: 0 28px 42px rgba(20, 33, 61, 0.12);
        }

        .strategy-signal-card::after {
            content: "";
            position: absolute;
            inset: auto -28px -45px auto;
            width: 140px;
            height: 140px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(20, 33, 61, 0.10) 0%, transparent 72%);
        }

        .strategy-card-top {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: baseline;
            gap: 0.7rem;
        }

        .strategy-allocation {
            font-family: var(--display);
            font-size: clamp(2rem, 2.5vw, 2.65rem);
            line-height: 1;
            color: #0f766e;
        }

        .strategy-card-kicker {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #678172;
        }

        .strategy-card-title {
            position: relative;
            z-index: 1;
            margin: 0.65rem 0 0;
            font-family: var(--display);
            font-size: 1.35rem;
            line-height: 1.15;
            color: var(--ink);
        }

        .strategy-card-copy {
            position: relative;
            z-index: 1;
            margin: 0.55rem 0 0;
            color: #4f5f6f;
            line-height: 1.62;
        }

        .strategy-meter {
            position: relative;
            z-index: 1;
            margin-top: 0.9rem;
            height: 0.8rem;
            border-radius: 999px;
            background: rgba(20, 33, 61, 0.08);
            overflow: hidden;
        }

        .strategy-fill {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #15897d 0%, #2bc2a8 100%);
            box-shadow: 0 10px 16px rgba(15, 118, 110, 0.22);
            transform-origin: left center;
            animation: strategyFill 1.2s ease both;
            animation-delay: calc(var(--strategy-delay) + 120ms);
        }

        .strategy-card-foot {
            position: relative;
            z-index: 1;
            margin-top: 0.8rem;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #5e7067;
        }

        .assistant-answer-shell {
            position: relative;
            overflow: hidden;
            margin-top: 1rem;
            margin-bottom: 0.9rem;
            padding: 1.05rem 1.1rem 1.1rem;
            border-radius: 26px;
            border: 1px solid rgba(20, 33, 61, 0.08);
            box-shadow: 0 18px 34px rgba(20, 33, 61, 0.08);
        }

        .assistant-answer-shell.advisor {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(241, 250, 246, 0.92) 100%);
        }

        .assistant-answer-shell.auditor {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(255, 245, 235, 0.92) 100%);
        }

        .assistant-answer-shell.researcher {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(239, 244, 252, 0.92) 100%);
        }

        .assistant-answer-topline {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: center;
            margin-bottom: 0.7rem;
        }

        .assistant-mode-chip,
        .assistant-intent-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            border: 1px solid rgba(20, 33, 61, 0.08);
            background: rgba(255, 255, 255, 0.72);
            color: var(--ink);
        }

        .assistant-answer-title {
            margin: 0;
            font-family: var(--display);
            font-size: clamp(1.5rem, 2vw, 1.95rem);
            line-height: 1.08;
            color: var(--ink);
        }

        .assistant-answer-copy {
            margin: 0.55rem 0 0;
            color: #566475;
            line-height: 1.65;
        }

        .note-grid {
            display: grid;
            gap: 0.9rem;
            margin-top: 0.85rem;
            margin-bottom: 0.3rem;
        }

        .note-card {
            position: relative;
            min-height: 128px;
            padding: 1rem 1rem 1rem 4rem;
            border-radius: 22px;
            border: 1px solid rgba(20, 33, 61, 0.09);
            background: rgba(255, 255, 255, 0.76);
            box-shadow: 0 16px 34px rgba(20, 33, 61, 0.05);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .note-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 44px rgba(20, 33, 61, 0.12);
        }

        .note-badge {
            position: absolute;
            left: 1rem;
            top: 1rem;
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 800;
            color: #ffffff;
            box-shadow: 0 10px 18px rgba(20, 33, 61, 0.14);
        }

        .note-text {
            color: var(--ink);
            line-height: 1.62;
        }

        .note-card.positive .note-badge {
            background: linear-gradient(135deg, #0f766e 0%, #19a389 100%);
        }

        .note-card.warning .note-badge {
            background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        }

        .note-card.tip .note-badge {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        }

        .advice-wrap {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(340px, 0.9fr);
            gap: 1.15rem;
            margin-top: 1.2rem;
        }

        .advice-panel {
            position: relative;
            overflow: hidden;
            min-height: 100%;
            padding: 1.2rem 1.2rem 1.25rem;
            border-radius: 28px;
            border: 1px solid rgba(20, 33, 61, 0.08);
            box-shadow: 0 18px 38px rgba(20, 33, 61, 0.08);
        }

        .advice-panel::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            filter: blur(4px);
            opacity: 0.5;
            transform: translate(-30%, -35%);
        }

        .advice-panel.positive {
            background: linear-gradient(180deg, rgba(240, 250, 247, 0.96) 0%, rgba(255, 255, 255, 0.96) 100%);
        }

        .advice-panel.positive::before {
            background: radial-gradient(circle, rgba(15, 118, 110, 0.22) 0%, transparent 72%);
        }

        .advice-panel.warning {
            background: linear-gradient(180deg, rgba(255, 248, 238, 0.96) 0%, rgba(255, 255, 255, 0.96) 100%);
        }

        .advice-panel.warning::before {
            background: radial-gradient(circle, rgba(217, 119, 6, 0.18) 0%, transparent 72%);
        }

        .advice-head {
            position: relative;
            z-index: 1;
            margin-bottom: 1rem;
        }

        .advice-kicker {
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #566170;
        }

        .advice-title {
            margin: 0.2rem 0 0;
            font-family: var(--display);
            font-size: 1.55rem;
            line-height: 1.05;
            letter-spacing: -0.03em;
            color: var(--ink);
        }

        .advice-copy {
            margin: 0.55rem 0 0;
            max-width: 62ch;
            color: #5a6777;
            line-height: 1.65;
            font-size: 0.97rem;
        }

        .advice-grid {
            position: relative;
            z-index: 1;
            display: grid;
            gap: 0.9rem;
        }

        .advice-card {
            position: relative;
            min-height: 150px;
            padding: 1rem 1rem 1rem 4.2rem;
            border-radius: 24px;
            border: 1px solid rgba(20, 33, 61, 0.08);
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 16px 32px rgba(20, 33, 61, 0.06);
            transition: transform 170ms ease, box-shadow 170ms ease, border-color 170ms ease;
        }

        .advice-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 24px 40px rgba(20, 33, 61, 0.11);
            border-color: rgba(20, 33, 61, 0.14);
        }

        .advice-index {
            position: absolute;
            left: 1rem;
            top: 1rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.35rem;
            height: 2.35rem;
            border-radius: 14px;
            font-size: 0.82rem;
            font-weight: 800;
            color: #ffffff;
            box-shadow: 0 12px 22px rgba(20, 33, 61, 0.12);
        }

        .advice-card.positive .advice-index {
            background: linear-gradient(135deg, #15897d 0%, #1fb39a 100%);
        }

        .advice-card.warning .advice-index {
            background: linear-gradient(135deg, #dd8a1b 0%, #ffb13c 100%);
        }

        .advice-text {
            color: #324152;
            font-size: 1.02rem;
            line-height: 1.78;
            font-weight: 600;
            letter-spacing: -0.01em;
        }

        .advice-card.warning .advice-text {
            color: #4d3a26;
        }

        @media (max-width: 1200px) {
            .advice-wrap {
                grid-template-columns: 1fr;
            }
        }

        .assistant-stage {
            padding: 1rem 1.05rem;
            border-radius: 24px;
            border: 1px solid rgba(20, 33, 61, 0.08);
            background: rgba(255, 255, 255, 0.62);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
        }

        @media (min-width: 1400px) {
            .hero-shell {
                padding: 2.15rem 2.15rem 2.25rem 2.15rem;
            }

            .panel-card,
            .support-card {
                padding: 1.2rem 1.25rem;
            }
        }

        @keyframes heroFloat {
            0%, 100% { transform: translate3d(0, 0, 0); }
            50% { transform: translate3d(0, 18px, 0); }
        }

        @keyframes heroPulse {
            0%, 100% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.08); opacity: 1; }
        }

        @keyframes flashScroll {
            from { transform: translateX(0); }
            to { transform: translateX(-100%); }
        }

        @keyframes strategyRise {
            from {
                opacity: 0;
                transform: translateY(18px) rotateX(8deg);
            }
            to {
                opacity: 1;
                transform: translateY(0) rotateX(0deg);
            }
        }

        @keyframes strategySweep {
            0%, 100% { transform: translateX(-120%); opacity: 0; }
            16% { opacity: 0.42; }
            52% { transform: translateX(42%); opacity: 0.18; }
            86% { opacity: 0; }
        }

        @keyframes strategyFloat {
            0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
            50% { transform: translate3d(0, 18px, 0) scale(1.05); }
        }

        @keyframes strategyFill {
            from { transform: scaleX(0); }
            to { transform: scaleX(1); }
        }

        @keyframes comparisonSweep {
            0%, 100% { transform: translateX(-120%); opacity: 0; }
            15% { opacity: 0.45; }
            50% { transform: translateX(35%); opacity: 0.18; }
            85% { opacity: 0; }
        }

        @keyframes comparisonFloat {
            0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
            50% { transform: translate3d(0, 16px, 0) scale(1.05); }
        }

        @keyframes comparisonRise {
            from {
                opacity: 0;
                transform: translateY(18px) rotateX(8deg);
            }
            to {
                opacity: 1;
                transform: translateY(0) rotateX(0deg);
            }
        }

        .mini-label {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <div class="eyebrow">Investor Guidance Hub</div>
            <h1 class="hero-title">Mutual Fund Recommendation Assistant</h1>
            <p class="hero-copy">
                Explore investor profiles, compare shortlisted funds, ask sharper follow-up questions,
                and navigate recommendations through a clearer, more engaging decision experience.
            </p>
            <div class="hero-chip-row">
                <span class="hero-chip">Profile-based fund matching</span>
                <span class="hero-chip">Rotating investor insights</span>
                <span class="hero-chip">Interactive prompt actions</span>
                <span class="hero-chip">Clearer fund explanations</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_waiting_panel(kicker: str, title: str, copy: str, chips: list[str]) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-kicker">{escape(kicker)}</div>
            <h3 class="panel-title">{escape(title)}</h3>
            <p class="panel-copy">{escape(copy)}</p>
            <div class="hero-chip-row">{theme_html(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_banner(kicker: str, title: str, copy: str, tone: str = "default") -> None:
    st.markdown(
        f"""
        <section class="section-banner {escape(tone)}">
            <div class="section-banner-kicker">{escape(kicker)}</div>
            <h3 class="section-banner-title">{escape(title)}</h3>
            <p class="section-banner-copy">{escape(copy)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_strategy_motion_stage(allocation_plan) -> None:
    cards_html = "".join(
        (
            f"<article class='strategy-signal-card' style='--strategy-delay:{index * 110}ms;'>"
            f"<div class='strategy-card-top'>"
            f"<div class='strategy-allocation'>{slice_item.allocation_pct}%</div>"
            f"<div class='strategy-card-kicker'>Role Strength</div>"
            f"</div>"
            f"<h4 class='strategy-card-title'>{escape(slice_item.label)}</h4>"
            f"<p class='strategy-card-copy'>{escape(clean_frontend_text(slice_item.rationale))}</p>"
            f"<div class='strategy-meter'><span class='strategy-fill' style='width:{slice_item.allocation_pct}%;'></span></div>"
            f"<div class='strategy-card-foot'>Signal {index:02d}</div>"
            "</article>"
        )
        for index, slice_item in enumerate(allocation_plan, start=1)
    )
    st.markdown(
        (
            "<section class='strategy-stage'>"
            "<div class='strategy-stage-orbit strategy-stage-orbit-a'></div>"
            "<div class='strategy-stage-orbit strategy-stage-orbit-b'></div>"
            f"<div class='strategy-grid'>{cards_html}</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_assistant_mode_banner(rag_answer) -> None:
    mode_slug = rag_answer.mode.lower()
    mode_descriptions = {
        MODE_ADVISOR: "Recommendation-first guidance with the clearest next move.",
        MODE_AUDITOR: "Stress-tests the idea, surfaces blind spots, and checks downside first.",
        MODE_RESEARCHER: "Evidence-led reading that leans on support signals before conclusions.",
    }
    st.markdown(
        f"""
        <section class="assistant-answer-shell {mode_slug}">
            <div class="assistant-answer-topline">
                <span class="assistant-mode-chip">{escape(rag_answer.mode)} Lens</span>
                <span class="assistant-intent-chip">{escape(rag_answer.intent.replace("_", " ").title())}</span>
            </div>
            <h3 class="assistant-answer-title">{escape(rag_answer.summary)}</h3>
            <p class="assistant-answer-copy">{escape(mode_descriptions.get(rag_answer.mode, ""))}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def assistant_note_labels(mode: str) -> tuple[str, str]:
    if mode == MODE_AUDITOR:
        return "Risk Flags", "What to Audit"
    if mode == MODE_RESEARCHER:
        return "Evidence Signals", "Research Gaps"
    return "Recommended Takeaways", "What to Double-Check"


def render_spotlight_panel(
    profile: InvestorProfile,
    recommendations: list,
    persona_text: str,
) -> None:
    if recommendations:
        top_pick = recommendations[0]
        blend = ", ".join(item.sub_category for item in recommendations[:3])
        title = top_pick.scheme_name
        copy = (
            f"{clean_frontend_text(persona_text)} The current shortlist is leaning toward {blend.lower()} "
            f"with {top_pick.scheme_name} as the strongest fit anchor."
        )
    else:
        title = "Shortlist coming into focus"
        copy = (
            "Choose your preferences on the left and the planner will build a more relevant shortlist, "
            "allocation view, and guidance notes."
        )

    chips = [
        profile.goal,
        f"{profile.risk_appetite} risk comfort",
        f"{profile.horizon_years}-year horizon",
        profile.investment_mode,
    ]
    if profile.needs_tax_saving:
        chips.append("Tax-aware")

    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-kicker">Current Spotlight</div>
            <h3 class="panel-title">{escape(title)}</h3>
            <p class="panel-copy">{escape(copy)}</p>
            <div class="hero-chip-row">{theme_html(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if recommendations:
        render_metric_grid(
            [
                ("Best Match", recommendations[0].fit_label),
                ("Lowest Entry", format_rupees(min(item.minimum_investment for item in recommendations))),
                ("Fund Styles", str(len({item.sub_category for item in recommendations}))),
            ],
            columns=3,
            compact=True,
        )


def render_recommendation_card(index: int, recommendation) -> None:
    st.markdown(
        f"""
        <div class="recommend-heading">Recommended Pick {index:02d}</div>
        <h3 class="recommend-title">{escape(recommendation.scheme_name)}</h3>
        <div class="recommend-meta">
            {escape(recommendation.amc_name)} | {escape(recommendation.category)} | {escape(recommendation.sub_category)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_metric_grid(
        [
            ("Fit", recommendation.fit_label),
            ("Match Score", str(recommendation.score)),
            ("Risk Band", risk_bucket(recommendation.risk_level)),
            ("Rating", str(recommendation.rating)),
            ("Expense Ratio", f"{recommendation.expense_ratio:.2f}%"),
        ],
        columns=5,
    )

    render_metric_grid(
        [
            ("3Y Return", format_percent(recommendation.returns_3yr)),
            ("5Y Return", format_percent(recommendation.returns_5yr)),
            ("Fund Age", f"{recommendation.fund_age_yr:.0f} yrs"),
            ("AUM", format_crores(recommendation.effective_aum_cr)),
            ("Minimum", format_rupees(recommendation.minimum_investment)),
        ],
        columns=5,
    )

    render_metric_grid(
        [
            ("Recent NAV", format_decimal(recommendation.nav)),
            ("Latest Update", format_date_label(recommendation.latest_nav_date)),
        ],
        columns=2,
    )

    advice_html = (
        "<div class='advice-wrap'>"
        + render_advice_panel(
            kicker="Investment Case",
            title="Why this made the shortlist",
            subtitle="These signals are pulled from the fund's category fit, risk alignment, time horizon match, and scheme-level metrics.",
            items=recommendation.why_it_fits,
            tone="positive",
            columns=2,
        )
        + render_advice_panel(
            kicker="Review Before You Invest",
            title="Important watch-outs",
            subtitle="This is the practical caution layer so the recommendation still feels responsible, not just optimistic.",
            items=[recommendation.caution],
            tone="warning",
            columns=1,
        )
        + "</div>"
    )
    st.markdown(advice_html, unsafe_allow_html=True)


def main() -> None:
    schemes = load_fund_schemes()
    export_canonical_fund_dataset(schemes)
    document_chunks = load_document_chunks()
    vector_index = load_or_build_chunk_vector_index(document_chunks)
    ensure_app_state()
    ensure_profile_state()

    apply_theme()
    if not st.session_state["risk_notice_acknowledged"]:
        show_risk_notice_dialog()
    render_hero()
    render_flash_quotes()

    if schemes:
        st.success(
            f"{len(schemes)} schemes are ready for profile-based recommendations and grounded assistant guidance."
        )
    else:
        st.error("No scheme data is available right now, so recommendations cannot be generated yet.")

    left_col, right_col = st.columns([0.86, 1.14], gap="large")

    with left_col:
        st.subheader("Shape Your Profile")
        render_profile_presets()
        with st.container(border=True):
            age_text = st.text_input(
                "Age",
                key="profile_age_input",
                placeholder="Enter your age",
            )
            st.caption(f"Example: {PROFILE_EXAMPLES['age']}")

            goal = st.selectbox(
                "Primary Goal",
                GOAL_OPTIONS,
                index=None,
                key="profile_goal",
                placeholder="Select your main objective",
            )
            st.caption(f"Example: {PROFILE_EXAMPLES['goal']}")

            risk_appetite = st.selectbox(
                "Risk Appetite",
                RISK_APPETITE_OPTIONS,
                index=None,
                key="profile_risk_appetite",
                placeholder="Choose your comfort range",
            )
            st.caption(f"Example: {PROFILE_EXAMPLES['risk_appetite']}")

            horizon_text = st.text_input(
                "Investment Horizon (Years)",
                key="profile_horizon_input",
                placeholder="Enter the number of years",
            )
            st.caption(f"Example: {PROFILE_EXAMPLES['horizon_years']}")

            investment_mode = st.selectbox(
                "Investment Mode",
                INVESTMENT_MODE_OPTIONS,
                index=None,
                key="profile_investment_mode",
                placeholder="Choose how you plan to invest",
            )
            st.caption(f"Example: {PROFILE_EXAMPLES['investment_mode']}")

            amount_text = st.text_input(
                "Planned Investment Amount (Rs.)",
                key="profile_amount_input",
                placeholder="Enter your planned amount",
            )
            st.caption(f"Example: {PROFILE_EXAMPLES['amount']}")

            needs_tax_saving_choice = st.selectbox(
                "Tax Saving is Important",
                YES_NO_OPTIONS,
                index=None,
                key="profile_needs_tax_saving_choice",
                placeholder="Choose Yes or No",
            )
            st.caption(f"Example: {PROFILE_EXAMPLES['needs_tax_saving']}")

            profile, profile_missing_fields, profile_issues = build_profile_from_inputs(
                age_text=age_text,
                goal=goal,
                risk_appetite=risk_appetite,
                horizon_text=horizon_text,
                investment_mode=investment_mode,
                amount_text=amount_text,
                needs_tax_saving_choice=needs_tax_saving_choice,
            )
            profile_has_any_input = any(
                [
                    age_text.strip(),
                    goal is not None,
                    risk_appetite is not None,
                    horizon_text.strip(),
                    investment_mode is not None,
                    amount_text.strip(),
                    needs_tax_saving_choice is not None,
                ]
            )

            if profile_issues:
                for issue in profile_issues:
                    st.error(issue)
            elif profile_missing_fields and profile_has_any_input:
                st.warning(
                    "Complete the remaining fields to unlock recommendations: "
                    + ", ".join(profile_missing_fields)
                )

    recommendations = recommend_schemes(profile, schemes=schemes, top_n=5) if profile is not None else []
    persona_text = investor_persona(profile) if profile is not None else ""
    allocation_plan = strategy_blueprint(profile) if profile is not None else []
    checklist = action_checklist(profile, recommendations) if profile is not None else []

    with right_col:
        st.subheader("Personalized Direction")
        if profile is not None:
            render_spotlight_panel(profile, recommendations, persona_text)
        else:
            render_waiting_panel(
                kicker="Current Spotlight",
                title="Your shortlist will appear here once the profile is complete.",
                copy=(
                    "This area stays in preview mode until your age, goal, risk comfort, horizon, amount, "
                    "investment mode, and tax preference are entered."
                ),
                chips=[
                    "Age",
                    "Goal",
                    "Risk comfort",
                    "Investment horizon",
                    "Amount",
                    "Tax preference",
                ],
            )
        st.markdown(
            """
            <div class="hint-strip">
                Use the shortlist as a decision-support guide, then confirm the latest factsheet,
                suitability details, and fund disclosures before investing.
            </div>
            """,
            unsafe_allow_html=True,
        )

    shortlist_tab, strategy_tab, assistant_tab = st.tabs(["Shortlist", "Strategy", "Ask the Assistant"])

    with shortlist_tab:
        if profile is None:
            st.info("Complete the profile form on the left to generate your shortlist.")
        elif not recommendations:
            st.warning("Recommendations will appear here once scheme data is available.")
        else:
            render_metric_grid(
                [
                    ("Shortlist Size", str(len(recommendations))),
                    ("Top Pick", recommendations[0].fit_label),
                    ("Risk Mix", ", ".join(sorted({risk_bucket(item.risk_level) for item in recommendations}))),
                    ("Lowest Start", format_rupees(min(item.minimum_investment for item in recommendations))),
                ],
                columns=4,
                compact=True,
            )

        for index, recommendation in enumerate(recommendations, start=1):
            with st.container(border=True):
                render_recommendation_card(index, recommendation)

        render_section_banner(
            kicker="Comparison Board",
            title="Shortlist snapshot at a glance",
            copy="Scan the shortlisted schemes side by side without repeating labels inside every tile.",
        )
        if recommendations:
            comparison_rows = [
                {
                    "Scheme": item.scheme_name,
                    "Fund Type": item.sub_category,
                    "Fit": item.fit_label,
                    "Risk": risk_bucket(item.risk_level),
                    "3Y Return": format_percent(item.returns_3yr),
                    "5Y Return": format_percent(item.returns_5yr),
                    "Expense Ratio": f"{item.expense_ratio:.2f}%",
                    "Minimum": format_rupees(item.minimum_investment),
                }
                for item in recommendations
            ]
            render_markdown_table(comparison_rows)
        else:
            st.caption("The comparison board will populate after the profile is completed and recommendations are available.")

    with strategy_tab:
        if profile is None:
            st.info("Complete the profile to unlock your investment style and usage tips.")
        else:
            st.markdown(
                f"""
                <div class="panel-card">
                    <div class="panel-kicker">Profile Read</div>
                    <h3 class="panel-title">{escape(profile.goal)}</h3>
                    <p class="panel-copy">{escape(clean_frontend_text(persona_text))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_strategy_motion_stage(allocation_plan)

            render_section_banner(
                kicker="Action Layer",
                title="How to use this shortlist well",
                copy="These prompts keep the shortlist practical so it stays useful beyond the first recommendation screen.",
                tone="strategy",
            )
            render_note_grid(checklist, tone="tip", columns=2)

    with assistant_tab:
        render_section_banner(
            kicker="Assistant Console",
            title="Ask through the lens you want",
            copy="Each lens now frames the answer differently, so you can switch between practical guidance, risk review, and evidence-led interpretation.",
            tone="assistant",
        )

        preset_questions = sample_rag_questions(profile, recommendations) if profile is not None else ()
        st.session_state.setdefault("rag_question", "")
        st.session_state.setdefault("rag_selected_preset", None)
        st.session_state.setdefault("rag_last_preset", None)
        if st.session_state["rag_selected_preset"] not in preset_questions:
            st.session_state["rag_selected_preset"] = None

        assistant_setup_col, assistant_question_col = st.columns([0.88, 1.12], gap="large")
        with assistant_setup_col:
            with st.container(border=True):
                selected_preset = st.selectbox(
                    "Suggested Question",
                    preset_questions,
                    index=None,
                    key="rag_selected_preset",
                    placeholder=(
                        "Choose a guided question once the profile is ready"
                        if preset_questions
                        else "Complete the profile to unlock guided questions"
                    ),
                    disabled=not bool(preset_questions),
                )
                st.caption(
                    "Example: Compare the top two shortlisted funds for a 7-year moderate-risk investor."
                )
                if selected_preset != st.session_state.get("rag_last_preset"):
                    if selected_preset is not None:
                        st.session_state["rag_question"] = selected_preset
                    st.session_state["rag_last_preset"] = selected_preset

                assistant_mode = st.selectbox(
                    "Conversation Lens",
                    [MODE_ADVISOR, MODE_AUDITOR, MODE_RESEARCHER],
                    index=None,
                    key="assistant_mode",
                    placeholder="Choose how the assistant should respond",
                )
                st.caption("Example: Advisor")

        with assistant_question_col:
            with st.container(border=True):
                question = st.text_area(
                    "Ask a question",
                    key="rag_question",
                    height=150,
                    placeholder="Write your own question here",
                )
                st.caption(
                    "Example: Why is the top recommendation suitable for a 7-year moderate-risk investor?"
                )

        if profile is None:
            st.info("Complete the profile first, then choose a lens and ask your question.")
        elif assistant_mode is None:
            st.info("Choose a conversation lens to activate the assistant response.")
        elif not question.strip():
            st.info("Enter a question or pick one of the guided prompts to see an answer.")
        else:
            rag_answer = build_grounded_answer(
                question=question,
                profile=profile,
                recommendations=recommendations,
                vector_index=vector_index,
                document_chunks=document_chunks,
                assistant_mode=assistant_mode,
            )

            render_metric_grid(
                [
                    ("Focus", rag_answer.intent.replace("_", " ").title()),
                    ("Confidence", rag_answer.confidence_label),
                    ("Support Notes", str(len(rag_answer.citations))),
                    ("Fund Anchors", str(len(rag_answer.related_schemes))),
                ],
                columns=4,
                compact=True,
            )

            render_assistant_mode_banner(rag_answer)
            st.markdown(rag_answer.answer)
            st.caption(clean_frontend_text(rag_answer.guardrail_note))

            takeaway_title, watchout_title = assistant_note_labels(rag_answer.mode)
            takeaway_col, watchout_col = st.columns(2, gap="large")
            with takeaway_col:
                st.markdown(f"**{takeaway_title}**")
                render_note_grid(rag_answer.key_points, tone="positive", columns=1)
            with watchout_col:
                st.markdown(f"**{watchout_title}**")
                render_note_grid(rag_answer.watchouts, tone="warning", columns=1)

            if rag_answer.related_schemes:
                st.markdown("**Funds Referenced in This Answer**")
                for scheme in rag_answer.related_schemes:
                    with st.container(border=True):
                        st.markdown(f"**{scheme.scheme_name}**")
                        st.caption(f"{scheme.amc_name} | {scheme.category} | {scheme.sub_category}")
                        render_metric_grid(
                            [
                                ("Fit", scheme.fit_label),
                                ("Risk", risk_bucket(scheme.risk_level)),
                                ("Expense", f"{scheme.expense_ratio:.2f}%"),
                                ("Minimum", format_rupees(scheme.minimum_investment)),
                            ],
                            columns=4,
                            compact=True,
                        )

            st.markdown("**Trusted Corpus Sources**")
            reference_cols = st.columns(2, gap="large")
            for index, reference in enumerate(rag_answer.online_references):
                with reference_cols[index % 2]:
                    st.markdown(
                        f"""
                        <div class="support-card">
                            <p class="support-title">{escape(reference.scheme_name)}</p>
                            <p class="support-meta">{escape(reference.source_label)}</p>
                            <div><strong>Why it looks strong:</strong> {escape(clean_frontend_text(reference.support_statement))}</div>
                            <div style="margin-top:0.65rem;"><strong>Source note:</strong> {escape(clean_frontend_text(reference.fact))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"[Read source]({reference.source_url})")

if __name__ == "__main__":
    main()
