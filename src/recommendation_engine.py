from __future__ import annotations

from dataclasses import dataclass

from src.fund_data import FundScheme, load_fund_schemes


RISK_TARGETS = {"Low": 2, "Moderate": 4, "High": 6}
RISK_COMFORT_MAX = {"Low": 3, "Moderate": 5, "High": 6}
RISK_BUCKETS = {1: "Low", 2: "Low", 3: "Moderate", 4: "Moderate", 5: "High", 6: "High"}


@dataclass(frozen=True)
class InvestorProfile:
    age: int
    goal: str
    risk_appetite: str
    horizon_years: int
    investment_mode: str
    amount: int
    needs_tax_saving: bool


@dataclass(frozen=True)
class Recommendation:
    scheme_name: str
    amc_name: str
    category: str
    sub_category: str
    score: int
    fit_label: str
    risk_level: int
    rating: int
    expense_ratio: float
    fund_age_yr: float
    fund_size_cr: float
    effective_aum_cr: float
    returns_3yr: float | None
    returns_5yr: float | None
    minimum_investment: int
    nav: float | None
    latest_nav_date: str | None
    scheme_code: str | None
    master_match_type: str
    nav_source: str
    nav_confidence: str
    nav_is_synthetic: bool
    why_it_fits: list[str]
    caution: str


@dataclass(frozen=True)
class AllocationSlice:
    label: str
    allocation_pct: int
    rationale: str


def recommend_schemes(
    profile: InvestorProfile,
    schemes: tuple[FundScheme, ...] | None = None,
    top_n: int = 5,
) -> list[Recommendation]:
    catalog = schemes if schemes is not None else load_fund_schemes()
    ranked: list[Recommendation] = []

    for scheme in catalog:
        if scheme.category == "Other":
            continue

        score, _ = score_scheme(profile, scheme)
        minimum_investment = scheme.min_sip if profile.investment_mode == "SIP" else scheme.min_lumpsum
        ranked.append(
            Recommendation(
                scheme_name=scheme.scheme_name,
                amc_name=scheme.amc_name,
                category=scheme.category,
                sub_category=scheme.sub_category,
                score=score,
                fit_label=score_to_label(score),
                risk_level=scheme.risk_level,
                rating=scheme.rating,
                expense_ratio=scheme.expense_ratio,
                fund_age_yr=scheme.fund_age_yr,
                fund_size_cr=scheme.fund_size_cr,
                effective_aum_cr=effective_aum_cr(scheme),
                returns_3yr=scheme.returns_3yr,
                returns_5yr=scheme.returns_5yr,
                minimum_investment=minimum_investment,
                nav=scheme.nav,
                latest_nav_date=scheme.latest_nav_date,
                scheme_code=scheme.scheme_code,
                master_match_type=scheme.master_match_type,
                nav_source=scheme.nav_source,
                nav_confidence=scheme.nav_confidence,
                nav_is_synthetic=scheme.nav_is_synthetic,
                why_it_fits=build_why_it_fits(profile, scheme, minimum_investment),
                caution=build_caution(profile, scheme, minimum_investment),
            )
        )

    ranked.sort(
        key=lambda item: (
            item.score,
            item.rating,
            item.returns_5yr or -999.0,
            item.returns_3yr or -999.0,
        ),
        reverse=True,
    )
    return diversify_recommendations(ranked, top_n=top_n)


def score_scheme(profile: InvestorProfile, scheme: FundScheme) -> tuple[int, list[str]]:
    reasons: list[str] = []
    total = 0

    risk_score, risk_reason = risk_alignment(profile, scheme)
    total += risk_score
    reasons.append(risk_reason)

    goal_score, goal_reason = goal_alignment(profile, scheme)
    total += goal_score
    reasons.append(goal_reason)

    horizon_score, horizon_reason = horizon_alignment(profile, scheme)
    total += horizon_score
    reasons.append(horizon_reason)

    affordability_score, affordability_reason = affordability_alignment(profile, scheme)
    total += affordability_score
    reasons.append(affordability_reason)

    quality_score, quality_reason = quality_alignment(scheme)
    total += quality_score
    reasons.append(quality_reason)

    total += tax_preference_adjustment(profile, scheme)
    total += style_penalty(profile, scheme)

    return min(total, 100), reasons


def build_why_it_fits(profile: InvestorProfile, scheme: FundScheme, minimum_investment: int) -> list[str]:
    reasons: list[str] = []
    tags = combined_tags(scheme)
    risk_band = risk_bucket(scheme.risk_level)
    style = style_label(scheme)
    min_years, max_years = recommended_horizon_bounds(scheme)

    category_reason = category_fit_reason(profile, scheme, tags)
    if category_reason:
        reasons.append(category_reason)

    if scheme.risk_level <= RISK_COMFORT_MAX[profile.risk_appetite]:
        reasons.append(
            f"It sits in the {risk_band.lower()} risk band, which stays within your selected {profile.risk_appetite.lower()} comfort range."
        )
    elif abs(RISK_TARGETS[profile.risk_appetite] - scheme.risk_level) <= 1:
        reasons.append(
            f"It lands close to your target risk zone at the {risk_band.lower()} end of the spectrum."
        )

    if profile.horizon_years >= min_years and (
        max_years is None or profile.horizon_years <= max_years
    ):
        reasons.append(
            f"Your {profile.horizon_years}-year horizon fits the usual holding range for this {style} style."
        )
    elif max_years is None and profile.horizon_years >= min_years:
        reasons.append(
            f"Your {profile.horizon_years}-year horizon is long enough for this {style} style."
        )

    if profile.amount >= minimum_investment:
        reasons.append(
            f"Your planned {profile.investment_mode.lower()} amount of Rs. {profile.amount:,} is above this fund's minimum of Rs. {minimum_investment:,}."
        )

    quality_reason = quality_reason_text(scheme)
    if quality_reason:
        reasons.append(quality_reason)

    return dedupe_messages(reasons)[:4]


def category_fit_reason(profile: InvestorProfile, scheme: FundScheme, tags: str) -> str:
    style = style_label(scheme)

    if "elss" in tags and (profile.goal == "Tax Saving" or profile.needs_tax_saving):
        return "Its ELSS structure directly supports the tax-saving need while keeping the investment growth-oriented."

    if profile.goal == "Emergency Fund":
        if any(keyword in tags for keyword in ("liquid", "overnight")):
            return f"Its {style} mandate is built for liquidity and short holding periods, which fits an emergency bucket well."
        if any(keyword in tags for keyword in ("ultra short", "money market", "low duration", "short duration")):
            return f"Its {style} profile is better suited to near-term cash management than equity-heavy categories."
        return f"Its {style} style is more defensive than long-duration or pure equity options for emergency planning."

    if profile.goal == "Capital Preservation":
        return f"Its {style} category leans more toward capital stability than aggressive growth-led fund types."

    if profile.goal == "Income Stability":
        return f"Its {style} mandate is better aligned to steadier income-oriented positioning than a pure equity sleeve."

    if profile.goal == "Retirement":
        return f"Its {style} exposure is more aligned with long-horizon retirement compounding than a narrow tactical bet."

    if profile.goal == "Tax Saving":
        if "elss" in tags:
            return "It directly addresses the tax-saving goal instead of acting only as a general investment substitute."
        return f"It still offers investment exposure through {style}, but it is not a direct tax-saving solution."

    if profile.risk_appetite == "Low":
        return f"Its {style} style is more measured than aggressive mid-cap, small-cap, or thematic options."

    if profile.risk_appetite == "Moderate":
        return f"Its {style} category gives you growth exposure without pushing into the narrowest or most concentrated styles."

    return f"Its {style} mandate supports long-term growth better than a short-horizon parking allocation."


def quality_reason_text(scheme: FundScheme) -> str:
    aum = effective_aum_cr(scheme)
    if scheme.rating >= 4 and scheme.expense_ratio <= 0.6:
        return f"It combines a {scheme.rating}-star rating with a relatively lean expense ratio of {scheme.expense_ratio:.2f}%."
    if scheme.returns_5yr is not None and scheme.returns_5yr >= 10:
        return f"Its 5-year return of {scheme.returns_5yr:.1f}% is one of the stronger long-term signals in this shortlist."
    if scheme.returns_3yr is not None and scheme.returns_3yr >= 8:
        return f"Its 3-year return of {scheme.returns_3yr:.1f}% gives it a respectable recent performance profile."
    if scheme.expense_ratio <= 0.5:
        return f"Its expense ratio of {scheme.expense_ratio:.2f}% is relatively low for a shortlist candidate."
    if scheme.fund_age_yr >= 10 and aum >= 1000:
        return f"It has a longer operating history of about {scheme.fund_age_yr:.0f} years and a sizeable AUM base of roughly Rs. {aum:,.0f} Cr."
    if scheme.fund_age_yr >= 10:
        return f"It has been around for about {scheme.fund_age_yr:.0f} years, so it is not a new or untested fund."
    if aum >= 1000:
        return f"It manages roughly Rs. {aum:,.0f} Cr., which gives it a meaningful operating scale."
    if scheme.rating >= 4:
        return f"Its rating of {scheme.rating} supports its place in the shortlist."
    return ""


def dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for message in messages:
        cleaned = " ".join(message.split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def risk_alignment(profile: InvestorProfile, scheme: FundScheme) -> tuple[int, str]:
    target = RISK_TARGETS[profile.risk_appetite]
    difference = abs(target - scheme.risk_level)
    score = max(4, 32 - (difference * 8))

    if scheme.risk_level <= RISK_COMFORT_MAX[profile.risk_appetite]:
        reason = "Risk level is aligned with the selected comfort range."
    else:
        score -= 4
        reason = "Risk level is above the selected comfort range, so this needs extra care."

    return max(score, 0), reason


def goal_alignment(profile: InvestorProfile, scheme: FundScheme) -> tuple[int, str]:
    tags = combined_tags(scheme)

    if "elss" in tags and (profile.goal == "Tax Saving" or profile.needs_tax_saving):
        return 28, "ELSS directly supports the tax-saving use case."

    if profile.goal == "Emergency Fund":
        return score_from_keywords(
            tags,
            high=("liquid", "overnight", "ultra short", "money market"),
            medium=("low duration", "short duration", "arbitrage"),
            low=("banking and psu", "corporate bond"),
            default_reason="This scheme is less suitable for emergency parking needs.",
        )

    if profile.goal == "Capital Preservation":
        return score_from_keywords(
            tags,
            high=("banking and psu", "short duration", "low duration", "ultra short", "corporate bond"),
            medium=("money market", "arbitrage", "dynamic bond"),
            low=("balanced advantage", "equity savings"),
            default_reason="This scheme is not primarily built for capital preservation.",
        )

    if profile.goal == "Income Stability":
        return score_from_keywords(
            tags,
            high=("banking and psu", "corporate bond", "short duration", "dynamic bond", "conservative hybrid"),
            medium=("arbitrage", "equity savings", "balanced advantage"),
            low=("index",),
            default_reason="This scheme is not strongly aligned to stable-income style investing.",
        )

    if profile.goal == "Retirement":
        return score_from_keywords(
            tags,
            high=("index", "flexi cap", "large cap", "large and mid cap", "balanced advantage", "retirement"),
            medium=("multi cap", "aggressive hybrid", "mid cap"),
            low=("small cap", "sectoral", "thematic"),
            default_reason="This scheme is not an obvious retirement-first fit from its category.",
        )

    if profile.goal == "Tax Saving":
        return score_from_keywords(
            tags,
            high=("elss",),
            medium=(),
            low=(),
            default_reason="This scheme does not directly solve the tax-saving goal.",
        )

    if profile.risk_appetite == "Low":
        return score_from_keywords(
            tags,
            high=("balanced advantage", "equity savings"),
            medium=("index", "large cap"),
            low=("flexi cap", "mid cap", "small cap", "sectoral", "thematic"),
            default_reason="This scheme is more growth-oriented than a low-risk wealth plan usually needs.",
        )

    if profile.risk_appetite == "Moderate":
        return score_from_keywords(
            tags,
            high=("index", "large cap", "large and mid cap", "balanced advantage"),
            medium=("flexi cap", "multi cap", "aggressive hybrid", "focused", "mid cap"),
            low=("small cap", "sectoral", "thematic"),
            default_reason="This scheme is not a strong match for balanced wealth creation goals.",
        )

    return score_from_keywords(
        tags,
        high=("index", "flexi cap", "large cap", "large and mid cap", "mid cap", "multi cap", "value", "focused"),
        medium=("balanced advantage", "aggressive hybrid", "small cap"),
        low=("sectoral", "thematic"),
        default_reason="This scheme is not a strong match for broad wealth creation goals.",
    )


def score_from_keywords(
    tags: str,
    high: tuple[str, ...],
    medium: tuple[str, ...],
    low: tuple[str, ...],
    default_reason: str,
) -> tuple[int, str]:
    if any(keyword in tags for keyword in high):
        return 28, "The scheme category strongly matches the selected goal."
    if any(keyword in tags for keyword in medium):
        return 18, "The scheme category is a reasonable match for the selected goal."
    if any(keyword in tags for keyword in low):
        return 8, "The scheme can fit the goal, but it may be more specialized or volatile."
    return 2, default_reason


def horizon_alignment(profile: InvestorProfile, scheme: FundScheme) -> tuple[int, str]:
    min_years, max_years = recommended_horizon_bounds(scheme)

    if profile.horizon_years < min_years:
        return 0, f"This type usually needs at least {min_years} years to be held comfortably."

    if max_years is not None and profile.horizon_years > max_years:
        return 5, "The horizon is longer than the usual holding pattern for this type."

    return 20, "Investment horizon fits the typical holding window for this scheme type."


def affordability_alignment(profile: InvestorProfile, scheme: FundScheme) -> tuple[int, str]:
    minimum = scheme.min_sip if profile.investment_mode == "SIP" else scheme.min_lumpsum
    if profile.amount >= minimum:
        return 10, f"The planned {profile.investment_mode.lower()} amount meets the minimum ticket size."
    if profile.amount >= (minimum * 0.5):
        return 2, "The amount is close to the minimum required for this scheme."
    return -10, "The planned investment amount is below this scheme's minimum requirement."


def quality_alignment(scheme: FundScheme) -> tuple[int, str]:
    score = 0

    if scheme.rating >= 4:
        score += 4
    elif scheme.rating >= 3:
        score += 3
    elif scheme.rating >= 2:
        score += 1

    if scheme.expense_ratio <= 0.5:
        score += 3
    elif scheme.expense_ratio <= 1.0:
        score += 2
    elif scheme.expense_ratio <= 1.5:
        score += 1

    if scheme.fund_age_yr >= 7:
        score += 2
    elif scheme.fund_age_yr >= 4:
        score += 1

    aum = effective_aum_cr(scheme)
    if aum >= 500:
        score += 2
    elif aum >= 100:
        score += 1

    if (scheme.returns_5yr is not None and scheme.returns_5yr >= 9) or (
        scheme.returns_3yr is not None and scheme.returns_3yr >= 7
    ):
        score += 2

    if scheme.nav is not None and not scheme.nav_is_synthetic and scheme.latest_nav_date:
        score += 1

    capped_score = min(score, 11)
    return capped_score, "Scheme-level metrics such as rating, cost, age, AUM, and return history support the shortlist."


def build_caution(profile: InvestorProfile, scheme: FundScheme, minimum_investment: int) -> str:
    tags = combined_tags(scheme)
    min_years, _ = recommended_horizon_bounds(scheme)
    messages: list[str] = []
    risk_band = risk_bucket(scheme.risk_level)

    if scheme.risk_level > RISK_COMFORT_MAX[profile.risk_appetite]:
        messages.append(
            f"It sits in the {risk_band.lower()} risk band, which is above your selected {profile.risk_appetite.lower()} comfort range."
        )

    if profile.horizon_years < min_years:
        messages.append(
            f"This {scheme.sub_category.lower()} style usually needs at least {min_years} years, while your current horizon is {profile.horizon_years}."
        )

    if profile.needs_tax_saving and "elss" not in tags:
        messages.append("It does not directly meet the tax-saving preference.")

    if profile.amount < minimum_investment:
        messages.append(
            f"The minimum {profile.investment_mode.lower()} amount is Rs. {minimum_investment:,}, which is above your current plan of Rs. {profile.amount:,}."
        )

    if "sectoral" in tags or "thematic" in tags:
        messages.append("Sectoral and thematic strategies can be more concentrated than diversified funds.")

    if "small cap" in tags and profile.risk_appetite != "High":
        messages.append("Small-cap funds can be volatile and usually suit investors with stronger risk tolerance.")

    if scheme.category == "Solution Oriented" and profile.goal != "Retirement":
        messages.append("Solution-oriented funds can carry goal-specific constraints or holding expectations.")

    if len(messages) < 2:
        messages.append(default_scheme_caution(scheme))

    if scheme.master_match_type == "unmatched":
        messages.append(
            "Some supporting details are still limited, so confirm the latest official factsheet before investing."
        )

    if scheme.nav_is_synthetic:
        messages.append(
            "The displayed NAV should be cross-checked with the latest official value before investing."
        )

    if not messages:
        messages.append(default_scheme_caution(scheme))

    return " ".join(dedupe_messages(messages)[:3])


def default_scheme_caution(scheme: FundScheme) -> str:
    tags = combined_tags(scheme)
    style = style_label(scheme)

    if scheme.category == "Debt":
        if "dynamic bond" in tags:
            return "Dynamic bond funds can shift duration meaningfully when rate expectations change, so review interest-rate sensitivity before investing."
        if any(keyword in tags for keyword in ("corporate bond", "credit", "banking and psu", "short duration", "low duration")):
            return f"Even {style} funds should still be checked for credit quality and interest-rate sensitivity before investing."
        return f"{style.capitalize()} funds can still move with rates and liquidity conditions, so the latest factsheet is worth checking."

    if scheme.category == "Hybrid":
        return f"{style.capitalize()} funds can change their equity-debt mix over time, so review the current allocation before treating them as low-volatility substitutes."

    if scheme.category == "Solution Oriented":
        return "Solution-oriented funds can carry goal-specific rules or lock-in expectations, so confirm the product constraints before deciding."

    if any(keyword in tags for keyword in ("index", "large cap", "flexi cap", "multi cap", "mid cap")):
        return f"Even {style} funds can overlap with other equity holdings, so compare current portfolio exposure before adding another one."

    return "No major profile mismatch stands out, but the latest scheme factsheet should still be checked."


def style_label(scheme: FundScheme) -> str:
    style = " ".join(scheme.sub_category.lower().split())
    for suffix in (" mutual funds", " funds"):
        if style.endswith(suffix):
            style = style[: -len(suffix)].rstrip()
    return style or scheme.sub_category.lower()


def score_to_label(score: int) -> str:
    if score >= 80:
        return "Strong Fit"
    if score >= 62:
        return "Good Fit"
    return "Consider Carefully"


def profile_summary(profile: InvestorProfile) -> list[str]:
    return [
        f"Goal: {profile.goal}",
        f"Risk appetite: {profile.risk_appetite}",
        f"Investment horizon: {profile.horizon_years} years",
        f"Investment mode: {profile.investment_mode}",
        f"Amount: Rs. {profile.amount:,}",
        f"Tax saving needed: {'Yes' if profile.needs_tax_saving else 'No'}",
    ]


def investor_persona(profile: InvestorProfile) -> str:
    if profile.goal == "Emergency Fund":
        return "Capital safety comes first. The assistant will bias toward high-liquidity, lower-volatility parking options."
    if profile.goal == "Tax Saving":
        return "You are goal-driven and tax-aware. The shortlist will strongly favor ELSS and highlight lock-in implications."
    if profile.goal == "Retirement":
        return "This profile is long-horizon oriented. The engine leans toward diversified growth categories with manageable risk."
    if profile.risk_appetite == "Low":
        return "This is a stability-first profile. The assistant avoids concentrated or very volatile strategies."
    if profile.risk_appetite == "Moderate":
        return "This is a balanced growth profile. The shortlist aims for diversified compounding without extreme concentration."
    return "This is a growth-seeking profile. The assistant allows broader equity exposure but still checks horizon and diversification."


def strategy_blueprint(profile: InvestorProfile) -> list[AllocationSlice]:
    if profile.goal == "Emergency Fund":
        return [
            AllocationSlice("Liquid Funds", 60, "Keeps a large portion highly accessible."),
            AllocationSlice("Money Market / Ultra Short", 40, "Adds a little yield while staying relatively liquid."),
        ]

    if profile.goal == "Capital Preservation":
        return [
            AllocationSlice("Short Duration Debt", 45, "Focuses on stability over aggressive growth."),
            AllocationSlice("Banking & PSU / Corporate Bond", 35, "Adds predictable debt exposure."),
            AllocationSlice("Arbitrage / Balanced Advantage", 20, "Provides a measured growth buffer."),
        ]

    if profile.goal == "Income Stability":
        return [
            AllocationSlice("Conservative Hybrid", 40, "Blends debt stability with modest growth."),
            AllocationSlice("Banking & PSU / Corporate Bond", 35, "Anchors the portfolio with income-oriented debt."),
            AllocationSlice("Balanced Advantage / Equity Savings", 25, "Adds controlled equity participation."),
        ]

    if profile.goal == "Tax Saving":
        return [
            AllocationSlice("ELSS Core", 70, "Handles the tax-saving objective directly."),
            AllocationSlice("Stability Sleeve", 30, "Reduces the chance of relying only on one high-volatility bucket."),
        ]

    if profile.goal == "Retirement":
        return [
            AllocationSlice("Index / Large Cap", 40, "Builds a diversified long-term core."),
            AllocationSlice("Flexi Cap / Large & Mid Cap", 35, "Adds adaptive long-horizon growth exposure."),
            AllocationSlice("Balanced Advantage / Hybrid", 25, "Improves downside discipline across market cycles."),
        ]

    if profile.risk_appetite == "Low":
        return [
            AllocationSlice("Balanced Advantage", 45, "Keeps equity exposure moderated."),
            AllocationSlice("Large Cap / Index", 30, "Adds a diversified growth engine."),
            AllocationSlice("Short Debt / Arbitrage", 25, "Improves stability and liquidity."),
        ]

    if profile.risk_appetite == "Moderate":
        return [
            AllocationSlice("Large Cap / Index", 40, "Forms a diversified core."),
            AllocationSlice("Flexi Cap / Large & Mid Cap", 35, "Adds stronger compounding potential."),
            AllocationSlice("Balanced Advantage / Hybrid", 25, "Helps control volatility and entry timing."),
        ]

    return [
        AllocationSlice("Flexi / Multi Cap", 40, "Drives long-term growth with broad equity exposure."),
        AllocationSlice("Large Cap / Index", 30, "Keeps the core diversified."),
        AllocationSlice("Mid / Small Cap Satellite", 15, "Adds selective higher-growth upside."),
        AllocationSlice("Hybrid Buffer", 15, "Maintains some volatility control."),
    ]


def action_checklist(profile: InvestorProfile, recommendations: list[Recommendation]) -> list[str]:
    notes = [
        "Use the shortlist as a starting point, then review scheme factsheets before any decision.",
        "Prefer 2 to 3 diversified schemes rather than chasing too many overlapping funds.",
    ]

    if profile.investment_mode == "SIP":
        notes.append("A SIP route can smooth entry points over time, especially for equity-heavy recommendations.")
    else:
        notes.append("For lump-sum investing, consider staggered deployment if the shortlist is equity-heavy.")

    if profile.goal == "Tax Saving":
        notes.append("Check ELSS lock-in expectations and avoid mixing tax-saving and emergency-fund goals in one bucket.")

    if any(item.master_match_type == "unmatched" for item in recommendations):
        notes.append("Some shortlist entries still need master-data linking, so scheme documents remain important for validation.")

    return notes


def risk_bucket(risk_level: int) -> str:
    return RISK_BUCKETS.get(risk_level, "Unknown")


def recommended_horizon_bounds(scheme: FundScheme) -> tuple[int, int | None]:
    tags = combined_tags(scheme)

    if "overnight" in tags or "liquid" in tags:
        return 0, 1
    if "ultra short" in tags or "money market" in tags:
        return 1, 2
    if "low duration" in tags or "short duration" in tags:
        return 1, 3
    if "banking and psu" in tags or "corporate bond" in tags or "dynamic bond" in tags:
        return 2, 4
    if "arbitrage" in tags or "equity savings" in tags or "balanced advantage" in tags:
        return 3, 5
    if "aggressive hybrid" in tags or "conservative hybrid" in tags:
        return 3, 5
    if "elss" in tags:
        return 3, None
    if "small cap" in tags or "sectoral" in tags or "thematic" in tags or "mid cap" in tags:
        return 7, None
    if "index" in tags or "flexi cap" in tags or "large cap" in tags or "large and mid cap" in tags:
        return 5, None
    if scheme.category == "Debt":
        return 2, 4
    if scheme.category == "Hybrid":
        return 3, 5
    if scheme.category in {"Equity", "Solution Oriented"}:
        return 5, None
    return 3, None


def combined_tags(scheme: FundScheme) -> str:
    return f"{scheme.category} {scheme.sub_category} {scheme.scheme_name}".lower().replace("&", "and")


def tax_preference_adjustment(profile: InvestorProfile, scheme: FundScheme) -> int:
    tags = combined_tags(scheme)

    if profile.goal == "Tax Saving" or profile.needs_tax_saving:
        if "elss" in tags:
            return 10
        return -18

    return 0


def style_penalty(profile: InvestorProfile, scheme: FundScheme) -> int:
    tags = combined_tags(scheme)

    if ("sectoral" in tags or "thematic" in tags) and profile.risk_appetite != "High":
        return -12
    if "small cap" in tags and profile.risk_appetite == "Low":
        return -16
    if "small cap" in tags and profile.risk_appetite == "Moderate":
        return -8
    if "mid cap" in tags and profile.risk_appetite == "Low":
        return -8
    if scheme.category == "Solution Oriented" and profile.goal != "Retirement":
        return -12
    if "child" in tags:
        return -14

    return 0


def effective_aum_cr(scheme: FundScheme) -> float:
    if scheme.average_aum_cr is not None and scheme.average_aum_cr > 0:
        return scheme.average_aum_cr
    return scheme.fund_size_cr


def diversify_recommendations(recommendations: list[Recommendation], top_n: int) -> list[Recommendation]:
    selected: list[Recommendation] = []
    seen_subcategories: set[str] = set()

    for item in recommendations:
        if item.sub_category in seen_subcategories:
            continue
        selected.append(item)
        seen_subcategories.add(item.sub_category)
        if len(selected) == top_n:
            return selected

    for item in recommendations:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) == top_n:
            break

    return selected


def roadmap_items() -> list[str]:
    return [
        "Expand the knowledge corpus beyond project docs into scheme factsheets, KIMs, and policy PDFs.",
        "Plug the prompt pack into an external LLM provider when API access is available for richer natural-language synthesis.",
        "Introduce evaluation cases for suitability, retrieval quality, citation coverage, and hallucination control.",
        "Add portfolio upload and overlap detection so the assistant can audit diversification, not just recommend new funds.",
        "Connect historical NAV trends to performance consistency, drawdown behavior, and market-cycle storytelling.",
    ]
