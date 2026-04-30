from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.document_pipeline import DocumentChunk, search_document_chunks
from src.fund_data import PROCESSED_DATASET_PATH, normalize_scheme_name
from src.recommendation_engine import (
    InvestorProfile,
    Recommendation,
    investor_persona,
    risk_bucket,
    strategy_blueprint,
)
from src.vector_store import ChunkVectorIndex, hybrid_search_vector_index, search_vector_index


INTENT_COMPARE = "compare"
INTENT_RISK = "risk_review"
INTENT_TAX = "tax_and_lockin"
INTENT_DATA = "data_quality"
INTENT_STRATEGY = "strategy"
INTENT_FIT = "scheme_fit"
INTENT_GENERAL = "general_guidance"

MODE_ADVISOR = "Advisor"
MODE_AUDITOR = "Auditor"
MODE_RESEARCHER = "Researcher"


@dataclass(frozen=True)
class AnswerCitation:
    source_type: str
    source_label: str
    source_url: str
    trust_tier: str
    domain: str
    title: str
    path: str
    page_numbers: tuple[int, ...]
    snippet: str
    score: float | None
    note: str


@dataclass(frozen=True)
class PromptPack:
    system_prompt: str
    user_prompt: str
    context_block: str


@dataclass(frozen=True)
class OnlineReferenceCard:
    scheme_name: str
    headline: str
    source_label: str
    source_url: str
    fact: str
    support_statement: str


@dataclass(frozen=True)
class RagAnswer:
    mode: str
    intent: str
    retrieval_query: str
    summary: str
    answer: str
    key_points: tuple[str, ...]
    watchouts: tuple[str, ...]
    citations: tuple[AnswerCitation, ...]
    online_references: tuple[OnlineReferenceCard, ...]
    related_schemes: tuple[Recommendation, ...]
    follow_ups: tuple[str, ...]
    confidence_label: str
    confidence_reason: str
    guardrail_note: str
    prompt_pack: PromptPack


def sample_rag_questions(
    profile: InvestorProfile,
    recommendations: list[Recommendation],
) -> tuple[str, ...]:
    top_name = recommendations[0].scheme_name if recommendations else "the top shortlisted fund"
    second_name = recommendations[1].scheme_name if len(recommendations) > 1 else "the second shortlisted fund"
    return (
        f"Why is {top_name} a strong fit for my profile?",
        "Compare the top shortlisted schemes for risk, cost, and holding period.",
        "What are the biggest risks or blind spots in this shortlist?",
        "What should I verify before choosing one of these funds?",
        f"How should I allocate across fund types for a {profile.goal.lower()} goal?",
        f"Which is better for this profile: {top_name} or {second_name}?",
    )


def build_grounded_answer(
    question: str,
    profile: InvestorProfile,
    recommendations: list[Recommendation],
    vector_index: ChunkVectorIndex,
    document_chunks: tuple[DocumentChunk, ...],
    assistant_mode: str = MODE_ADVISOR,
) -> RagAnswer:
    cleaned_question = " ".join((question or "").split())
    if not cleaned_question:
        cleaned_question = "Which shortlisted funds best match this profile and why?"

    intent = classify_query_intent(cleaned_question)
    related_schemes = select_related_schemes(cleaned_question, recommendations, intent)
    retrieval_query = build_retrieval_query(cleaned_question, profile, related_schemes, intent)
    citations = build_citations(
        cleaned_question,
        profile,
        recommendations,
        related_schemes,
        vector_index,
        document_chunks,
        retrieval_query,
    )

    summary, answer, key_points, watchouts = synthesize_response(
        cleaned_question,
        profile,
        recommendations,
        related_schemes,
        citations,
        intent,
        assistant_mode,
    )
    confidence_label, confidence_reason = assess_confidence(related_schemes, citations)
    online_references = build_online_reference_cards(citations, profile, related_schemes, recommendations)
    follow_ups = suggest_follow_ups(intent, related_schemes, profile)
    guardrail_note = build_guardrail_note(recommendations, related_schemes)
    prompt_pack = build_prompt_pack(
        cleaned_question,
        profile,
        related_schemes,
        citations,
        assistant_mode,
        intent,
    )

    return RagAnswer(
        mode=assistant_mode,
        intent=intent,
        retrieval_query=retrieval_query,
        summary=summary,
        answer=answer,
        key_points=tuple(key_points),
        watchouts=tuple(watchouts),
        citations=tuple(citations),
        online_references=tuple(online_references),
        related_schemes=tuple(related_schemes),
        follow_ups=tuple(follow_ups),
        confidence_label=confidence_label,
        confidence_reason=confidence_reason,
        guardrail_note=guardrail_note,
        prompt_pack=prompt_pack,
    )


def classify_query_intent(question: str) -> str:
    text = question.lower()

    if any(token in text for token in ("compare", "versus", " vs ", "better", "difference")):
        return INTENT_COMPARE
    if any(token in text for token in ("nav", "aum", "master link", "synthetic", "confidence", "source", "data quality")):
        return INTENT_DATA
    if any(token in text for token in ("tax", "80c", "elss", "lock-in", "lock in")):
        return INTENT_TAX
    if any(token in text for token in ("risk", "risky", "safe", "volatility", "volatile", "downside")):
        return INTENT_RISK
    if any(token in text for token in ("allocate", "allocation", "split", "strategy", "portfolio", "sip")):
        return INTENT_STRATEGY
    if any(token in text for token in ("fit", "why", "suitable", "recommend", "shortlist")):
        return INTENT_FIT
    return INTENT_GENERAL


def select_related_schemes(
    question: str,
    recommendations: list[Recommendation],
    intent: str,
) -> list[Recommendation]:
    if not recommendations:
        return []

    scored_matches: list[tuple[int, Recommendation]] = []
    question_normalized = normalize_scheme_name(question)
    question_tokens = set(question_normalized.split())

    for recommendation in recommendations:
        scheme_normalized = normalize_scheme_name(recommendation.scheme_name)
        scheme_tokens = set(scheme_normalized.split())
        overlap = len(question_tokens.intersection(scheme_tokens))
        if scheme_normalized and scheme_normalized in question_normalized:
            overlap += 4
        if recommendation.sub_category.lower() in question.lower():
            overlap += 2
        if recommendation.amc_name.lower() in question.lower():
            overlap += 1
        if overlap > 0:
            scored_matches.append((overlap, recommendation))

    scored_matches.sort(key=lambda item: (item[0], item[1].score), reverse=True)
    if scored_matches:
        selected = [item[1] for item in scored_matches[:3]]
        if intent == INTENT_COMPARE and len(selected) == 1 and len(recommendations) > 1:
            selected = recommendations[: min(3, len(recommendations))]
        return selected

    if intent == INTENT_COMPARE:
        return recommendations[: min(3, len(recommendations))]
    if intent in {INTENT_FIT, INTENT_RISK, INTENT_TAX, INTENT_DATA}:
        return recommendations[:1]
    if intent == INTENT_STRATEGY:
        return recommendations[: min(2, len(recommendations))]
    return recommendations[: min(3, len(recommendations))]


def build_retrieval_query(
    question: str,
    profile: InvestorProfile,
    related_schemes: list[Recommendation],
    intent: str,
) -> str:
    terms = [question, profile.goal, profile.risk_appetite, f"{profile.horizon_years} year"]
    if related_schemes:
        anchor = related_schemes[0]
        terms.extend([anchor.category, anchor.sub_category])
        if intent != INTENT_COMPARE:
            terms.append(anchor.scheme_name)

    if intent == INTENT_DATA:
        terms.extend(["synthetic nav", "master link", "data preprocessing"])
    elif intent == INTENT_TAX:
        terms.extend(["tax saving", "lock-in", "ELSS"])
    elif intent == INTENT_RISK:
        terms.extend(["risk horizon suitability", "volatility"])
    elif intent == INTENT_STRATEGY:
        terms.extend(["allocation blueprint", "goal planning"])

    return " ".join(term for term in terms if term)


def build_citations(
    question: str,
    profile: InvestorProfile,
    recommendations: list[Recommendation],
    related_schemes: list[Recommendation],
    vector_index: ChunkVectorIndex,
    document_chunks: tuple[DocumentChunk, ...],
    retrieval_query: str,
) -> list[AnswerCitation]:
    citations: list[AnswerCitation] = []
    citations.extend(build_dataset_citations(related_schemes))

    seen_chunk_ids: set[str] = set()
    vector_hits = hybrid_search_vector_index(retrieval_query, vector_index, top_n=4)
    if len(vector_hits) < 2:
        semantic_hits = search_vector_index(retrieval_query, vector_index, top_n=4)
        for item in semantic_hits:
            if item.record.chunk_id in seen_chunk_ids:
                continue
            vector_hits.append(item)

    for item in vector_hits[:4]:
        if item.record.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.record.chunk_id)
        citations.append(
            AnswerCitation(
                source_type="document",
                source_label=item.record.source_label,
                source_url=item.record.source_url,
                trust_tier=item.record.trust_tier,
                domain=item.record.domain,
                title=item.record.title,
                path=item.record.path,
                page_numbers=item.record.page_numbers,
                snippet=truncate_text(item.record.text, 280),
                score=item.score,
                note=f"{item.mode.title()} retrieval over the managed knowledge corpus.",
            )
        )

    if len(citations) < 3:
        keyword_hits = search_document_chunks(question, chunks=document_chunks, top_n=3)
        for item in keyword_hits:
            chunk_key = f"{item.chunk.doc_id}-{item.chunk.page_numbers}"
            if chunk_key in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_key)
            citations.append(
                AnswerCitation(
                    source_type="document",
                    source_label=item.chunk.source_label,
                    source_url=item.chunk.source_url,
                    trust_tier=item.chunk.trust_tier,
                    domain=item.chunk.domain,
                    title=item.chunk.title,
                    path=item.chunk.path,
                    page_numbers=item.chunk.page_numbers,
                    snippet=truncate_text(item.chunk.text, 280),
                    score=item.score,
                    note="Keyword retrieval fallback.",
                )
            )

    if not related_schemes and recommendations:
        citations.extend(build_dataset_citations(recommendations[:1]))

    return citations[:6]


def build_dataset_citations(recommendations: list[Recommendation]) -> list[AnswerCitation]:
    citations: list[AnswerCitation] = []
    for item in recommendations[:3]:
        nav_note = f"NAV {item.nav:.2f}" if item.nav is not None else "NAV unavailable"
        snippet = (
            f"{item.scheme_name} | score {item.score} | {item.fit_label} | risk {risk_bucket(item.risk_level)} | "
            f"expense {item.expense_ratio:.2f}% | 3Y {format_percent_value(item.returns_3yr)} | "
            f"5Y {format_percent_value(item.returns_5yr)} | {nav_note}"
        )
        citations.append(
            AnswerCitation(
                source_type="dataset",
                source_label="Canonical Scheme Dataset",
                source_url="",
                trust_tier="internal",
                domain="local.project",
                title=item.scheme_name,
                path=str(PROCESSED_DATASET_PATH),
                page_numbers=tuple(),
                snippet=snippet,
                score=float(item.score),
                note=f"Structured scheme record from the canonical processed dataset. Link quality: {item.master_match_type}.",
            )
        )
    return citations


def build_online_reference_cards(
    citations: list[AnswerCitation],
    profile: InvestorProfile,
    related_schemes: list[Recommendation],
    recommendations: list[Recommendation],
) -> list[OnlineReferenceCard]:
    schemes = related_schemes or recommendations[: min(3, len(recommendations))]
    cards: list[OnlineReferenceCard] = []
    trusted_citations = [
        item
        for item in citations
        if item.source_type == "document" and item.source_url and item.trust_tier in {"regulatory", "official", "amc", "reference"}
    ]

    for index, citation in enumerate(trusted_citations[:3]):
        scheme_name = schemes[min(index, len(schemes) - 1)].scheme_name if schemes else "Shortlist Support"
        support_scheme = schemes[min(index, len(schemes) - 1)] if schemes else None
        cards.append(
            OnlineReferenceCard(
                scheme_name=scheme_name,
                headline=f"Trusted source support for {scheme_name}",
                source_label=f"{citation.source_label} ({citation.trust_tier})",
                source_url=citation.source_url,
                fact=citation.snippet,
                support_statement=build_online_support_statement(profile, support_scheme) if support_scheme else "This source reinforces the answer with higher-trust corpus evidence.",
            )
        )

    return cards


def build_online_support_statement(profile: InvestorProfile, scheme: Recommendation) -> str:
    reasons = list(scheme.why_it_fits[:2])
    points: list[str] = [
        f"It currently ranks as a {scheme.fit_label.lower()} for a {profile.goal.lower()} goal."
    ]

    if reasons:
        points.append(reasons[0])
    if len(reasons) > 1:
        points.append(reasons[1])

    if scheme.returns_5yr is not None and scheme.returns_5yr > 0:
        points.append(f"It also shows a 5-year return of {scheme.returns_5yr:.1f}%.")
    elif scheme.returns_3yr is not None and scheme.returns_3yr > 0:
        points.append(f"It also shows a 3-year return of {scheme.returns_3yr:.1f}%.")
    elif scheme.rating >= 4:
        points.append(f"It carries a rating of {scheme.rating}.")

    if scheme.expense_ratio > 0:
        points.append(f"Its expense ratio is {scheme.expense_ratio:.2f}%.")

    return " ".join(points)


def synthesize_response(
    question: str,
    profile: InvestorProfile,
    recommendations: list[Recommendation],
    related_schemes: list[Recommendation],
    citations: list[AnswerCitation],
    intent: str,
    assistant_mode: str,
) -> tuple[str, str, list[str], list[str]]:
    if intent == INTENT_COMPARE:
        return build_compare_response(profile, related_schemes or recommendations[:3], assistant_mode)
    if intent == INTENT_DATA:
        return build_data_response(profile, recommendations, related_schemes, assistant_mode)
    if intent == INTENT_TAX:
        return build_tax_response(profile, related_schemes or recommendations[:2], assistant_mode)
    if intent == INTENT_RISK:
        return build_risk_response(profile, related_schemes or recommendations[:2], assistant_mode)
    if intent == INTENT_STRATEGY:
        return build_strategy_response(profile, related_schemes or recommendations[:2], assistant_mode)
    if intent == INTENT_FIT:
        return build_fit_response(profile, related_schemes or recommendations[:1], assistant_mode)
    return build_general_response(question, profile, related_schemes or recommendations[:3], citations, assistant_mode)


def mode_summary(
    assistant_mode: str,
    advisor_text: str,
    auditor_text: str,
    researcher_text: str,
) -> str:
    if assistant_mode == MODE_AUDITOR:
        return auditor_text
    if assistant_mode == MODE_RESEARCHER:
        return researcher_text
    return advisor_text


def format_mode_answer(
    assistant_mode: str,
    sections_by_mode: dict[str, tuple[str, list[tuple[str, list[str]]]]],
) -> str:
    title, sections = sections_by_mode.get(assistant_mode, sections_by_mode[MODE_ADVISOR])
    lines = [f"### {title}", ""]
    for heading, bullets in sections:
        valid_bullets = [bullet for bullet in bullets if bullet]
        if not valid_bullets:
            continue
        lines.append(f"**{heading}**")
        lines.extend(f"- {bullet}" for bullet in valid_bullets)
        lines.append("")
    return "\n".join(lines).strip()


def build_fit_response(
    profile: InvestorProfile,
    related_schemes: list[Recommendation],
    assistant_mode: str,
) -> tuple[str, str, list[str], list[str]]:
    if not related_schemes:
        return build_empty_response(profile)

    scheme = related_schemes[0]
    summary = mode_summary(
        assistant_mode,
        advisor_text=f"{scheme.scheme_name} is the clearest shortlist leader for this profile right now.",
        auditor_text=f"{scheme.scheme_name} can fit this profile, but it should pass a risk check before being treated as the obvious choice.",
        researcher_text=f"The available evidence supports {scheme.scheme_name} as a leading candidate, but it still needs final verification before becoming investable.",
    )
    answer = format_mode_answer(
        assistant_mode,
        {
            MODE_ADVISOR: (
                "Advisor View",
                [
                    (
                        "Best-fit read",
                        [
                            f"For a {profile.risk_appetite.lower()}-risk investor with a {profile.horizon_years}-year horizon, {scheme.scheme_name} currently lands as a {scheme.fit_label.lower()} with a match score of {scheme.score}.",
                            f"It sits in the {risk_bucket(scheme.risk_level).lower()} risk band and lines up well with the role this profile is trying to fill.",
                        ],
                    ),
                    (
                        "Why it earns attention",
                        [
                            scheme.why_it_fits[0] if scheme.why_it_fits else "",
                            f"The minimum {profile.investment_mode.lower()} amount is Rs. {scheme.minimum_investment:,}, the expense ratio is {scheme.expense_ratio:.2f}%, and the usual holding pattern is {holding_period_hint(scheme)}.",
                        ],
                    ),
                    (
                        "Best next move",
                        [
                            f"Keep {scheme.scheme_name} as the lead candidate, then compare its latest factsheet with one backup option before investing.",
                            f"The currently displayed NAV is {format_nav_value(scheme.nav)}, so treat the shortlist as guidance first and final execution second.",
                        ],
                    ),
                ],
            ),
            MODE_AUDITOR: (
                "Auditor View",
                [
                    (
                        "Primary concern",
                        [
                            scheme.caution,
                            f"The fund may still fit the profile, but the first job is to test whether its category behavior truly stays comfortable for a {profile.horizon_years}-year holding period.",
                        ],
                    ),
                    (
                        "Pressure points",
                        [
                            f"A strong score of {scheme.score} does not remove the need to question category volatility, concentration, and how the fund should behave in weak markets.",
                            f"Cost is {scheme.expense_ratio:.2f}% and minimum entry is Rs. {scheme.minimum_investment:,}, so suitability still matters more than convenience.",
                        ],
                    ),
                    (
                        "Checks before acting",
                        [
                            "Read the newest factsheet, confirm the latest portfolio stance, and make sure the current riskometer still fits your comfort range.",
                            "If you would feel stressed during a drawdown, this should stay on the shortlist rather than becoming an automatic selection.",
                        ],
                    ),
                ],
            ),
            MODE_RESEARCHER: (
                "Researcher View",
                [
                    (
                        "Evidence-led interpretation",
                        [
                            f"{scheme.scheme_name} is being surfaced because the current scheme metrics put it near the top of the shortlist for this profile, with a score of {scheme.score} and a {scheme.fit_label.lower()} reading.",
                            scheme.why_it_fits[0] if scheme.why_it_fits else "",
                        ],
                    ),
                    (
                        "Signals supporting the case",
                        [
                            f"The current profile fit is supported by its {risk_bucket(scheme.risk_level).lower()} risk band, rating of {scheme.rating}, and expense ratio of {scheme.expense_ratio:.2f}%.",
                            f"Its expected holding pattern is {holding_period_hint(scheme)}, which helps explain why it sits ahead of less aligned candidates.",
                        ],
                    ),
                    (
                        "What remains open",
                        [
                            f"The displayed NAV reads {format_nav_value(scheme.nav)}, so the final research step is to verify the most recent AMC factsheet and fund page before treating the idea as investable.",
                            "The current evidence is strong enough for shortlisting, not strong enough to skip document-level confirmation.",
                        ],
                    ),
                ],
            ),
        },
    )

    key_points = {
        MODE_ADVISOR: [
            f"Lead candidate: {scheme.scheme_name} with score {scheme.score} and fit label {scheme.fit_label}.",
            f"Core support: {scheme.why_it_fits[0] if scheme.why_it_fits else 'Profile alignment is strong.'}",
            "Treat the shortlist as a decision filter, then confirm the latest factsheet before execution.",
        ],
        MODE_AUDITOR: [
            f"Main risk check: {scheme.caution}",
            f"Stress point: {risk_bucket(scheme.risk_level)} risk and a {profile.horizon_years}-year holding assumption need to match your real comfort.",
            "A high shortlist score should still be challenged before it becomes a real allocation.",
        ],
        MODE_RESEARCHER: [
            f"Evidence signal: score {scheme.score}, rating {scheme.rating}, expense ratio {scheme.expense_ratio:.2f}%.",
            f"Context signal: {scheme.why_it_fits[0] if scheme.why_it_fits else 'This scheme sits high in the shortlist because of its profile alignment.'}",
            "Research remains incomplete until the current AMC factsheet confirms the latest scheme details.",
        ],
    }.get(assistant_mode, [])

    watchouts = [scheme.caution]
    if scheme.nav_is_synthetic:
        watchouts.append("Cross-check the latest quoted NAV before investing so the most recent fund value is confirmed.")
    return summary, answer, key_points, watchouts


def build_compare_response(
    profile: InvestorProfile,
    related_schemes: list[Recommendation],
    assistant_mode: str,
) -> tuple[str, str, list[str], list[str]]:
    if not related_schemes:
        return build_empty_response(profile)

    schemes = related_schemes[:3]
    best = max(schemes, key=lambda item: item.score)
    safest = min(schemes, key=lambda item: item.risk_level)
    cheapest = min(schemes, key=lambda item: item.expense_ratio)
    strongest_return = max(schemes, key=lambda item: item.returns_5yr or item.returns_3yr or -999.0)

    summary = mode_summary(
        assistant_mode,
        advisor_text=f"{best.scheme_name} is the strongest overall match in this comparison view.",
        auditor_text="The comparison is closer than it first appears, so the wrong winner can be chosen for the wrong reason if the trade-offs are not checked.",
        researcher_text="The compared schemes separate into different roles, with the evidence supporting a profile-based winner rather than a universal winner.",
    )
    answer = format_mode_answer(
        assistant_mode,
        {
            MODE_ADVISOR: (
                "Advisor View",
                [
                    (
                        "Who currently leads",
                        [
                            f"{best.scheme_name} leads the comparison because it carries the strongest suitability score for this profile and sits in a category that matches the stated goal well.",
                            f"If stability matters most, {safest.scheme_name} is the calmer option. If cost discipline matters most, {cheapest.scheme_name} is the cheapest on expense ratio.",
                        ],
                    ),
                    (
                        "How the options separate",
                        [
                            f"{strongest_return.scheme_name} shows the strongest trailing return signal among the compared names, but that should be treated as context rather than a forecast.",
                            f"Across the shortlist, the holding style broadly suits {', '.join(holding_period_hint(item) for item in schemes[:2])} style horizons.",
                        ],
                    ),
                    (
                        "Practical move from here",
                        [
                            f"Keep {best.scheme_name} as the core candidate and select one backup name rather than spreading across too many overlapping funds.",
                            "The best final choice should come from fit first, then cost, then the latest official documents.",
                        ],
                    ),
                ],
            ),
            MODE_AUDITOR: (
                "Auditor View",
                [
                    (
                        "Where the decision can go wrong",
                        [
                            "The biggest error is choosing the most aggressive or best-performing recent option without asking whether its category stress would still feel tolerable for this profile.",
                            f"{best.scheme_name} may lead on score, but that does not automatically make it the safest behavioural match.",
                        ],
                    ),
                    (
                        "Trade-offs to stress-test",
                        [
                            f"{safest.scheme_name} offers the lower-risk profile, while {cheapest.scheme_name} offers the lower-cost profile, so the top score is not the only variable worth protecting.",
                            f"{strongest_return.scheme_name} has the best trailing return signal here, but return leadership can reverse faster than suitability leadership.",
                        ],
                    ),
                    (
                        "What to verify before choosing",
                        [
                            "Check overlap, downside tolerance, and whether the intended role of the fund matches the investor's real behaviour under stress.",
                            "If two options look close, prefer the one whose category you would be more comfortable holding through a weak cycle.",
                        ],
                    ),
                ],
            ),
            MODE_RESEARCHER: (
                "Researcher View",
                [
                    (
                        "What the comparison data supports",
                        [
                            f"The evidence currently points to {best.scheme_name} as the strongest profile match, while {safest.scheme_name} and {cheapest.scheme_name} become useful alternates depending on whether risk control or fee discipline takes priority.",
                            f"The trailing return signal is strongest for {strongest_return.scheme_name}, but that is only one data point inside the broader suitability picture.",
                        ],
                    ),
                    (
                        "Signals worth separating",
                        [
                            "This comparison works best when fit score, risk bucket, fee level, and holding-period behaviour are read as different signals rather than one blended number.",
                            "That distinction matters because one fund can win on returns while another wins on behavioural fit or category stability.",
                        ],
                    ),
                    (
                        "What remains unresolved",
                        [
                            "The evidence can identify a likely front-runner, but it still cannot replace the latest factsheet, current portfolio stance, and investor-specific behaviour check.",
                            "Treat the comparison as evidence-backed shortlisting rather than final selection.",
                        ],
                    ),
                ],
            ),
        },
    )

    key_points = {
        MODE_ADVISOR: [
            f"Leader: {best.scheme_name}. Conservative option: {safest.scheme_name}. Lowest-cost option: {cheapest.scheme_name}.",
            f"Return leader: {strongest_return.scheme_name}, but trailing returns should stay secondary to fit.",
            "Choose one lead candidate and one backup candidate instead of overloading similar funds.",
        ],
        MODE_AUDITOR: [
            f"Risk checkpoint: {safest.scheme_name} is the calmer comparison point.",
            f"Behavioural checkpoint: {best.scheme_name} still needs to pass the investor comfort test, not just the scoring test.",
            "A strong recent return signal should not overpower suitability and downside tolerance.",
        ],
        MODE_RESEARCHER: [
            f"Evidence split: {best.scheme_name} leads on fit, {cheapest.scheme_name} leads on cost, {strongest_return.scheme_name} leads on trailing return.",
            "The comparison is stronger when those signals are kept separate instead of collapsed into a single story.",
            "Document-level confirmation is still required before treating any winner as final.",
        ],
    }.get(assistant_mode, [])

    watchouts = [item.caution for item in schemes[:2]]
    return summary, answer, key_points, watchouts


def build_data_response(
    profile: InvestorProfile,
    recommendations: list[Recommendation],
    related_schemes: list[Recommendation],
    assistant_mode: str,
) -> tuple[str, str, list[str], list[str]]:
    scheme = related_schemes[0] if related_schemes else None

    if scheme is not None:
        summary = mode_summary(
            assistant_mode,
            advisor_text=f"{scheme.scheme_name} looks usable for shortlisting, but the last step still belongs to document verification.",
            auditor_text=f"{scheme.scheme_name} should be treated with a verification-first mindset before any final decision.",
            researcher_text=f"{scheme.scheme_name} is best read as a documented shortlist candidate rather than a fully confirmed selection.",
        )
        answer = format_mode_answer(
            assistant_mode,
            {
                MODE_ADVISOR: (
                    "Advisor View",
                    [
                        (
                            "What this means in practice",
                            [
                                "If some values look uncertain, treat them as shortlist guidance and then verify the latest official fund documents before acting.",
                                f"For {scheme.scheme_name}, the practical move is to confirm the newest NAV, factsheet, portfolio positioning, and minimum investment before execution.",
                            ],
                        ),
                        (
                            "How to use the shortlist correctly",
                            [
                                "A strong suitability match and a final investable decision are not the same thing.",
                                "This tool can point you toward the right scheme family, but the AMC factsheet should still close the loop.",
                            ],
                        ),
                    ],
                ),
                MODE_AUDITOR: (
                    "Auditor View",
                    [
                        (
                            "Verification risk",
                            [
                                "The main failure mode here is treating supportive scheme data like a live confirmation signal when it should only be the first pass.",
                                f"For {scheme.scheme_name}, the last-mile checks matter because a good fit score should not override missing or stale details.",
                            ],
                        ),
                        (
                            "What must be confirmed",
                            [
                                "Check the latest NAV, the most recent factsheet date, the current portfolio stance, and any changes to minimum investment or category positioning.",
                                "If the official documents materially differ from the displayed details, trust the official source and revisit the shortlist.",
                            ],
                        ),
                    ],
                ),
                MODE_RESEARCHER: (
                    "Researcher View",
                    [
                        (
                            "Evidence interpretation",
                            [
                                f"The available scheme details are strong enough to keep {scheme.scheme_name} in the decision set, but the evidence is still only shortlist-grade until it is confirmed against the AMC record.",
                                "This is a classic case where evidence quality matters as much as evidence direction.",
                            ],
                        ),
                        (
                            "Where confirmation adds value",
                            [
                                "The final research step is to compare the displayed fields against the live fund page, latest factsheet, and current disclosures.",
                                "That extra pass matters most when you are relying on NAV, holding period expectations, or operational thresholds like minimum investment.",
                            ],
                        ),
                    ],
                ),
            },
        )
    else:
        summary = mode_summary(
            assistant_mode,
            advisor_text="The shortlist becomes useful when it is followed by a quick verification step.",
            auditor_text="The shortlist should not be treated as self-sufficient when data confirmation is still pending.",
            researcher_text="The evidence is helpful for direction, but the final validation still belongs to official scheme sources.",
        )
        answer = format_mode_answer(
            assistant_mode,
            {
                MODE_ADVISOR: (
                    "Advisor View",
                    [
                        (
                            "Practical takeaway",
                            [
                                "When an investor sees estimated or incomplete details, the right move is not to panic but to verify the latest official scheme information before acting.",
                                "The shortlist can still help with fit, holding period, and category direction, but final confirmation should come from the AMC factsheet or official disclosures.",
                            ],
                        ),
                    ],
                ),
                MODE_AUDITOR: (
                    "Auditor View",
                    [
                        (
                            "Main caution",
                            [
                                "The real risk is false confidence: treating partial fields as if they were already validated by the official source.",
                                "That can distort decisions around NAV, thresholds, and whether two close candidates are actually still close after verification.",
                            ],
                        ),
                    ],
                ),
                MODE_RESEARCHER: (
                    "Researcher View",
                    [
                        (
                            "Evidence-led read",
                            [
                                "Even without a single anchor scheme, the evidence still points to the same conclusion: shortlist logic is useful for narrowing choices, not for replacing the primary document check.",
                                "Research quality improves sharply once the shortlist is reconciled with official live fund information.",
                            ],
                        ),
                    ],
                ),
            },
        )

    key_points = {
        MODE_ADVISOR: [
            "Treat the shortlist as decision support, not as the final verification layer.",
            "Confirm the latest NAV, factsheet, and disclosures before investing.",
            "If details feel uncertain, prefer the AMC site for the closing check.",
        ],
        MODE_AUDITOR: [
            "Do not confuse shortlist confidence with source confirmation.",
            "Operational details such as NAV, thresholds, and dates deserve a final official check.",
            "If a fund is only barely ahead, source verification can change the final ranking.",
        ],
        MODE_RESEARCHER: [
            "Shortlist evidence and source confirmation play different roles in the decision chain.",
            "The research becomes materially stronger once the shortlist is reconciled with the AMC factsheet.",
            "Verification is part of the evidence process, not a separate optional step.",
        ],
    }.get(assistant_mode, [])
    watchouts = [
        "Do not treat estimated or stale values as live market quotes.",
        "If a fund is close to your shortlist threshold, verify the latest official details before choosing it.",
    ]
    return summary, answer, key_points, watchouts


def build_tax_response(
    profile: InvestorProfile,
    related_schemes: list[Recommendation],
    assistant_mode: str,
) -> tuple[str, str, list[str], list[str]]:
    if not related_schemes:
        return build_empty_response(profile)

    scheme = related_schemes[0]
    elss_like = "elss" in scheme.sub_category.lower() or "elss" in scheme.scheme_name.lower()
    summary = mode_summary(
        assistant_mode,
        advisor_text="Tax-sensitive decisions work best when tax fit is checked after suitability, not before it.",
        auditor_text="Tax benefit can easily distract from liquidity and suitability risk if the fund role is not defined clearly.",
        researcher_text="The tax question should be read as a suitability-plus-lock-in question rather than a simple benefit screen.",
    )
    answer = format_mode_answer(
        assistant_mode,
        {
            MODE_ADVISOR: (
                "Advisor View",
                [
                    (
                        "Tax fit in context",
                        [
                            f"For this profile, tax-saving logic prefers ELSS-style solutions when the goal or preference is tax saving. In the current shortlist, {scheme.scheme_name} is the main reference point.",
                            "Tax alignment still has to sit inside the broader portfolio role, so risk appetite and holding period remain part of the decision.",
                        ],
                    ),
                    (
                        "What that means for this fund",
                        [
                            f"{scheme.scheme_name} appears aligned to the tax-saving use case, but the lock-in implication still needs to be checked before treating it like short-term or emergency money." if elss_like else f"{scheme.scheme_name} may fit the portfolio on other dimensions, but it does not solve the tax-saving need in the same direct way as an ELSS fund.",
                            "Choose the tax bucket only after deciding whether the money can stay committed for the required period.",
                        ],
                    ),
                ],
            ),
            MODE_AUDITOR: (
                "Auditor View",
                [
                    (
                        "Where investors misread the setup",
                        [
                            "The most common error is mixing tax-saving goals and liquidity goals inside the same product bucket.",
                            f"{scheme.scheme_name} should not be treated as suitable just because it looks helpful on tax logic.",
                        ],
                    ),
                    (
                        "What to test before acting",
                        [
                            "Check lock-in, liquidity needs, and whether the investor would still be comfortable holding through the required period.",
                            "If the money may be needed early, the tax label should not override access needs.",
                        ],
                    ),
                ],
            ),
            MODE_RESEARCHER: (
                "Researcher View",
                [
                    (
                        "Evidence-led tax read",
                        [
                            f"The evidence suggests that {scheme.scheme_name} should be interpreted first through product structure and only then through tax benefit.",
                            "Tax suitability is stronger when lock-in behaviour, goal alignment, and risk tolerance all point in the same direction.",
                        ],
                    ),
                    (
                        "What remains important",
                        [
                            f"{scheme.scheme_name} {'appears' if elss_like else 'does not appear'} to sit in the more direct tax-saving lane, but that alone does not settle whether it belongs in this portfolio.",
                            "The final judgment still depends on how long the money can stay committed and what role the fund is meant to play.",
                        ],
                    ),
                ],
            ),
        },
    )

    key_points = {
        MODE_ADVISOR: [
            "Tax fit should be checked alongside risk and holding period, not in isolation.",
            "ELSS may satisfy the tax preference, but lock-in implications still matter.",
            f"Current profile tax preference: {'Yes' if profile.needs_tax_saving else 'No'}.",
        ],
        MODE_AUDITOR: [
            "Do not let the tax benefit hide a liquidity mismatch.",
            "Lock-in matters more when the money may be needed earlier than planned.",
            "A tax-friendly fund can still be behaviourally unsuitable.",
        ],
        MODE_RESEARCHER: [
            "Tax classification and portfolio role should be read together.",
            "Lock-in and liquidity evidence matter just as much as the tax label.",
            "A tax-aligned scheme is not automatically the best overall fit.",
        ],
    }.get(assistant_mode, [])
    watchouts = [
        "Avoid using a locked-in product for money that may be needed for emergencies.",
        "Past returns do not make a tax product automatically suitable.",
    ]
    return summary, answer, key_points, watchouts


def build_risk_response(
    profile: InvestorProfile,
    related_schemes: list[Recommendation],
    assistant_mode: str,
) -> tuple[str, str, list[str], list[str]]:
    if not related_schemes:
        return build_empty_response(profile)

    scheme = related_schemes[0]
    summary = mode_summary(
        assistant_mode,
        advisor_text="Risk review is strongest when category behaviour and profile fit are checked together.",
        auditor_text="The right risk question here is not just how risky the fund is, but how badly a mismatch would feel if the market turns.",
        researcher_text="The risk picture should be read as a combination of category behaviour, profile fit, and caution evidence rather than one label alone.",
    )
    answer = format_mode_answer(
        assistant_mode,
        {
            MODE_ADVISOR: (
                "Advisor View",
                [
                    (
                        "Risk read",
                        [
                            f"The main risk anchor in the shortlist is {scheme.scheme_name}, which sits in the {risk_bucket(scheme.risk_level).lower()} risk band for a {profile.risk_appetite.lower()}-risk investor.",
                            f"Its caution note is: {scheme.caution}",
                        ],
                    ),
                    (
                        "How to interpret that",
                        [
                            "Risk is best treated as a fit question first: is the category volatility and holding pattern reasonable for the investor's goal and time horizon?",
                            "If the answer is yes, the risk may be acceptable. If not, the fund should stay on watch rather than move to execution.",
                        ],
                    ),
                ],
            ),
            MODE_AUDITOR: (
                "Auditor View",
                [
                    (
                        "Main threat",
                        [
                            f"The biggest concern is not the label alone, but whether {scheme.scheme_name} would become emotionally difficult to hold if markets weaken.",
                            scheme.caution,
                        ],
                    ),
                    (
                        "Stress check",
                        [
                            f"For a {profile.horizon_years}-year horizon, the question is whether the investor can tolerate this category's rough periods without abandoning the plan halfway through.",
                            "If the likely behaviour under stress is poor, the theoretical suitability case weakens fast.",
                        ],
                    ),
                ],
            ),
            MODE_RESEARCHER: (
                "Researcher View",
                [
                    (
                        "Evidence-led risk interpretation",
                        [
                            f"{scheme.scheme_name} provides the main risk anchor because its current bucket, category behaviour, and caution profile create the clearest read on how the shortlist behaves.",
                            "The evidence is stronger when the category label, caution note, and investor time horizon all tell the same story.",
                        ],
                    ),
                    (
                        "What the evidence does not prove",
                        [
                            "A category risk label does not guarantee how any one investor will experience drawdowns or volatility in practice.",
                            "The research helps explain likely behaviour, but the investor's real comfort still has to be judged outside the dataset.",
                        ],
                    ),
                ],
            ),
        },
    )

    key_points = {
        MODE_ADVISOR: [
            f"Anchor risk bucket: {risk_bucket(scheme.risk_level)}.",
            f"Profile horizon: {profile.horizon_years} years; investment mode: {profile.investment_mode}.",
            "Use risk review to check fit, not just to read a category label.",
        ],
        MODE_AUDITOR: [
            f"Main pressure point: {scheme.caution}",
            "The behavioural response to volatility matters as much as the category label.",
            "A mismatch under stress can undo an otherwise strong-looking shortlist score.",
        ],
        MODE_RESEARCHER: [
            "Risk interpretation is stronger when caution notes, category behaviour, and horizon all point in the same direction.",
            "Evidence explains likely behaviour, but not guaranteed investor behaviour.",
            "Category labels are useful signals, not complete conclusions.",
        ],
    }.get(assistant_mode, [])
    watchouts = [scheme.caution]
    return summary, answer, key_points, watchouts


def build_strategy_response(
    profile: InvestorProfile,
    related_schemes: list[Recommendation],
    assistant_mode: str,
) -> tuple[str, str, list[str], list[str]]:
    blueprint = strategy_blueprint(profile)
    leading_slice = blueprint[0]
    supporting_slice = blueprint[1] if len(blueprint) > 1 else blueprint[0]
    scheme_line = ""
    if related_schemes:
        scheme_line = f" The current shortlist supports that structure with names such as {', '.join(item.scheme_name for item in related_schemes[:2])}."

    summary = mode_summary(
        assistant_mode,
        advisor_text=f"The best strategy frame for this profile starts with {leading_slice.label.lower()}.",
        auditor_text="The strategy should stay diversified enough to avoid one-category overconfidence.",
        researcher_text="The strategy blueprint is strongest when it is read as a role-based structure rather than a list of winning funds.",
    )
    answer = format_mode_answer(
        assistant_mode,
        {
            MODE_ADVISOR: (
                "Advisor View",
                [
                    (
                        "Portfolio direction",
                        [
                            investor_persona(profile),
                            f"The blueprint currently leans first toward {leading_slice.label.lower()} at {leading_slice.allocation_pct}% and then toward {supporting_slice.label.lower()} at {supporting_slice.allocation_pct}%.",
                        ],
                    ),
                    (
                        "How to use the structure",
                        [
                            f"That structure is designed to support the goal of {profile.goal.lower()} without depending on one scheme or one return story.{scheme_line}",
                            "Use the sleeves as role definitions, then choose funds that fit each sleeve cleanly.",
                        ],
                    ),
                ],
            ),
            MODE_AUDITOR: (
                "Auditor View",
                [
                    (
                        "Strategic risk",
                        [
                            "The main strategic mistake to avoid is over-concentrating in one category simply because it looks strongest on trailing returns.",
                            f"If {leading_slice.label.lower()} becomes too dominant in practice, the plan can stop behaving like a balanced strategy and start behaving like a one-theme bet.{scheme_line}",
                        ],
                    ),
                    (
                        "What to protect against",
                        [
                            "The allocation sleeves should reduce concentration risk, role confusion, and overlap between similar funds.",
                            "A good blueprint is not just about upside; it is also about avoiding a fragile structure.",
                        ],
                    ),
                ],
            ),
            MODE_RESEARCHER: (
                "Researcher View",
                [
                    (
                        "Evidence-led structure",
                        [
                            f"The strategy blueprint reads strongest when each sleeve is treated as a portfolio role: {leading_slice.label.lower()} leads, while {supporting_slice.label.lower()} supports the structure.{scheme_line}",
                            "This keeps the recommendation grounded in role design instead of only scheme rankings.",
                        ],
                    ),
                    (
                        "What the blueprint explains",
                        [
                            f"The suggested weights are evidence-backed portfolio cues for the goal of {profile.goal.lower()}, not a claim that one category has permanently superior returns.",
                            "That distinction matters because role design usually ages better than return chasing.",
                        ],
                    ),
                ],
            ),
        },
    )

    key_points = {
        MODE_ADVISOR: [
            f"{slice_item.label}: {slice_item.allocation_pct}% suggested weight. {slice_item.rationale}"
            for slice_item in blueprint[:3]
        ],
        MODE_AUDITOR: [
            f"{leading_slice.label} is useful, but it should not swallow the rest of the structure.",
            "Use the sleeves to stop one attractive category from dominating the whole plan.",
            "Overlap control matters as much as allocation percentages.",
        ],
        MODE_RESEARCHER: [
            f"{leading_slice.label} leads because it carries the most portfolio weight in the current role design.",
            f"{supporting_slice.label} supports the blueprint by balancing the primary sleeve.",
            "The blueprint is stronger as a role-based system than as a return-ranking exercise.",
        ],
    }.get(assistant_mode, [])
    watchouts = [
        "Treat the blueprint as a decision-support model, not a fixed allocation rule.",
        "Recheck overlap before combining multiple funds from similar sub-categories.",
    ]
    return summary, answer, key_points, watchouts


def build_general_response(
    question: str,
    profile: InvestorProfile,
    related_schemes: list[Recommendation],
    citations: list[AnswerCitation],
    assistant_mode: str,
) -> tuple[str, str, list[str], list[str]]:
    if not related_schemes:
        return build_empty_response(profile)

    scheme_names = ", ".join(item.scheme_name for item in related_schemes[:2])
    summary = mode_summary(
        assistant_mode,
        advisor_text="The assistant is combining shortlist fit with supporting evidence to answer this question directly.",
        auditor_text="This question needs a cautious read so the answer does not jump from shortlist logic straight to conviction.",
        researcher_text="The answer is being built from both shortlist anchors and supporting evidence, with the evidence weighted more heavily in this lens.",
    )
    answer = format_mode_answer(
        assistant_mode,
        {
            MODE_ADVISOR: (
                "Advisor View",
                [
                    (
                        "Direct read",
                        [
                            f"For this profile, the most relevant shortlist anchors are {scheme_names}. They are being prioritized because they align better with the goal, risk range, and horizon than the average scheme in the dataset.",
                            f"Your question was interpreted as {classify_query_intent(question).replace('_', ' ')}, so the answer is staying close to the shortlist while still using supporting evidence.",
                        ],
                    ),
                    (
                        "How to use this answer",
                        [
                            "Use it to narrow the decision, then compare the leading funds against the latest official documents before acting.",
                            "The objective here is to move the decision forward, not to skip verification.",
                        ],
                    ),
                ],
            ),
            MODE_AUDITOR: (
                "Auditor View",
                [
                    (
                        "Cautious interpretation",
                        [
                            f"The shortlist anchors are still {scheme_names}, but the more important question is whether any hidden assumptions are being mistaken for certainty.",
                            "Auditor mode deliberately treats fit, data quality, and behavioural mismatch as questions to test rather than conclusions to trust immediately.",
                        ],
                    ),
                    (
                        "What to challenge",
                        [
                            "Check whether the apparent leader still looks strong after you question volatility tolerance, category overlap, and the need for fresh source confirmation.",
                            "A persuasive shortlist answer should survive challenge, not avoid it.",
                        ],
                    ),
                ],
            ),
            MODE_RESEARCHER: (
                "Researcher View",
                [
                    (
                        "Evidence-led interpretation",
                        [
                            f"The answer is centred on {scheme_names} because those names align most closely with the profile and also connect best to the available support material.",
                            f"Your question was interpreted as {classify_query_intent(question).replace('_', ' ')}, so the explanation leans more heavily on evidence than on recommendation language.",
                        ],
                    ),
                    (
                        "Support strength",
                        [
                            f"The current answer is drawing on {len(citations)} supporting reference points." if citations else "The current answer is operating with limited supporting material, so the evidence weight is lighter than usual.",
                            "Researcher mode is designed to show what the available support can justify and where it still falls short of full certainty.",
                        ],
                    ),
                ],
            ),
        },
    )

    key_points = {
        MODE_ADVISOR: [
            f"Profile anchor: {profile.goal}, {profile.risk_appetite} risk, {profile.horizon_years}-year horizon.",
            f"Shortlist anchor: {related_schemes[0].scheme_name} with score {related_schemes[0].score}.",
            "The answer is recommendation-first, with supporting evidence used to keep it grounded.",
        ],
        MODE_AUDITOR: [
            "The answer should survive challenge around risk, overlap, and data freshness.",
            f"Primary anchor under review: {related_schemes[0].scheme_name}.",
            "Shortlist confidence is useful, but not a substitute for contradiction testing.",
        ],
        MODE_RESEARCHER: [
            f"Evidence-backed anchor: {related_schemes[0].scheme_name}.",
            f"Support references used: {len(citations)}.",
            "This lens is evidence-first and recommendation-second.",
        ],
    }.get(assistant_mode, [])
    watchouts = [related_schemes[0].caution]
    return summary, answer, key_points, watchouts


def build_empty_response(profile: InvestorProfile) -> tuple[str, str, list[str], list[str]]:
    summary = "The assistant needs scheme context before it can generate a grounded answer."
    answer = (
        f"No shortlist was available for this {profile.goal.lower()} profile, so a grounded answer could not be composed yet. "
        "Load the dataset and regenerate the recommendations first."
    )
    return summary, answer, ["No scheme shortlist available."], ["Dataset context is missing."]


def assess_confidence(
    related_schemes: list[Recommendation],
    citations: list[AnswerCitation],
) -> tuple[str, str]:
    document_count = sum(1 for item in citations if item.source_type == "document")
    dataset_count = sum(1 for item in citations if item.source_type == "dataset")

    if related_schemes and dataset_count >= 1 and document_count >= 2:
        return "High", "Specific shortlist context was found and the answer is backed by both structured data and retrieved documents."
    if related_schemes and dataset_count >= 1:
        return "Medium", "The answer is well-grounded in structured scheme data, but document support is still limited."
    if document_count >= 2:
        return "Medium", "The answer has document grounding but only limited scheme-specific context."
    return "Low", "The answer is based on partial context and should be treated as exploratory."


def suggest_follow_ups(
    intent: str,
    related_schemes: list[Recommendation],
    profile: InvestorProfile,
) -> list[str]:
    scheme_name = related_schemes[0].scheme_name if related_schemes else "the top shortlisted fund"

    if intent == INTENT_COMPARE:
        return [
            f"Which of these compared schemes has the cleanest downside profile for a {profile.horizon_years}-year horizon?",
            f"Can you explain why {scheme_name} ranks above the others on suitability?",
            "Which shortlisted schemes are most likely to overlap with each other?",
        ]
    if intent == INTENT_DATA:
        return [
            "Which details should I verify before investing?",
            "Which shortlisted funds need the most careful factsheet review?",
            "What should I double-check if two shortlisted funds look very close?",
        ]
    if intent == INTENT_STRATEGY:
        return [
            "Which shortlisted fund should be the core holding in this blueprint?",
            "How can I keep this allocation diversified without using too many funds?",
            "Which sleeve in the blueprint is most important for downside control?",
        ]
    return [
        f"Can you explain the main caution behind {scheme_name} in more detail?",
        "Which shortlisted alternative is more conservative?",
        "What should I verify in the scheme factsheet before treating this as investable?",
    ]


def build_guardrail_note(
    recommendations: list[Recommendation],
    related_schemes: list[Recommendation],
) -> str:
    if any(item.nav_is_synthetic for item in related_schemes):
        return "At least one referenced fund should be cross-checked against the latest official NAV before any real-world decision."
    if any(item.master_match_type == "unmatched" for item in related_schemes):
        return "Some referenced funds have limited supporting detail, so the latest official factsheet should be checked before any real-world use."
    if not recommendations:
        return "No recommendation context was available, so this answer is only partially grounded."
    return "Educational use only. This assistant is designed for grounded decision support, not guaranteed financial advice."


def build_prompt_pack(
    question: str,
    profile: InvestorProfile,
    related_schemes: list[Recommendation],
    citations: list[AnswerCitation],
    assistant_mode: str,
    intent: str,
) -> PromptPack:
    scheme_context = "\n".join(
        f"- {item.scheme_name} | score {item.score} | {item.fit_label} | risk {risk_bucket(item.risk_level)} | expense {item.expense_ratio:.2f}% | NAV source {item.nav_source}"
        for item in related_schemes[:3]
    ) or "- No specific scheme anchor was matched."

    context_block = "\n\n".join(
        f"[{index}] {citation.title}\nSource: {citation.source_label or Path(citation.path).name}\nPages: {', '.join(str(p) for p in citation.page_numbers) if citation.page_numbers else 'dataset'}\nURL: {citation.source_url or 'local-only'}\nTrust: {citation.trust_tier}\nNote: {citation.note}\nSnippet: {citation.snippet}"
        for index, citation in enumerate(citations, start=1)
    )

    system_prompt = (
        "You are a citation-first mutual fund recommendation assistant. "
        "Use only the provided scheme context and retrieved evidence. "
        "Do not guarantee returns, do not present synthetic NAV as live market data, and say when evidence is limited. "
        "Structure the response as: direct answer, reasoning, watch-outs, and cited support."
    )
    user_prompt = (
        f"Assistant mode: {assistant_mode}\n"
        f"Intent: {intent}\n"
        f"Investor profile:\n"
        f"- Goal: {profile.goal}\n"
        f"- Risk appetite: {profile.risk_appetite}\n"
        f"- Horizon: {profile.horizon_years} years\n"
        f"- Mode: {profile.investment_mode}\n"
        f"- Amount: Rs. {profile.amount:,}\n"
        f"- Tax saving needed: {'Yes' if profile.needs_tax_saving else 'No'}\n\n"
        f"Relevant schemes:\n{scheme_context}\n\n"
        f"Question: {question}"
    )

    return PromptPack(system_prompt=system_prompt, user_prompt=user_prompt, context_block=context_block)


def truncate_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def format_percent_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def format_nav_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def holding_period_hint(item: Recommendation) -> str:
    tags = f"{item.category} {item.sub_category} {item.scheme_name}".lower()

    if "overnight" in tags or "liquid" in tags:
        return "up to about 1 year"
    if "ultra short" in tags or "money market" in tags:
        return "around 1 to 2 years"
    if "low duration" in tags or "short duration" in tags:
        return "around 1 to 3 years"
    if "banking and psu" in tags or "corporate bond" in tags or "dynamic bond" in tags:
        return "around 2 to 4 years"
    if "arbitrage" in tags or "equity savings" in tags or "balanced advantage" in tags:
        return "around 3 to 5 years"
    if "aggressive hybrid" in tags or "conservative hybrid" in tags:
        return "around 3 to 5 years"
    if "elss" in tags:
        return "3 years and beyond"
    if "small cap" in tags or "sectoral" in tags or "thematic" in tags or "mid cap" in tags:
        return "7 years and beyond"
    if "index" in tags or "flexi cap" in tags or "large cap" in tags or "large and mid cap" in tags:
        return "5 years and beyond"
    if item.category == "Debt":
        return "around 2 to 4 years"
    if item.category == "Hybrid":
        return "around 3 to 5 years"
    if item.category in {"Equity", "Solution Oriented"}:
        return "5 years and beyond"
    return "medium to long term"
