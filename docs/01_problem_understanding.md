# Problem Understanding

## Domain context

Mutual fund investors often face information overload. They must compare risk, returns, expense ratio, exit load, lock-in rules, asset allocation, and investment horizon across many schemes. Beginners especially struggle to convert this information into a confident investment decision.

## Core problem

Users need a system that can translate personal investment preferences into understandable, suitable mutual fund recommendations without overwhelming them with jargon.

## Target outcome

Build an assistant that:

- understands the investor profile and goals
- recommends suitable mutual fund categories and later specific schemes
- explains the reasoning in simple language
- highlights risks and trade-offs
- reduces confusion and improves trust through transparency

## Primary users

1. Beginner investors who do not know which fund category fits their profile
2. Young professionals planning SIPs for long-term wealth creation
3. Investors saving for specific goals such as emergency fund, tax saving, or retirement

## Pain points

- too many choices with confusing terminology
- tendency to choose funds only by recent returns
- lack of understanding of risk and investment horizon
- low trust in black-box recommendations
- no easy way to compare scheme intent with personal goals

## Why AI and LLM are useful here

- AI can convert user inputs into a structured investor profile
- ranking logic can shortlist suitable options consistently
- an LLM can explain recommendations in natural language
- RAG can ground answers using real scheme documents and factsheets

## Success criteria

- recommendations align with investor risk and time horizon
- explanations are clear and traceable
- the system minimizes hallucination by grounding responses in data
- users can understand why a category is recommended or avoided

