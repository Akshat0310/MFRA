# Requirement Analysis

## Scope

The initial release will recommend mutual fund categories based on user profile. A later phase will extend this to specific scheme comparison using retrieved fund documents and structured metrics.

## Functional requirements

1. Collect user profile inputs:
   - age
   - investment goal
   - risk appetite
   - investment horizon
   - investment mode
   - amount
   - tax-saving intent
2. Generate a ranked list of suitable fund categories
3. Explain why each recommendation matches the user profile
4. Show caution notes for mismatched or higher-risk categories
5. Support future document-based question answering using RAG

## Non-functional requirements

- responses should be easy to understand for non-experts
- recommendations should be explainable and auditable
- the app should be lightweight and demo-friendly
- the system should support future evaluation for relevance and hallucination control

## Assumptions

- the user is comfortable entering a small set of personal investment preferences
- the first milestone can work with category-level knowledge before scheme-level data is integrated
- the final project will use publicly available educational or scheme-related documents

## Constraints

- no recommendation should claim guaranteed returns
- the system should avoid pretending to be a certified financial advisor
- recommendations should remain conservative when user inputs are incomplete or contradictory
- scheme-level intelligence depends on availability and quality of real mutual fund datasets and PDFs

## Inputs

- age
- goal
- risk appetite
- horizon in years
- monthly SIP or lump sum choice
- amount
- tax-saving preference

## Outputs

- top recommended fund categories
- match score
- reasoning and caution summary
- next best action for the user

## Future expansion

- upload current portfolio for diversification analysis
- compare multiple schemes with grounded citations
- goal-based SIP planning
- tax-aware explanations
- conversational assistant over scheme documents

