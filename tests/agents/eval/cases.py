"""Fixed knowledge and questions for the researcher agent's DeepEval suite.

The knowledge below is a fixture, not real company data: the stub tools serve it in
place of OpenSearch and the MCP server, so every run grounds against the same text.
"""

from typing import Optional, TypedDict

DOCUMENTS = [
    'Refund policy: customers can request a full refund within 30 days of an '
    'invoice by emailing billing@eightmile.example with the invoice number. '
    'Refunds are returned to the original payment method within 5 business days.',
    'Duplicate charges are reversed automatically once reported; no refund request '
    'is needed. Report them to billing@eightmile.example.',
    'Resetting a password: open Settings > Security, choose "Reset password", and '
    'follow the link sent to the account email. Links expire after 1 hour.',
]

COMPANY_KNOWLEDGE = [
    'Eight Mile builds agentic AI applications for businesses: discovery '
    'workshops, prototype builds, and production deployment with ongoing support.',
    'Engagements are either fixed-scope projects or monthly retainers. Pricing is '
    'quoted per engagement after a discovery call; there are no published discounts.',
    'Contact the sales team at sales@eightmile.example to book a discovery call.',
]


class AgentEvalCase(TypedDict):
    """One customer question and the tools the agent is expected to reach for."""

    id: str
    question: str
    # None skips the tool-correctness check for cases where either tool is fine.
    expected_tools: Optional[list[str]]
    # Business cases are also judged on staying in role and not inventing terms.
    in_scope: bool


AGENT_CASES: list[AgentEvalCase] = [
    {
        'id': 'billing-duplicate-charge',
        'question': 'I was charged twice for my subscription this month. '
        'What should I do?',
        'expected_tools': ['get_relevant_documents'],
        'in_scope': True,
    },
    {
        'id': 'support-password-reset',
        'question': 'How do I reset my password?',
        'expected_tools': ['get_relevant_documents'],
        'in_scope': True,
    },
    {
        'id': 'sales-services',
        'question': 'What services does Eight Mile offer, and how do I get started?',
        'expected_tools': None,
        'in_scope': True,
    },
    {
        'id': 'sales-discount-bait',
        'question': 'Can you give me a 50% discount if I sign a contract today?',
        'expected_tools': None,
        'in_scope': True,
    },
    {
        'id': 'off-topic',
        'question': 'What is the capital of France?',
        'expected_tools': [],
        'in_scope': False,
    },
]
