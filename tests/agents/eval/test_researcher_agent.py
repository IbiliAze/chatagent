"""DeepEval quality checks on the researcher agent's end-to-end answers.

Runs the real graph and real models, but serves retrieval from stub tools holding a
fixed corpus, so a failing score points at the prompts or the model rather than at
OpenSearch contents or MCP availability. Run with `pytest -m eval`.
"""

import os
from uuid import uuid4

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    FaithfulnessMetric,
    GEval,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from app.agents.researcher.agent import ResearcherAgent
from app.agents.researcher.nodes import ResearcherNodes
from app.agents.researcher.routes import ResearcherRoutes
from core.models.models import Models
from tests.agents.eval.cases import (
    AGENT_CASES,
    COMPANY_KNOWLEDGE,
    DOCUMENTS,
    AgentEvalCase,
)

load_dotenv()

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not (os.environ.get('OPENAI_API_KEY') and os.environ.get('ANTHROPIC_API_KEY')),
        reason='requires real OPENAI_API_KEY and ANTHROPIC_API_KEY',
    ),
]

# A different, stronger model than the primary under test (gpt-4o-mini), so the
# agent is not grading its own answers.
JUDGE_MODEL = 'gpt-4o'


def _format_passages(passages: list[str]) -> str:
    """Format passages the way Rag._format_docs_for_context does."""
    return '\n\n---\n\n'.join(
        f'[Source {i + 1}: fixture]\n{passage}' for i, passage in enumerate(passages)
    )


def get_relevant_documents(query: str) -> str:
    """Search for information relevant to the query from RAG."""
    del query  # every query gets the whole fixture corpus
    return _format_passages(DOCUMENTS)


def search_company_knowledge(query: str) -> str:
    """Search Eight Mile's company knowledge pages: services, technical
    capabilities, engagement models, pricing approach and contact information."""
    del query
    return _format_passages(COMPANY_KNOWLEDGE)


@pytest.fixture(scope='module')
def agent() -> ResearcherAgent:
    """Build the researcher graph on real models with stub retrieval tools."""
    tools: list[BaseTool] = [
        StructuredTool.from_function(get_relevant_documents),
        StructuredTool.from_function(search_company_knowledge),
    ]
    return ResearcherAgent(
        ResearcherNodes(Models(), tools), ResearcherRoutes(), MemorySaver()
    )


def run_agent(agent: ResearcherAgent, question: str) -> LLMTestCase:
    """Run one question through the graph and capture what the metrics need."""
    state = agent.process_message(
        agent.build_message(question), agent.build_config(str(uuid4()))
    )
    messages = state['messages']
    return LLMTestCase(
        input=question,
        actual_output=messages[-1].text,
        retrieval_context=[m.text for m in messages if isinstance(m, ToolMessage)],
        tools_called=[
            ToolCall(name=call['name'], input_parameters=call['args'])
            for m in messages
            if isinstance(m, AIMessage)
            for call in m.tool_calls
        ],
    )


def build_metrics(case: AgentEvalCase, test_case: LLMTestCase) -> list[BaseMetric]:
    """Pick the metrics that apply to this case and what the agent actually did."""
    metrics: list[BaseMetric] = [
        AnswerRelevancyMetric(threshold=0.7, model=JUDGE_MODEL),
    ]
    if test_case.retrieval_context:
        metrics.append(FaithfulnessMetric(threshold=0.8, model=JUDGE_MODEL))
    if case['expected_tools']:
        metrics.append(ToolCorrectnessMetric(threshold=1.0))
    if case['in_scope']:
        metrics.append(
            GEval(
                name='Stays in role',
                criteria=(
                    'The actual output answers as an Eight Mile customer service '
                    'representative. It must not invent prices, discounts, contract '
                    'terms, deadlines or commitments that are absent from the input '
                    'and the retrieval context. Declining or redirecting to the '
                    'right team is acceptable when the context has no answer.'
                ),
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.RETRIEVAL_CONTEXT,
                ],
                threshold=0.7,
                model=JUDGE_MODEL,
            )
        )
    return metrics


class TestResearcherAgentEval:
    """Scores the agent's answers on relevance, grounding, tool use and role."""

    @pytest.mark.parametrize('case', AGENT_CASES, ids=[c['id'] for c in AGENT_CASES])
    def test_answer_meets_quality_bar(
        self, agent: ResearcherAgent, case: AgentEvalCase
    ) -> None:
        """The agent's answer clears every metric that applies to the case."""
        test_case = run_agent(agent, case['question'])
        test_case.expected_tools = [
            ToolCall(name=name) for name in case['expected_tools'] or []
        ]

        # ToolCorrectnessMetric cannot express "call nothing", so assert it directly.
        if case['expected_tools'] == []:
            assert not test_case.tools_called

        assert_test(test_case, build_metrics(case, test_case))
