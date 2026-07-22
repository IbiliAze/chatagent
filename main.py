import json

from langchain_core.messages import HumanMessage

from app.agents.researcher.agent import ResearcherAgent
from app.common.models.models import Models

models = Models()
agent = ResearcherAgent(models)
agent.get_graph_png()

result = agent.process_message(
  {'messages': [HumanMessage('Hi')], 'error': '', 'model_used': '', 'retry_count': 0}
)

for message in result['messages']:
  print(message)
