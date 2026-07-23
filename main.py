import tempfile

from langchain_core.messages import HumanMessage

# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.researcher.agent import ResearcherAgent, ThreadConfig
from app.agents.researcher.nodes import ResearcherNodes
from app.common.models.models import Models

with tempfile.NamedTemporaryFile(suffix='.db', delete=True) as f:
  db_path = f.name

with SqliteSaver.from_conn_string(db_path) as saver:
  print(f'DB was created at {db_path}')
  models = Models()
  nodes = ResearcherNodes(models=models)
  agent = ResearcherAgent(nodes=nodes, saver=saver)
  agent.get_graph_png()
  config: ThreadConfig = {'configurable': {'thread_id': '123'}}

  agent.process_message(
    {
      'messages': [HumanMessage('Hi, my name is ibi')],
      'error': '',
      'model_used': '',
      'retry_count': 0,
    },
    config=config,
  )
  agent.process_message(
    {
      'messages': [HumanMessage('How are you?')],
      'error': '',
      'model_used': '',
      'retry_count': 0,
    },
    config=config,
  )
  result = agent.process_message(
    {
      'messages': [HumanMessage('What is my name?')],
      'error': '',
      'model_used': '',
      'retry_count': 0,
    },
    config=config,
  )

  for message in result['messages']:
    print(message.content)
    print()

  print(agent.get_current_state(config))
  print()

  for i, snapshot in enumerate(agent.get_state_history(config)):
    print(f' {i}: {snapshot.values["messages"]}')
