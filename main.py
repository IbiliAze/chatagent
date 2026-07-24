from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.researcher.agent import ResearcherAgent
from app.agents.researcher.nodes import ResearcherNodes
from app.agents.researcher.tools import ResearcherTools
from app.common.models.models import Models
from app.common.rag.rag import Rag
from core.config.settings import get_settings

settings = get_settings()
models_to_use = Models()
embedding_llm = models_to_use.embedding_llm
vectorstore = OpenSearchVectorSearch(
  opensearch_url=settings.opensearch_url,
  index_name=settings.opensearch_documents_index,
  embedding_function=embedding_llm,
  http_auth=(
    (settings.opensearch_user, settings.opensearch_password)
    if settings.opensearch_user
    else None
  ),
)
rag = Rag(vectorstore)

if rag.get_document_count() == 0:
  rag.add_texts(
    [
      'TradeOps is our internal platform for managing trade lifecycle '
      'operations, including trade capture, settlement, and reconciliation.',
      'Eight Mile Services is a vendor providing outsourced back-office '
      'support for trade settlement and client onboarding.',
    ],
    source='seed',
  )


db_path = 'checkpoints.db'

with SqliteSaver.from_conn_string(db_path) as saver:
  print(f'DB was created at {db_path}')
  models = Models()
  tools = ResearcherTools(rag=rag)
  tool_list = [tools.get_relevant_documents]
  nodes = ResearcherNodes(models=models, tools=tool_list)
  agent = ResearcherAgent(nodes=nodes, saver=saver)
  agent.get_graph_png()
  config: RunnableConfig = {'configurable': {'thread_id': '123'}}

  agent.get_graph_png()
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
  agent.process_message(
    {
      'messages': [HumanMessage('What is the capital of Azerbaijan?')],
      'error': '',
      'model_used': '',
      'retry_count': 0,
    },
    config=config,
  )
  agent.process_message(
    {
      'messages': [
        HumanMessage('What do you know about tradeops or eight mile services?')
      ],
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

  # print(agent.get_current_state(config))
  # print()

  # for i, snapshot in enumerate(agent.get_state_history(config)):
  #   print(f' {i}: {snapshot.values["messages"]}')
