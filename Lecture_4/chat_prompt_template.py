from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')

    #This does not work the same way as prompt template
    # SystemMessage(content='You are a helpful {domain} expert'),
    # HumanMessage(content='Explain in simple terms, what is the {topic}')
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'LBW'})

print(prompt)