from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()
passthrough = RunnablePassthrough()

prompt1 = PromptTemplate(
    template='Generate a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Give explanation of the joje - {text}',
    input_variables=['text']
)

chain1 = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(chain1, parallel_chain)
res = final_chain.invoke({'topic': 'AI'})
print(res)