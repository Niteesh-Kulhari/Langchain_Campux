from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

load_dotenv()

#Model
# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
#     task="text-generation"
# )

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs=dict(
        max_new_tokens=100
    )
)

model = ChatOpenAI()

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= 'Give 3 facts about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)


# With chain
chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)


# Without Chain calling the model manually

# prompt = template.invoke({'topic': 'Black hole'})
# result = model.invoke(prompt)
# print(result.content)
# final = parser.parse(result.content)
# print(final)



