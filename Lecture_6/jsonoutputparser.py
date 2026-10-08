from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task = "text-generation"
)

parser = JsonOutputParser()
model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template= 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
res = chain.invoke({})

# prompt = template.format()
# res = model.invoke(prompt)

print(res)
