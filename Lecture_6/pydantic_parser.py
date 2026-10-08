from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
model = ChatOpenAI()

class Person(BaseModel):
    name: str = Field(description='name of the person')
    age: int = Field(gt=18, description='age of the person')
    city: str = Field(description='Name of the city to which the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template =PromptTemplate(
    template='Generate the name age and city of a fictional {place} person \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()} 
)

chain = template | model | parser
res = chain.invoke({'place': 'USA'})
print(res)

# prompt = template.invoke({'place': 'italian'})
# result = model.invoke(prompt)
# final = parser.parse(result.content)
# print(final)

018905014684

