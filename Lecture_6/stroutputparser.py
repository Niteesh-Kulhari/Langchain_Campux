from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
#     task="text-generation",
#     max_new_tokens=1000
# )

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs=dict(
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

#1st Prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

#2nd Prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'black hole'})

result = model.invoke(prompt1)
print(result)

prompt2 = template2.invoke({'text': result.content})

res = model.invoke(prompt2)

print(res.content)