from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

'''
This project has two chains:
1) Generate a story 
2) Analyse the story
'''

#Chain to generate the story
prompt = PromptTemplate.from_template("Generate a short story related to {topic}")
output = StrOutputParser()

chain1 = prompt|model|output
# story = chain1.invoke({"topic":"Dogs"})

#Analyse the story 
prompt_analyse = PromptTemplate.from_template("Analyse the following story's mood: {story}" \
"Give the analysis a emotion tag and keep the output at 10 words maximum")

chain_analysis = prompt_analyse|model|output

#merging both the chains 
merged_chain = chain1|chain_analysis
# final_result = merged_chain.invoke({"topic":"Dogs"})
# print(final_result)

merged_chain_preserve_data = chain1|{"analysis":chain_analysis}
final_result = merged_chain_preserve_data.invoke({"topic":"Dogs"})
print(final_result.keys())  #dict_keys(['analysis'])

# The problem with the above approach is that we lose our original story

# Using RunnablePassthrough.assign to preserve the data
# runnable_chain = RunnablePassthrough().assign(
#     story=chain1 #Add story key with generated content
# ).assign(
#     analysis = chain_analysis #add analysis key with actual analysis
# )

# result = runnable_chain.invoke({"topic":"Dogs"})
# print(result['topic'])
# print(result['story'])
# print(result['analysis'])