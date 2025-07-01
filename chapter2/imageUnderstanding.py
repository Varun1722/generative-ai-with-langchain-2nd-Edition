from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def analyse_image(img_url: str, question: str) -> str:
    llm = ChatOpenAI(model='gpt-4o-mini',max_completion_tokens=256)

    message = HumanMessage(
        content=[
            {'type':'text',
            'text':question},

            {
                'type':'image_url',
                'image_url':{
                    'url':img_url,
                    'details':'auto'   
                }
            }

        ]
    )

    result = llm.invoke([message])
    return result.content

image_url = "https://images.unsplash.com/photo-1575936123452-b67c3203c357?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8aW1hZ2V8ZW58MHx8MHx8fDA%3D"
question = [
    "Describe this image in 50 words",
    "Describe the emotion of the image in 5 words"
]

for q in question:
    print(f'Q:{q}')
    print(f'Answer: {analyse_image(image_url,q)}')