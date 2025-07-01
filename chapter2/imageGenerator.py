from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from dotenv import load_dotenv
from IPython.display import Image, display
import requests

load_dotenv()

llm = DallEAPIWrapper(
    model='dall-e-3',
    size='1024x1024',
    quality="standard",
    n=1
)

#generate the image
image_url = llm.run("A child running on the banks of a river")

#display the image
display(Image(url=image_url))

#save it locally
output = requests.get(image_url)

with open("image_generated.png",'wb') as f:
    f.write(output.content)
