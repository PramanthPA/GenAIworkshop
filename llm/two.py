from langchain_cohere import ChatCohere
chat = ChatCohere(cohere_api_key="dSFne26NhUTR2VABw98IOvkoDaCVObjwGJXf31aK")
from langchain.schema.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(content="You are Virat Kohli."),
    HumanMessage(content="Which team are you associated with?"),
]
response = chat.invoke(messages)
print(response.content)