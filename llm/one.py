from langchain_cohere.llms import Cohere
llm = Cohere(cohere_api_key="dSFne26NhUTR2VABw98IOvkoDaCVObjwGJXf31aK")
response = llm.invoke("List the seven wonders of the world.")
print(response)
for chunk in llm.stream("Where were the 2012 Olympics held?"):
    print(chunk, end="", flush=True)
response = llm.invoke("Hello, how can I help you today?")
print(response)