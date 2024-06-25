import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_cohere import ChatCohere

load_dotenv()

# Set Cohere API key from environment variables
os.environ["COHERE_API_KEY"] = "dSFne26NhUTR2VABw98IOvkoDaCVObjwGJXf31aK"

# Load environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize ChatCohere
llm = ChatCohere(api_key=COHERE_API_KEY)

# Define tools (example tools)
search_tool = Tool(
    name="Web Search",
    func=lambda input_text: f"Performing web search for: {input_text}",
    description="A tool for performing web searches.",
)

wikipedia_tool = Tool(
    name="Wikipedia",
    func=lambda input_text: f"Searching Wikipedia for: {input_text}",
    description="A tool for searching Wikipedia.",
)

# Define prompts (example prompts)
prompt = PromptTemplate(
    template="""Plan: {input}

History: {chat_history}

Let's think about the answer step by step.
If it's an information retrieval task, solve it like a professor in a particular field.""",
    input_variables=["input", "chat_history"],
)

plan_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""Prepare a plan for task execution. (e.g. retrieve current date to find weather forecast)

    Tools to use: wikipedia, web search

    REMEMBER: Keep in mind that you don't have information about the current date, temperature, information after September 2021. Because of that, you need to use tools to find them.

    Question: {input}

    History: {chat_history}

    Output looks like this:
    '''
        Question: {input}

        Execution plan: [execution_plan]

        Rest of needed information: [rest_of_needed_information]
    '''

    IMPORTANT: if there is no question, or the plan is not needed (YOU HAVE TO DECIDE!), just populate {input} (pass it as a result). Then the output should look like this:
    '''
        input: {input}
    '''
    """,
)

# Define memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up conversation chain
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    input_key="input",
    prompt=plan_prompt,
    output_key="output",
)

# Initialize Agent
# Note: The exact agent initialization depends on the latest conventions in langchain-cohere
# Check the documentation or examples provided by langchain-cohere for the correct usage

# For demonstration purposes, create a simple function to handle agent interactions
def handle_agent_interaction(input_text):
    response = conversation_chain.invoke(input_text)
    return response['output']

# Example usage:
if __name__ == "__main__":
    user_input = "Where is India?"
    agent_response = handle_agent_interaction(user_input)
    print(f"Agent Response: {agent_response}")