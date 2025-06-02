import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv(".env")
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

conv_memory = ConversationBufferMemory()    # This object will just store the conversation history in a buffer
conv_sumary_memory = ConversationSummaryMemory(llm=llm) # This memory will summarize the conversation history
conversation_chain = ConversationChain(llm=llm, memory=conv_sumary_memory)
while True:
    user_input = input("Hi, what would you like to know: ")
    response = conversation_chain.invoke({"input": user_input})
    print(f"AI: {response['response']}")
    print(f"token count for chat history is {response['history']}")
# print(conversation_chain.prompt.template)
# print(conversation_chain.memory.buffer)
# print(conv_sumary_memory.memory.prompt) 