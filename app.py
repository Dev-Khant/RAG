from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from datetime import datetime
import streamlit as st


GEMINI_KEY="AIzaSyDzzRUVOxOfk3A1F1gWV7_pJX4q4jUeTGg"


ALLOWED_HOSTS = ['*']
chat_data = []

@st.cache_resource() # Cache the result of this function
def init_agent():
    csv_google_agent = create_csv_agent(ChatGoogleGenerativeAI(model="gemini-pro", 
                                                    google_api_key=GEMINI_KEY,
                                                    temperature=0
                                                    ), 
        "search_contents_2024-02-07.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    return csv_google_agent

csv_agent = init_agent()

def main():
    st.title("AI Chatbot")

    query = st.text_input("Enter your query:")

    if st.button("Generate Response"):
        if not query:
            st.error("Please enter a query.")
        else:
            generate_response(query)

def generate_response(query):
    try:
        global chat_data

        start_time = datetime.now()
        result = csv_agent.run(query)
        end_time = datetime.now()

        print("Answer Ready!")
        print(f"Time took: {end_time - start_time}")

        
        st.write("---")
        st.write(f"Human: ", query)
        st.write(f"AI:", result)
        st.write("---")

    except Exception as e:
        st.error(str(e))


if __name__ == '__main__':
    main()
