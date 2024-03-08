import json
from datetime import datetime
import streamlit as st
from utils.chains import RAG, load_and_create_vectordb

ALLOWED_HOSTS = ['*']
chat_data = []

@st.cache_resource() # Cache the result of this function
def init_vectordb():
    return load_and_create_vectordb("dummy_data.pdf")

new_rds = init_vectordb()

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


        print("Initializing RAG")
        rag = RAG(llm_name="gemini-pro", temperature=0, vector_db=new_rds, results_to_retrieve=3, max_tokens=200)
        rag_chain = rag.run()

        with open("chat_history.json", "r") as file:
            json_content = file.read()
        chat_data = json.loads(json_content)

        if len(chat_data) > 2:
            msg_history = chat_data[-2:]
        else:
            msg_history = chat_data

        result = rag_chain.invoke({"question": query, "query_language": "English", "chat_history": msg_history})

        end_time = datetime.now()
        print("Answer Ready!")
        print(f"Time took: {end_time - start_time}")

        msg_history.append({"Human": query, "AI": result['chain_output']})
        with open("chat_history.json", "w") as file:
            json.dump(msg_history, file)

        
        st.write("---")
        st.write(f"Human: ", query)
        st.write(f"AI:", result["chain_output"])
        st.write("---")

    except Exception as e:
        st.error(str(e))


if __name__ == '__main__':
    main()
