import streamlit as st
from query import ask_database

# Configure the Streamlit page
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="📈",
    layout="centered"
)

st.title("AI Financial Analyst")
st.write("Ask me anything about Apple & Tesla SEC 10-K report!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! What would you like to know?"}
    ]

# Display the Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("E.g., What is the company's total revenue?"):
    
    # display the user's message on the screen
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save the user's message to memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and Display the AI Response
    with st.chat_message("assistant"):
        # Show a loading spinner while query.py script working
        with st.spinner("Analyzing the document..."):
            
            # Call the LCEL chain in query.py
            ai_response = ask_database(prompt)
            
            # Display the answer
            st.markdown(ai_response)
            
    # Save the AI's response to memory
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
