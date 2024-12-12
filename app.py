import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import utils
import os
import io
import textwrap

# Sidebar contents
with st.sidebar: 
    st.title('🌨️ LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Ollama](https://ollama.com/) LLM model
    ''')
    add_vertical_space(5)

def main():
    st.header("Chat with PDF")
    load_dotenv()

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text=text)

        # Embeddings setup
        store_name = pdf.name[:-4]  # Remove the .pdf extension

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = Ollama(model="llama2")

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            # Display the response
            st.write(response)

            # Generate a PDF containing the response with word wrapping
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            width, height = letter
            x = 50  # Left margin
            y = height - 50  # Top margin

            # Define max line width and wrap each line to fit the page
            max_line_width = 80  # Adjust based on font and page size
            wrapped_query = textwrap.fill("Question: " + query, max_line_width)
            wrapped_response = textwrap.wrap("Answer:\n" + response, max_line_width)

            # Draw the wrapped question in bold
            c.setFont("Times-Bold", 12)  # Set font to Times-Bold for question
            for line in wrapped_query.splitlines():
                c.drawString(x, y, line)
                y -= 15

            # Draw each line of the wrapped response with pagination in regular font
            c.setFont("Times-Roman", 12)  # Set font to Times-Roman for answer
            for line in wrapped_response:
                if y < 40:  # If near the bottom margin, create a new page
                    c.showPage()
                    y = height - 50
                    c.setFont("Times-Roman", 12)  # Reset font after new page
                c.drawString(x, y, line)
                y -= 15

            c.save()

            # Convert the buffer to a downloadable PDF file
            pdf_buffer.seek(0)
            st.download_button(
                label="Download Answer as PDF",
                data=pdf_buffer,
                file_name="answer.pdf",
                mime="application/pdf"
            )

if __name__ == '__main__':
    main()
