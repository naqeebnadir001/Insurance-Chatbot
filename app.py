from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import os
os.environ["TOGETHER_API_KEY"] = "808314f23415a0b647a83b4ce6ff7082302278e554fd37e977b65601c1c5dda4"
import re
import logging
import requests
from decimal import Decimal
from docx import Document
from mysql.connector import Error
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Together
from langchain.prompts import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System message to set the context for the chatbot
SYSTEM_MESSAGE = """
You are an insurance chatbot designed to assist users with insurance-related queries. Your role is to:
1. Provide clear and concise answers to questions about insurance policies, products, and processes.
2. Guide users through the policy creation process.
3. Avoid providing irrelevant or unsolicited information.
4. If you don't know the answer, politely inform the user and suggest alternative ways to find the information.
5. Always maintain a professional and friendly tone.
"""

CUSTOMER_FIELDS = ['first_name', 'last_name', 'email', 'phone_number', 'cnic',
                   'address', 'office_address', 'poc_name', 'poc_number',
                   'poc_cnic', 'relationship_with_customer']

DEVICE_FIELDS = ['brand_name', 'device_model', 'device_serial_number',
                 'purchase_date', 'device_value', 'device_condition', 'warranty_status']

class ConversationState:
    def __init__(self):
        self.phase = None  # 'customer' or 'device'
        self.missing_fields = []
        self.collected_data = {}

class DatabaseHandler:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def get_connection(self):
        try:
            conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            return conn
        except Error as e:
            logger.error(f"Error connecting to MySQL database: {e}")
            return None

    def get_customer_info(self, cnic):
        conn = self.get_connection()
        if not conn:
            return None

        cursor = conn.cursor(dictionary=True)
        query = "SELECT first_name, last_name, email, phone_number FROM customers WHERE cnic = %s"
        cursor.execute(query, (cnic,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result

    def get_device_info(self, device_serial_number):
        conn = self.get_connection()
        if not conn:
            return None

        cursor = conn.cursor(dictionary=True)
        query = "SELECT brand_name, device_model, purchase_date, device_value, device_condition, warranty_status FROM devices WHERE device_serial_number = %s"
        cursor.execute(query, (device_serial_number,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result

    def save_customer_info(self, customer_data):
        invalid_fields = []
        for field, value in customer_data.items():
            if value is None:
                invalid_fields.append(field)
                continue
            if field == 'first_name' and re.search(r'\d',value):
                invalid_fields.append(field)
            elif field == 'last_name' and re.search(r'\d', value):
                invalid_fields.append(field)
            elif field == 'email' and not value.endswith('@gmail.com'):
                invalid_fields.append(field)
            elif field == 'phone_number' and not re.match(r'^03\d{9}$', value):
                invalid_fields.append(field)
            elif field == 'cnic' and not re.match(r'^\d{5}-\d{7}-\d{1}$', value):
                invalid_fields.append(field)
        return invalid_fields

    def save_device_info(self, device_data):
        invalid_fields = []
        for field, value in device_data.items():
            if value is None:
                invalid_fields.append()
                continue
            
            if field == 'brand_name' and not re.match(r'^[A-Za-z ]+$', value):
                invalid_fields.append(field)
            elif field == 'device_value' and not re.match(r'^\d+$', value):
                invalid_fields.append(field)
        return invalid_fields

def check_and_collect_customer(db_handler, state, cnic):
    customer = db_handler.get_customer_info(cnic)
    if customer:
        state.collected_data.update(customer)
        return True
    else:
        state.phase = 'customer'
        state.missing_fields = CUSTOMER_FIELDS.copy()
        state.missing_fields.remove('cnic')  # Already have CNIC
        state.collected_data['cnic'] = cnic
        return False

def check_and_collect_device(db_handler, state, device_serial_number):
    device = db_handler.get_device_info(device_serial_number)
    if device:
        state.collected_data.update(device)
        return True
    else:
        state.phase = 'device'
        state.missing_fields = DEVICE_FIELDS.copy()
        state.missing_fields.remove('device_serial_number')  # Already have device serial number
        state.collected_data['device_serial_number'] = device_serial_number
        return False

def generate_missing_field_prompt(state, qa_chain):
    return f"Please provide your {state.missing_fields[0]}."

class DocumentHandler:
    @staticmethod
    def load_documents(file_path):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

class VectorStoreHandler:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def create_vector_store(self, documents):
        vectorstore = Chroma.from_texts([documents], self.embeddings, persist_directory=self.persist_directory)
        vectorstore.persist()
        return vectorstore

    def load_or_create_vectorstore(self, documents):
        if os.path.exists(self.persist_directory):
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        else:
            return self.create_vector_store(documents)

class Chatbot:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = Together(model="mistralai/Mistral-7B-Instruct-v0.1")
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.vectorstore.as_retriever())

    def generate_response(self, query, documents):
        try:
            full_query = f"{SYSTEM_MESSAGE}\n\nUser Query: {query}"
            docs = self.vectorstore.similarity_search(full_query, k=5)
            context = "\n".join([doc.page_content for doc in docs])

            if not context:
                return "I'm sorry, I couldn't find relevant information to answer your question. Please try rephrasing or ask another question."

            prompt = f"{SYSTEM_MESSAGE}\n\nUser Query: {query}\n\nBased on the following information, generate a conversational response:\n\n{documents}\n\nBot:"
            response = self.qa_chain.invoke(prompt)
            return response if isinstance(response, str) else response.get("result", "Error generating response")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, something went wrong while processing your request. Please try again."

BASE_URL = "https://insurance-crm-backend.vercel.app/api/policies"
HEADERS = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOjIsImlhdCI6MTczOTcxNjE4MSwiZXhwIjoxNzQzMzE2MTgxfQ.UbFkBilRXmEuEuYvA_0nuo2M1oVJPm7ren24Fo53wZ4",
    "Content-Type": "application/json"
}

def create_policy(state):
    post_data = {
        "license_type": "Conventional",
        "device_type": "Mobile"
    }
    response = requests.post(BASE_URL, json=post_data, headers=HEADERS)
    if response.status_code == 201:
        state.policy_id = response.json().get("policy_id")
        print(f"Policy created successfully with ID: {state.policy_id}")
    else:
        print(f"Failed to create policy: {response.text}")

def update_policy(state):
    if state.policy_id:
        put_url = f"{BASE_URL}/{state.policy_id}"
        sanitized_data = {
            key: (float(value) if isinstance(value, Decimal) else value)
            for key, value in state.collected_data.items()
        }

        put_data = {
            **sanitized_data,
            "policy_number": "POL123456",
            "quote_amount": 25000,
            "product_id": 1
        }
        response = requests.put(put_url, json=put_data, headers=HEADERS)
        if response.status_code == 200:
            print("Policy updated successfully.")
        else:
            print(f"Failed to update policy: {response.text}")
    else:
        print("No policy ID available. Policy creation might have failed.")

db_handler = DatabaseHandler(
    host="mysql-11613de7-insurance-crm.f.aivencloud.com",
    port=26392,
    user="avnadmin",
    password="AVNS_Bb3O_Yl-biwqrry1i9k",
    database="insurance_crm",
)

document_handler = DocumentHandler()
documents = document_handler.load_documents("insurance-dataset.docx")

vectorstore_handler = VectorStoreHandler()
vectorstore = vectorstore_handler.load_or_create_vectorstore(documents)

chatbot = Chatbot(vectorstore)

state = ConversationState()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    if "confirm" in user_input.lower():
        state.phase = 'customer'
        state.missing_fields = CUSTOMER_FIELDS.copy()
        return jsonify({"response": "To create a policy, I'll need some information. Let's start with your CNIC number in the format XXXXX-XXXXXXX-X."})
    
    elif state.phase == 'customer':
        if 'cnic' not in state.collected_data:
            cnic = user_input
            
            # Validate the CNIC format
            if not re.match(r'^\d{5}-\d{7}-\d{1}$', cnic):
                prompt = "The CNIC provided is invalid. Please provide a valid CNIC in the format XXXXX-XXXXXXX-X."
                return jsonify({"response": prompt})
            else:
                # CNIC is valid, proceed to check if it exists in the database
                if check_and_collect_customer(db_handler, state, cnic):
                    # CNIC is already in the database
                    customer_info = db_handler.get_customer_info(cnic)
                    state.collected_data.update(customer_info)
                    
                    # Move to device phase
                    state.phase = 'device'
                    state.missing_fields = ['device_serial_number']  # Only ask for serial number first
                    prompt = "I found your customer information in our system. Please provide your device serial number."
                    return jsonify({"response": prompt})
                else:
                    # CNIC is valid but not in the database, proceed to collect other fields
                    state.missing_fields = CUSTOMER_FIELDS.copy()
                    state.missing_fields.remove('cnic')  # Already have CNIC
                    state.collected_data['cnic'] = cnic
                    next_field = state.missing_fields[0]
                    prompt = f"Please provide your {next_field}."
                    return jsonify({"response": prompt})
        else:
            if state.missing_fields:
                next_field = state.missing_fields[0]
                state.collected_data[next_field] = user_input
                
                # Validate the input
                invalid_fields = db_handler.save_customer_info(state.collected_data)
                if next_field in invalid_fields:
                    prompt = f"The information provided for {next_field} is invalid. Please provide a valid {next_field}."
                    return jsonify({"response": prompt})
                else:
                    state.missing_fields.remove(next_field)
                    if state.missing_fields:
                        next_field = state.missing_fields[0]
                        prompt = f"Please provide your {next_field}."
                        return jsonify({"response": prompt})
                    else:
                        # All customer fields collected, move to device phase
                        state.phase = 'device'
                        state.missing_fields = DEVICE_FIELDS.copy()
                        next_field = state.missing_fields[0]
                        prompt = f"Please provide the {next_field} of your device."
                        return jsonify({"response": prompt})
            else:
                # If no missing fields, move to the device phase
                state.phase = 'device'
                state.missing_fields = DEVICE_FIELDS.copy()
                next_field = state.missing_fields[0]
                prompt = f"Please provide the {next_field} of your device."
                return jsonify({"response": prompt})
    
    elif state.phase == 'device':
        if 'device_serial_number' not in state.collected_data:
            # First ask for device serial number
            device_serial_number = user_input
            if check_and_collect_device(db_handler, state, device_serial_number):
                # Device exists in database
                device_info = db_handler.get_device_info(device_serial_number)
                state.collected_data.update(device_info)
                
                # Display all collected information
                response_message = "I found both your customer and device information in our system.\n\n"
                response_message += "Customer Information:\n"
                for field in CUSTOMER_FIELDS:
                    if field in state.collected_data:
                        response_message += f"{field.replace('_', ' ').title()}: {state.collected_data[field]}\n"
                
                response_message += "\nDevice Information:\n"
                for field in DEVICE_FIELDS:
                    if field in state.collected_data:
                        response_message += f"{field.replace('_', ' ').title()}: {state.collected_data[field]}\n"
                
                # Create policy with existing information
                create_policy(state)
                update_policy(state)
                response_message += "\nYour policy has been created successfully with the existing information!"
                
                return jsonify({"response": response_message})
            else:
                # Device not found, collect device info
                state.missing_fields = DEVICE_FIELDS.copy()
                state.missing_fields.remove('device_serial_number')  # Already have serial number
                state.collected_data['device_serial_number'] = device_serial_number
                next_field = state.missing_fields[0]
                prompt = f"Please provide the {next_field} of your device."
                return jsonify({"response": prompt})
        else:
            if state.missing_fields:
                next_field = state.missing_fields[0]
                state.collected_data[next_field] = user_input
                
                # Validate the input
                invalid_fields = db_handler.save_device_info(state.collected_data)
                if next_field in invalid_fields:
                    prompt = f"The information provided for {next_field} is invalid. Please provide a valid {next_field}."
                    return jsonify({"response": prompt})
                else:
                    state.missing_fields.remove(next_field)
                    if state.missing_fields:
                        next_field = state.missing_fields[0]
                        prompt = f"Please provide the {next_field} of your device."
                        return jsonify({"response": prompt})
                    else:
                        # All fields collected, proceed to create policy
                        create_policy(state)
                        update_policy(state)
                        
                        # Prepare the response with collected data
                        response_message = "Thank you! Your policy has been created successfully.\n\nHere are the details you provided:\n"
                        for key, value in state.collected_data.items():
                            response_message += f"{key.replace('_', ' ').title()}: {value}\n"
                        
                        return jsonify({"response": response_message})
            else:
                # If no missing fields, return the final response
                response_message = "Thank you! Your policy has been created successfully.\n\nHere are the details you provided:\n"
                for key, value in state.collected_data.items():
                    response_message += f"{key.replace('_', ' ').title()}: {value}\n"
                
                return jsonify({"response": response_message})
    
    else:
        response = chatbot.generate_response(user_input, documents)
        return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)