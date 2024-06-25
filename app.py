import streamlit as st
import os
from langchain_openai import ChatOpenAI
from utils import extract_text_from_xml, generate_summary,gen_action_plan_wo_rag,gen_action_plan_w_rag,format_json_recursively
from prompts.prompts import (summary_template,
                            action_plan_worag_human_template,
                            action_plan_worag_systemt_template,
                            action_plan_wrag_human_template,action_plan_wrag_systemt_template,
                            further_action_system_template,further_action_human_template)
                            
import weaviate
import json_repair
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.agents import AgentExecutor
# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function


# Load environment variables from a .env file into the environment
load_dotenv()

# Initialize the OpenAI language model client, specifying to use 'gpt-4'
llm = ChatOpenAI(model='gpt-4')  # Specified model due to issues with 'gpt-4 turbo' parsing

# Boolean flag to indicate if RAG (Retrieval-Augmented Generation) context should be used
use_rag_context = False

# Initialize the Weaviate client
client = weaviate.Client(
        url=os.getenv('WEAVIATE_URL'),  # Retrieve the Weaviate endpoint URL from environment variables
        auth_client_secret=weaviate.auth.AuthApiKey(api_key=os.getenv('WEAVIATE_API_KEY')),  # Use API key for authentication
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY')  # Provide OpenAI API key for additional authorization
        }
    )


st.set_page_config(layout="wide")
################################################### Funtion schema for funtion calling ##############################################
class SaveReferralLetterInput(BaseModel):
    genereted_referral_letter: str = Field(description="Name of the patient")
  
def save_generated_referral_letter(referral_letter):
    st.write(f"Referral Letter: This is the content of referral letter for {referral_letter}")


save_referral_letter = StructuredTool.from_function(
    func=save_generated_referral_letter,
    name="save_referral_letter",
    description="useful for saveing the referral letter when referral letter is genereted based on the action plan of the patient",
    args_schema=SaveReferralLetterInput,
    return_direct=False,
)

class SavePrescriptionInput(BaseModel):
    prescription_letter: str = Field(description="Name of the patient")

def save_generated_prescription_letter(prescription_letter):
    st.write(
        f"Prescription Letter: This is the content of prescription letter for {prescription_letter}"
    )

save_prescription_letter = StructuredTool.from_function(
    func=save_generated_prescription_letter,
    name="save_presecription_letter",
    description="useful for saving the presecription genereted based on the action plan of the patient",
    args_schema=SaveReferralLetterInput,
    return_direct=False,
)

################################################### Funtion schema for funtion calling ##############################################


def load_file_content(file_name):
    # Reads text content from selected file
    f = open(os.path.join("input_transcripts", file_name), "r")
    return f.read()


def generate_summary_and_plan():
    # TODO : 1# Replace this with your actual logic to generate summary and action plan
    summary,disorder=generate_summary(llm,content=file_content, template=summary_template)

    if use_rag_context:
        generate_plan=gen_action_plan_w_rag(llm,summary,symptoms_info=disorder,client=client,system_temp=action_plan_wrag_systemt_template,human_temp=action_plan_wrag_human_template)
    else:
        generate_plan=gen_action_plan_wo_rag(llm,summary,symptoms_info=disorder,system_temp=action_plan_worag_systemt_template,human_temp=action_plan_worag_human_template)
    
    
    generate_plan = json_repair.loads(generate_plan)  # Load JSON string into Python dictionary
    # Call the function and write to a file
    generate_plan = format_json_recursively(generate_plan, use_rag=use_rag_context)
    #generate_plan = "\n".join([f"- ({item})" for item in generate_plan if item and item.strip()])
    st.session_state.summary = summary
    st.session_state.action_plan = generate_plan


def save_summary_and_plan(summary, action_plan):
    st.session_state.summary = summary
    st.session_state.action_plan = action_plan


def generate_further_actions(action_plan):
    """
    Determines further actions based on a patient's action plan and generates appropriate letters 
    (either prescription or referral), saving them using the specified tool.

    Args:
    action_plan (str): The action plan for the patient.

    Returns:
    tuple: A tuple containing the name of the tool used and the letter generated.
    """

    

    # Construct the prompt with placeholders for system and human message templates
    further_action_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(further_action_system_template),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template(input_variables=["action_plan"], template=further_action_human_template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # List of tools available for saving letters
    tools = [save_prescription_letter, save_referral_letter]

    # Create and configure an agent to handle the action plan with specified tools
    agent = create_openai_tools_agent(llm, tools, further_action_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Invoke the agent with the action plan and capture the response
    response = agent_executor.invoke({"action_plan": action_plan})

    # Extract tool name and generated letter from the response JSON
    tool_name_used = json_repair.loads(response['output'])['tool_name']
    generated_letter = json_repair.loads(response['output'])['generated_letter']
    
    return tool_name_used, generated_letter


# Initialize session_state variables if they do not exist
if "selected_file" not in st.session_state:
    st.session_state.selected_file = ""
if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "summary" not in st.session_state:
    st.session_state.summary= ""
if "action_plan" not in st.session_state:
    st.session_state.action_plan = ""

# Layout: 3 Columns
col1, col2, col3 = st.columns(3)

# Column 1: Inputs
with col1:
    st.header("Inputs")
    # File selection
    files = os.listdir("input_transcripts")  # List files in the directory
    selected_file = st.selectbox("Select a file", files, index=0, key="file_select_box")
    if selected_file != st.session_state.selected_file:
        st.session_state.selected_file = selected_file
        st.session_state.file_content = load_file_content(selected_file)
    # Text area for file content
    file_content = st.text_area(
        "File Content",
        value=st.session_state.file_content,
        height=300,
        key="file_content_text",
    )
    # Generate button
    if st.button("Generate"):
        st.session_state.file_content = (
            file_content  # Updating session state with edited value
        )
        generate_summary_and_plan()

# Column 2: Summary and Action Plan Outputs
with col2:
    st.header("Summary & Action Plan Output")
    with st.form(key="summary_action_plan_form"):
        # Text area for summary
        summary = st.text_area(
            "Summary", value=st.session_state.summary, height=150, key="summary_text"
        )
        # Text area for action plan
        action_plan = st.text_area(
            "Action Plan",
            value=st.session_state.action_plan,
            height=150,
            key="action_plan_text",
        )
        save_outputs = st.form_submit_button("Save & Generate Letters")
        # Save button
        if save_outputs:
            save_summary_and_plan(summary, action_plan)

# Column 3: Letters
with col3:
    st.header("Letters")
    ## TODO: Replace below with the return of generate_further_actions() which is based on the action plan 
    # and will run the appropriate function to generate the letters required
    tool_name_used, generated_letter=generate_further_actions(action_plan)
    st.write(f"Letter 1: This is the content of letter 1:\n{generated_letter}")
