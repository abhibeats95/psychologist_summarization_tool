
import xml.etree.ElementTree as ET

import os
import weaviate
import json
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from typing import Any, List, Tuple, Optional, Dict
from langchain_community.vectorstores import weaviate as lc_weaviate
from typing import List, Tuple, Optional, Dict, Any
import os
import weaviate
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate


#######################
class_name='' # define a class name for vector db
#######################


def extract_text_from_xml_summary(xml_string, tag):
    # Find the start of the opening tag
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    
    # Find the indices for the start and end of the content
    start_index = xml_string.find(start_tag) + len(start_tag)
    end_index = xml_string.find(end_tag)
    
    # Extract the content between the tags
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return xml_string[start_index:end_index]
    else:
        return "Tag not found or malformed XML"
    


def _read_json(file_path):
    # Open the JSON file
    with open(file_path, 'r') as f:
        # Load the JSON data
        data = json.load(f)
    return data

def extract_elements_after_overview(pdf_elements):
    """
    Extracts all elements from a list after a specific title ('Overview').
    
    Parameters:
        pdf_elements (list of dict): The list containing dictionary elements of a PDF document.
    
    Returns:
        dict: A dictionary of elements after the 'Overview' title. 
              If 'Overview' is not found, returns the entire list as a dictionary.
    """
    # Flag to start collecting elements after 'Overview' is found
    collect = False
    # Dictionary to store the elements after 'Overview'
    elements_dict = []
    # Iterate over each element in the PDF elements list
    for idx, ele in enumerate(pdf_elements):
        # Check if we should start collecting elements
        if ele['type'] == 'Title' and ele['text'] == 'Overview':
            collect = True
            continue  # Skip adding the 'Overview' title itself
        # If we are collecting, add elements to the dictionary
        if collect:
          
            elements_dict.append(ele)
    
    # If 'Overview' was found and elements were collected, return them
    if collect:
        return elements_dict
    else:
        # If 'Overview' was not found, return all elements as a dictionary
        return pdf_elements


def parse_plan_w_bullet(result_action_plan):
    # Parse the XML string
    root = ET.fromstring(result_action_plan)

    action_plan=[]

    for idx,item in enumerate(root.findall('item')):
        # Write item content with a bullet point
        action_plan.append(f"({idx+1}) {item.text}")

    return '\n'.join(action_plan)



def format_elements(result_dict):
    """
    Formats elements from the dictionary based on their type, excluding certain tags,
    and applying HTML-like formatting. Titles, list items, narrative texts, and tables
    are formatted distinctly, while headers are excluded.

    Parameters:
        result_dict (dict): Dictionary of elements where each value is an element dictionary.

    Returns:
        str: A formatted string of the collected elements.
    """
    collect_text = []  # List to hold the collected and formatted text
    tags_to_exclude=['© NICE 2023. All rights reserved. Subject to Notice of rights (https://www.nice.org.uk/terms-and- conditions#notice-of-rights).']


    # Iterate over each element in the dictionary
    for ele in result_dict:
        # Check if the text should be excluded
        """if ele['text'] in tags_to_exclude:
            continue

        # Apply formatting based on the type of element
        if ele['type'] == 'Title':
            collect_text.append(f"<Title>{ele['text']}</Title>")
        elif ele['type'] == 'ListItem':
            collect_text.append(f"<bullet>{ele['text']}</bullet>")
        elif ele['type'] == 'NarrativeText':
            collect_text.append(f"<p>{ele['text']}</p>")
        elif ele['type'] == 'Table':
            collect_text.append(f"<table>{ele['text']}</table>")
        elif ele['type'] not in ['Header']:
            # Add other types of text directly
            collect_text.append(ele['text'])"""

        if ele['text'] in tags_to_exclude:
            continue
        
        # Apply formatting based on the type of element
        if ele['type'] == 'Title':
            collect_text.append(f"<Title>{ele['text'].strip()}</Title>")
        elif ele['type'] == 'Table':
            collect_text.append(f"<table>{ele['text'].strip()}</table>")
        elif ele['type'] not in ['Header']:
            # Add other types of text directly
            collect_text.append(ele['text'].strip())

    # Join all collected text into a single string with new lines between elements
    return " ".join(collect_text)





def clean_pdf_elements(elements, metadata, splitter='recursive', chunk_size=1000, chunk_overlap=200):
    """
    Clean PDF elements, format them, and split the text using a specified method.

    Parameters:
        elements (list): List of PDF elements.
        metadata (str): Metadata associated with the PDF.
        splitter (str, optional): Method to split the text. Defaults to 'recursive'.
        chunk_size (int, optional): Size of each chunk for splitting. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.

    Returns:
        list: List of documents after splitting the text.
        
    Raises:
        ValueError: If the specified splitter is not supported.
    """
    result_dict = extract_elements_after_overview(elements)
    formatted_text = format_elements(result_dict)
    
    if splitter == 'recursive':
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.create_documents([formatted_text], [{'title': metadata}])
        return docs
    else:
        raise ValueError("Unsupported text splitter. Please use 'recursive'.")

def create_chunks(processed_data_folder):
    """
    Create text chunks from processed PDF data.

    Parameters:
        processed_data_folder (str): Path to the folder containing processed PDF data.

    Returns:
        list: List of text chunks.
    """
    json_pdf_files = os.listdir(processed_data_folder)
    chunks = []
    for json_data in json_pdf_files:
        json_path = os.path.join(processed_data_folder, json_data)
        pdf_elements = _read_json(json_path)
        file_name_meta = os.path.splitext(os.path.basename(json_path))[0]

        result_dict = extract_elements_after_overview(pdf_elements)
        formatted_text = format_elements(result_dict)
        docs = clean_pdf_elements(elements=pdf_elements, metadata=file_name_meta, splitter='recursive')

        chunks.extend(docs)
    
    return chunks


def extract_text_from_xml(xml_content):
    # Parse the XML data
    root = ET.fromstring(xml_content)
    
    # Extract text from each element and filter out None values
    extracted_texts = [elem.text for elem in root.iter() if elem.text is not None]

    with open("action_plan.txt", "w") as file:
        for text in extracted_texts:
            if text is not None:
                file.write(text + '\n')

        print('Generated plan is saved in the root directory')
    
    return extracted_texts
# Using the function



def rag_custom_search(client: Any, 
                      query: str,
                      symptoms_info: Optional[str] = None, 
                      top_k: int = 5) -> Tuple[List[Any], Optional[Dict[str, Any]]]:
    """
    Executes a custom search on a vector database using a given query and optional symptoms information.

    Args:
    client (Any): The client object used to interact with the vector database.
    query (str): The main text query for document retrieval.
    symptoms_info (Optional[str]): Additional symptoms information provided as a string, optional.
    top_k (int): The number of top results to retrieve.

    Returns:
    Tuple[List[Any], Optional[Dict[str, Any]]]: A tuple containing the list of documents and an optional dictionary.
    """

    # Initialize the vector store with specific configuration for Weaviate client
    vectorstore = lc_weaviate.Weaviate(client, class_name, text_key="content", attributes=['content', 'title'])

    # Convert symptoms_info into keywords, if provided
    symptoms_keywords = []
    if symptoms_info:
        # Splitting the string into words and converting them to lowercase
        symptoms_keywords = [key.lower() for key in symptoms_info.split(' ')]

    # Construct the filter query based on symptoms keywords
    filter_operands = []
    if symptoms_keywords:
        # Adding a condition to filter documents containing any of the symptoms keywords in their title
        filter_operands.append({
            "path": ["title"],
            "operator": "ContainsAny",
            "valueTextArray": symptoms_keywords
        })

    # Building the final filter based on accumulated operands
    if filter_operands:
        # Use 'And' operator if multiple operands exist, otherwise use the single operand
        filter = {"operator": "And", "operands": filter_operands} if len(filter_operands) > 1 else filter_operands[0]
    else:
        # No filter applied if no conditions are met
        filter = None
        print('No filter is applied while searching documents')

    # Prepare search arguments for the vector retriever
    search_args = {
        "where_filter": filter if filter else {},
        "additional": ["distance"],  # Include distance in the results
        "k": top_k  # Number of documents to retrieve
    }

    # Retrieve documents using the prepared search arguments
    retriever = vectorstore.as_retriever(search_kwargs=search_args)
    docs = retriever.get_relevant_documents(query)

    return docs

def generate_summary(llm, content, template):
    """
    Generates a summary and primary diagnosis from given content using a template-based approach with an LLM.

    Args:
    llm: A language model client configured to interact with a language model.
    content (str): The text content from which to generate the summary.
    template (Template): A template object that formats the input according to the requirements of the language model.

    Returns:
    tuple: A tuple containing the generated summary and the primary diagnosis extracted from the model's output.
    """

    # Create a system message prompt from the given template
    system_message_summary_prompt = SystemMessagePromptTemplate.from_template(template)

    # Generate a chat prompt template based on the system message prompt
    summary_generator_prompt = ChatPromptTemplate.from_messages([system_message_summary_prompt])

    # Initialize an output parser to handle the model's XML output
    output_parser = StrOutputParser()

    # Create a pipeline by combining the prompt template, language model, and output parser
    summary_generator = summary_generator_prompt | llm | output_parser

    # Invoke the summary generator with the provided content
    result = summary_generator.invoke({'content': content})

    # Extract the summary text from the XML formatted result
    generated_summary = extract_text_from_xml_summary(result, "summary")

    # Extract the primary diagnosis from the XML formatted result
    patient_disorder = extract_text_from_xml_summary(result, "primary_diagnosis")
    
    # Return the summary and primary diagnosis
    return generated_summary, patient_disorder


def gen_action_plan_w_rag(llm: Any,
                          summary: str,
                          symptoms_info: Dict[str, Any],
                          client: weaviate.Client,
                          system_temp: str,
                          human_temp: str,
                          top_k_docs: int = 4) -> str:
    """
    Generates an action plan with RAG (Retrieve and Generate) model.

    Args:
    llm (Any): The language model used for generating the action plan.
    summary (str): The summary of the medical report.
    symptoms_info (Dict[str, Any]): Information about the patient's symptoms.
    client (Client): The Weaviate client for accessing the database.
    system_temp (float): System temperature.
    human_temp (float): Human temperature.
    top_k_docs (int, optional): Number of top documents to retrieve. Defaults to 4.

    Returns:
    Dict[str, Any]: The generated action plan.
    """
    # Retrieve context using RAG custom search
    retrived_context = rag_custom_search(client, query=summary,
                                          symptoms_info=symptoms_info, top_k=top_k_docs)

    # Prepare message prompts
    sys_msg = SystemMessagePromptTemplate.from_template(system_temp)
    human_msg = HumanMessagePromptTemplate.from_template(human_temp)
    prompt = ChatPromptTemplate.from_messages([sys_msg, human_msg])

    # Set up output parser
    output_parser = StrOutputParser()

    # Combine prompts, language model, and output parser into an action plan
    action_plan = prompt | llm | output_parser

    # Invoke the action plan with context and summary
    result_action_plan = action_plan.invoke({'summary': summary, 'context': retrived_context})

    # Parse the action plan with bullet points
    #result_action_plan = parse_plan_w_bullet(result_action_plan)

    return result_action_plan




def gen_action_plan_wo_rag(llm: Any,
                          summary: str,
                          symptoms_info: Dict[str, Any],
                          system_temp: str,
                          human_temp: str) -> str:
    """
    Generates an action plan with RAG (Retrieve and Generate) model.

    Args:
    llm (Any): The language model used for generating the action plan.
    summary (str): The summary of the medical report.
    symptoms_info (Dict[str, Any]): Information about the patient's symptoms.
    client (Client): The Weaviate client for accessing the database.
    system_temp (float): System temperature.
    human_temp (float): Human temperature.
    top_k_docs (int, optional): Number of top documents to retrieve. Defaults to 4.

    Returns:
    Dict[str, Any]: The generated action plan.
    """
    # Retrieve context using RAG custom search
   
    # Prepare message prompts
    sys_msg = SystemMessagePromptTemplate.from_template(system_temp)
    human_msg = HumanMessagePromptTemplate.from_template(human_temp)
    prompt = ChatPromptTemplate.from_messages([sys_msg, human_msg])

    # Set up output parser
    output_parser = StrOutputParser()

    # Combine prompts, language model, and output parser into an action plan
    action_plan = prompt | llm | output_parser

    # Invoke the action plan with context and summary
    result_action_plan = action_plan.invoke({'summary': summary})

    # Parse the action plan with bullet points
    #result_action_plan = parse_plan_w_bullet(result_action_plan)

    return result_action_plan


import json

def format_json_recursively(data, indent=0, use_rag=False):
    formatted_text = ""
    if isinstance(data, dict):
        for key, value in data.items():
            formatted_text += f"{' ' * indent}##{key}:\n{format_json_recursively(value, indent + 2)}"
    elif isinstance(data, list):
        for item in data:
            formatted_text += f"{' ' * indent}- {item}\n"
    else:
        formatted_text += f"{' ' * indent}{data}\n"

    with open(f"OUTPUT\\action_plan_use_rag_{use_rag}.txt", 'w') as file:
        file.write(formatted_text)
    return formatted_text



        

