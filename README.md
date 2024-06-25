
# LLM-based summary generation and action plan generation from the given conversation transcript of doctor and patient

## Project Overview
It is language model-based application designed to assist mental health clinicians by automating the generation of consultation summaries and action plans. The system adheres to the National Institute for Health and Care Excellence (NICE) guidelines, ensuring that the recommendations are accurate and effective.

## Features
- **Summary Generation**: Automatically generates a "summary" from provided consultation transcripts.
- **Action Plan Creation**: Produces a detailed "action plan" in bullet-point format based on the consultation details.
- **Guideline Adherence**: Utilizes "Retrieval Augmented Generation" (RAG) techniques to align the action plans with NICE guidelines, even if specific actions are not mentioned during the consultation.

## Note: Set use rag flag to true in streamlit_app.py so the relevant NICE guidelines will be provide to LLM while producing the action plan based on the provided summary.

 ```bash 
   use_rag_context = False
   ```

## Technical Setup
### Prerequisites
- An Anaconda environment is recommended for setting up the project.
- Python is required to run the application.

### Installation
1. Clone the repository to your local machine.
2. Set up a Conda environment:

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
To start the application, use the following command:
```bash
streamlit run app.py
```

## Please provide your LLM api key in .env file
Currently only openai supported

## Usage
Place the input transcripts into the `/input_transcripts` directory. Use the application interface to generate summaries and action plans.

## Outputs
The output folder contains the generated action plans

## For reference Samples action plans are in the output folder 