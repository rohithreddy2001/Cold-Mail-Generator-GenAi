import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
# from dotenv import load_dotenv
# load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        
        ### INSTRUCTION:
        You are an expert at extracting structured data from unstructured career-page text.
        
        Your task:
        - Identify all distinct job postings from the provided text.
        - Return them strictly as a **valid JSON list**.
        - Each job posting must be an object with the following keys:
          - "role": string
          - "experience": string (e.g., "2+ years" or "Not specified")
          - "skills": string (comma-separated list, or "Not specified")
          - "description": short plain-text summary of the role
        
        Rules:
        - Do NOT include any explanation, commentary, headings, or preamble.
        - If multiple jobs are present, return all in a JSON list.
        - If a field is missing in the source text, fill it with "Not specified".
        
        ### OUTPUT:
        Return ONLY the JSON list of job postings.
        """

        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
        
            ### INSTRUCTION:
            You are Rohith Reddy Rudraiah Gari, a Data Analyst skilled in Python, SQL, Power BI, and data-driven solutions, with experience in:
            - AtliQ Mart supply-chain analytics (improved delivery KPI tracking with Power BI)
            - GoodCabs transportation insights
            - Sports analytics dashboards
            - A Gen-AI powered application for automated email drafting.
        
            Your task is to write ONE professional job-application email for the above role.
        
            STRICT RULES:
            - DO NOT include, restate, or display the job-description text inside the email. Use it ONLY as context to tailor the message.
            - Always use the provided job title EXACTLY as given in the job description when writing the Subject line and the opening sentence of the email.
            - Select up to 3 most relevant items from {link_list}, and mention them briefly with a one-line result/outcome **without repeating your portfolio link after each project**.
            - Include your portfolio link **only once, near the end of the email**, to invite the recruiter to learn more about your work.
            - Start the email with a **Subject line** in the format:
              Subject: [A clear, short subject line that reflects applying for the specific job role]
            - Then write the email body (120–150 words) in a polite, confident, and professional tone.
            - Output ONLY ONE email — no variations, no extra notes.
            - End with a courteous closing and an invitation for further discussion.
        
            Remember you are Rohith Reddy Rudraiah Gari.
            Do not provide a preamble or explanation.
        
            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))