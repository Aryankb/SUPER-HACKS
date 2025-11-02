
from composio import Composio , ComposioToolSet , Action , App
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
import composio_langchain
import json
import re
import pytz
from selenium import webdriver
import chromadb
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import requests
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import zipfile
import io
import os
import re
import time
import google.generativeai as genai
from pathlib import Path
import operator
import subprocess
from dateutil.parser import parse
from tools.dynamo import db_client, s3_client
import tempfile
import numpy as np
import faiss
import uuid
from bs4 import BeautifulSoup
import boto3
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from tools.rag import rerank_results, get_embedding_from_query, list_project_files, load_file_index_and_metadata,search_single_file
from tools.llm import get_chain, generate_con 
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dateutil import parser
from typing import Dict
from datetime import timezone
from tools.specific_func import user_tool
# chroma_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
chroma_path="/app/chroma_storage"
chroma_client = chromadb.PersistentClient(path=chroma_path)

prompt = hub.pull("hwchase17/openai-functions-agent")

from dotenv import load_dotenv  

load_dotenv()


os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")



class PERSONAL_TOOLS:
    def __init__(self, api_key, kwargs, model, llm, user_id):
        self.api_key = api_key
        self.kwargs = kwargs
        self.model = model
        self.llm = llm
        self.user_id = user_id
        self.tool_set = ComposioToolSet(api_key=api_key)
    

    async def execute(self):
        action = self.kwargs["action"]
        del self.kwargs["action"]

        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")

    async def _1_STARTUP_VALIDATION(self,details,email):
        """
        Saves the startup details in the database and sends them to validators for approval.
        
        Args:
            details (markdown str): This will only contains 3 bullet points - "Startup name", "sector"  and the  "link" (complete aws s3 downloadable link including accesskeyid) of pdf report (STRICTLY TAKE ONLY THIS MUCH. NO EXTRA DETAILS.)
            email (str): The email address of the user submitting the startup details

        Returns:
            str: A success message indicating that the details were sent and stored successfully
        """
        validators = []
        url = "https://graph.facebook.com/v17.0/12334/messages"
        headers = {
            "Authorization": f"Bearer ",
            "Content-Type": "application/json"
        }
        for number in validators:
            data = {
                "messaging_product": "whatsapp",
                "to": number,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {
                        "text": details
                    },
                    "action": {
                        "buttons": [
                            {
                                "type": "reply",
                                "reply": {
                                    "id": "1",
                                    "title": "Assign Investors"
                                }
                            },
                            {
                                "type": "reply",
                                "reply": {
                                    "id": "2",
                                    "title": "Disqualify"
                                }
                            }
                        ]
                    }
                }
            }


            response = requests.post(url, headers=headers, json=data)
            print("Status Code:", response.status_code)
            print("Response:", response.json())
            msg_id=response.json().get("messages", [{}])[0].get("id", "")

            # Create a unique timestamp for ordering
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Store the details and msg_id in DynamoDB
            try:
                db_client.put_item(
                    TableName="_1_startups",
                    Item={
                        "whatsapp_id": {"S": msg_id},
                        "details": {"S": details},
                        "timestamp": {"S": timestamp},
                        "email": {"S": email}
                    }
                )
                print(f"Stored startup details in DynamoDB with whatsapp_id: {msg_id}")
                return "Stored startup details and sent to all validators"
            except Exception as e:
                print(f"Error storing details in DynamoDB: {e}")
                return "Failed to store startup details"


class IMAGE():
    def __init__(self, api_key, kwargs, model, llm,user_id):
        
        """
        Initializes the IMAGE class with API key and additional parameters.

        Parameters:
        - api_key (str): The API key for authentication.
        - kwargs (dict): Additional parameters for the class.
        """
        self.kwargs = kwargs
        self.model = model
        self.llm = llm
        self.user_id = user_id
        self.tool_set = ComposioToolSet(api_key=api_key)
    
    async def IMAGE_ANALYSER(self, image_path="s3://workflow-files-2709/your_image.jpg", query="What is in this image?"):
        """
        Analyzes an image based on a natural language query and generates a comprehensive report.
        
        Parameters:
        - image_path (str): The S3 path of the image to analyze. (e.g., "s3://workflow-files-2709/your_image.jpg")
        - query (str): The natural language query about the image. (e.g., "What is in this image?")

        Returns:
        - str: A dictionary containing the status and the analysis report.
        """
        # Download the image from S3
        s3_client = boto3.client('s3', region_name='us-east-1')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file_path = temp_file.name
        
        try:
            s3_parts = image_path.replace("s3://", "").split("/", 1)
            if len(s3_parts) == 2:
                bucket_name, key = s3_parts
                s3_client.head_object(Bucket=bucket_name, Key=key)
                s3_client.download_file(bucket_name, key, temp_file_path)
            else:
                raise ValueError("Invalid S3 path format for image_path")
            sample_file = genai.upload_file(path=temp_file_path, display_name="uploaded_image.jpg")
            # Prepare the prompt for the LLM
            prompt = f"""
                        You are an expert image analyst with advanced visual recognition capabilities.
                        Please analyze the provided image and extract all relevant information based on the user query: "{query}".
                        
                        Pay special attention to:
                        - Any visible text content (including small or partially obscured text)
                        - Data in tables, charts, or graphs (provide structured interpretation)
                        
                        For text extraction, maintain original formatting where possible.
                        If tables are present, return them in a structured JSON format with proper column headers and row values.
                        If charts/graphs are present, describe the data trends and key insights.
                        
                        Focus on being comprehensive yet precise in your analysis.
                        No introductory statements or concluding remarks needed - provide only the extracted information.
                        """
            
            # Send to LLM
            try:
                response = await generate_con(uid=self.user_id, model=self.model, inside=prompt, file=sample_file)
            except Exception as e:
                error_str = str(e)
                print(f"Error occurred: {error_str}")
                if "429" in error_str and "retry_delay" in error_str:
                    # Extract retry delay seconds
                    retry_seconds = 60  # Default fallback
                    # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    # if retry_match:
                    #     retry_seconds = int(retry_match.group(1))
                    
                    print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                    # Retry the request
                    response = await generate_con(uid=self.user_id, model=self.model, inside=prompt, file=sample_file)
                else:
                    raise e
            
            analysis_text = response.strip()
            return analysis_text
        except Exception as e:  
            print(f"Error processing image: {e}")
            return {"status": False, "error": str(e)}
    async def execute(self):
        """
        Executes the specified action dynamically.

        Parameters:
        - action (str): The name of the method to execute.

        Returns:
        - The result of the executed method or an error message if the method is not found.
        """
        action = self.kwargs["action"]
        del self.kwargs["action"]

        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")
            
           


class NOTION():
    def __init__(self, api_key, kwargs, model, llm,user_id):
        self.model = model
        self.llm = llm
        self.kwargs = kwargs
        self.user_id = user_id
        self.tool_set = ComposioToolSet(api_key=api_key)
        self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)

    async def NOTION_CREATE_PAGE_IN_PAGE(self, parent_page_link, page_title):
        """
        Creates a new page inside a parent page in Notion.

        Parameters:
        - parent_page_link (str): The URL of the parent Notion page. (e.g., "https://www.notion.so/ParentPage-1234567890abcdef1234567890abcdef")
        - page_title (str): The title of the new page to be created. (e.g., "New Page Title")

        Returns:
        - dict: A dictionary containing the status and the URL of the newly created page.
        """
        match = re.search(r'[^-]+$', parent_page_link)
        if match:
            print(match.group())
        id = match.group()[:8] + "-" + match.group()[8:12] + "-" + match.group()[12:16] + "-" + match.group()[16:20] + "-" + match.group()[20:]
        print(id)
        rret = self.tool_set.execute_action(
            action="NOTION_CREATE_NOTION_PAGE",
            params={"parent_id": id, "title": page_title},
        )
        return {
            "status": rret["successfull"],
            "new_page_url": rret["data"]["data"]["url"]
        }

    async def NOTION_ADD_ONE_CONTENT_BLOCK_IN_PAGE(self, parent_page_link, content_block, block_property="paragraph"):
        """
        Adds a single content block to a Notion page.

        Parameters:
        - parent_page_link (str): The URL of the parent Notion page. (e.g., "https://www.notion.so/ParentPage-1234567890abcdef1234567890abcdef")
        - content_block (str): The content to add to the page in MARKDOWN FORMAT. (e.g., "This is a new content block.")
        - block_property (str): The type of block to add. (e.g., "paragraph", "heading_1", "bulleted_list_item")

        Returns:
        - dict: A dictionary containing the status of the operation.
        """
        match = re.search(r'[^-]+$', parent_page_link)
        if match:
            print(match.group())
        id = match.group()[:8] + "-" + match.group()[8:12] + "-" + match.group()[12:16] + "-" + match.group()[16:20] + "-" + match.group()[20:]
        print(id)
        task = f"""
        parent_block_id : {id} 
        content : {content_block}
        block_property : {block_property}
        please format the above information in the required input format
        """
        tools = self.prompt_toolset.get_tools(actions=['NOTION_ADD_PAGE_CONTENT'])
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        result = agent_executor.invoke({"input": task})
        return {
            "status": result["output"],
        }
    async def NOTION_UPDATE_ROW_DATABASE(self, row_id="", **row_content):
        """
        Updates a row in a Notion database.
        Parameters:
        - row_id (str): The ID of the row/page to update. (e.g., "12345678-1234-1234-1234-1234567890ab")
        - **row_content: Key-value pairs representing the updated row content. (e.g., {"Task Name": "Complete project", "Due Date": "2023-10-01"})

        """

        try:
            row_details = self.tool_set.execute_action(
                action="NOTION_FETCH_ROW",
                params={"page_id": row_id}
            )["data"]["page_data"]
            if not row_details["archived"] and not row_details["in_trash"]:
                properties=row_details.get("properties", {})
                prop_nam_typ = [{properties[prop]["name"]: properties[prop]["type"]} for prop in properties]
                assignees= self.tool_set.execute_action(
            action="NOTION_LIST_USERS",
            params={},
        )["data"]["response_data"]["results"]
                task = f''' You are notion row updater agent. update the row in notion database with the following information:
                row_id : {row_id}
                properties : {prop_nam_typ}
                new content : {row_content}
                assignees : {assignees}
                please format the above information in the required input format
                '''
                tools = self.prompt_toolset.get_tools(actions=['NOTION_UPDATE_ROW_DATABASE'])
                agent = create_openai_functions_agent(self.llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                result = agent_executor.invoke({"input": task})
                return {
                    "status": result["output"],
                }
        except:
            return "row not found or row is archived/in trash."

    async def NOTION_INSERT_ROW_DATABASE(self, parent_page_link, database_name="", database_id="", **row_content):
        """
        Inserts a row into a Notion database.

        Parameters:
        - parent_page_link (str): The URL of the Notion page containing the database. (e.g., "https://www.notion.so/ParentPage-1234567890abcdef1234567890abcdef")
        - database_name (str): The title of the database in the Notion page. (e.g., "Tasks Database")
        - **row_content: Key-value pairs representing the row content to insert. (e.g., {"Task Name": "Complete project", "Due Date": "2023-10-01"})
        - database_id (str) (optional): The ID of the database to query. If provided, it will be used instead of the page link.

        IMPORTANT : If provided database_id , no need to give parent_page_link and database_name

        Returns:
        - dict: A dictionary containing the status of the operation.
        """
        if database_id:
            db_id=  database_id
        else:
            print(row_content, parent_page_link, database_name)
            db_id = None
            blocks=None
            if "-" in parent_page_link:
                match = re.search(r'[^-]+$', parent_page_link)
                if match:
                    print(match.group())
                id = match.group()[:8] + "-" + match.group()[8:12] + "-" + match.group()[12:16] + "-" + match.group()[16:20] + "-" + match.group()[20:]
                print(id)
                blocks = self.tool_set.execute_action(
                    action="NOTION_FETCH_NOTION_CHILD_BLOCK",
                    params={"block_id": id},
                )

                # Extracting database ID
                print(blocks)
                for block in blocks["data"]["block_child_data"]["results"]:
                    if block["type"] == "child_database" and block["child_database"]["title"] == database_name:
                        db_id = block["id"]
                        break
            else:
                db_id_match = re.search(r'notion\.so/([^/?]+)', parent_page_link)
                if db_id_match:
                    db_id = db_id_match.group(1).replace("-", "")
                    # Format the ID properly with dashes
                    db_id = db_id[:8] + "-" + db_id[8:12] + "-" + db_id[12:16] + "-" + db_id[16:20] + "-" + db_id[20:]


        
        # Getting properties of database (schema)
        properties = self.tool_set.execute_action(
            action="NOTION_FETCH_DATABASE",
            params={"database_id": db_id},
        )["data"]["properties"]
        print(type(properties))

        assignees= self.tool_set.execute_action(
            action="NOTION_LIST_USERS",
            params={},
        )["data"]["response_data"]["results"]

        prop_nam_typ = [{properties[prop]["name"]: properties[prop]["type"]} for prop in properties]

        task = f'''
        properties schema (type for each property):- {prop_nam_typ}\nSTRICTLY FOLLOW THE ABOVE SCHEMA AND EXTRACT INFORMATION FROM THE ROW CONTENT SUCH THAT IT FOLLOWS THE SCHEMA\n
        db id :- {db_id}\n
        row content to insert :- {row_content}\n
        List of users in notion workspace :- \n{assignees}\n\n
        if anything missing, then keep it empty
        '''
        print(task)
        tools = self.prompt_toolset.get_tools(actions=['NOTION_INSERT_ROW_DATABASE'])
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        result = agent_executor.invoke({"input": task})
        print(result)
        return {
            "status": result["output"],
        }

    async def NOTION_QUERY_DATABASE(self, parent_page_link="", database_name="", query="",database_id=""):
        """
        Queries a Notion database based on a natural language query, parent_page_link and database_name.
        IMPORTANT : If provided database_id , no need to give parent_page_link and database_name

        Parameters:
        - parent_page_link (str): The URL of the Notion page containing the database. (e.g., "https://www.notion.so/ParentPage-1234567890abcdef1234567890abcdef")
        - database_name (str): The title of the database in the Notion page. (e.g., "Tasks Database")
        - query (str): The natural language query to run on the database. (e.g., "Show me all tasks due this week")
        - database_id (str) (optional): The ID of the database to query. If provided, it will be used instead of fetching from the page link.

        Returns:
        - str : A structured natural language response containing relevant rows and columns from the database based on the query.
        """
        if database_id:
            db_id = database_id
        else:
            db_id = None
            blocks=None
            if "-" in parent_page_link:
                match = re.search(r'[^-]+$', parent_page_link)
                if match:
                    print(match.group())
                id = match.group()[:8] + "-" + match.group()[8:12] + "-" + match.group()[12:16] + "-" + match.group()[16:20] + "-" + match.group()[20:]
                print(id)
                blocks = self.tool_set.execute_action(
                    action="NOTION_FETCH_NOTION_CHILD_BLOCK",
                    params={"block_id": id},
                )

                # Extracting database ID
                print(blocks)
                for block in blocks["data"]["block_child_data"]["results"]:
                    if block["type"] == "child_database" and block["child_database"]["title"] == database_name:
                        db_id = block["id"]
                        break
            else:
                db_id_match = re.search(r'notion\.so/([^/?]+)', parent_page_link)
                if db_id_match:
                    db_id = db_id_match.group(1).replace("-", "")
                    # Format the ID properly with dashes
                    db_id = db_id[:8] + "-" + db_id[8:12] + "-" + db_id[12:16] + "-" + db_id[16:20] + "-" + db_id[20:]

        
        results=self.tool_set.execute_action(
                action="NOTION_QUERY_DATABASE",
                params={"database_id": db_id,"page_size":100},
            )
        
        next_cursor= results["data"]["response_data"].get("next_cursor", None)
        if next_cursor:
            while next_cursor:
                additional_results = self.tool_set.execute_action(
                    action="NOTION_QUERY_DATABASE",
                    params={"database_id": db_id, "page_size": 100, "start_cursor": next_cursor},
                )
                results["data"]["response_data"]["results"].extend(additional_results["data"]["response_data"]["results"])
                next_cursor = additional_results["data"]["response_data"].get("next_cursor", None)
        ROWS=[]
        for row in results["data"]["response_data"]["results"]:
            if not row["archived"] and not row["in_trash"]:
                properties = row["properties"]
                row_id= row["id"]
                row_data = {"id": row_id,"properties": properties}
                ROWS.append(row_data)
        task = f"""
According to the user's query and the data from the Notion database, please provide a relevant and structured response in natural language with all the rows and columns that are relevant to the query, remove unnecessary information.\n
The user's query is: "{query}"\n
The data from the Notion database is as follows:\n
{ROWS}
"""
        try:
            response = await generate_con(uid=self.user_id, model=self.model, inside=task)
        except Exception as e:
            error_str = str(e)
            print(f"Error occurred: {error_str}")
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = await generate_con(uid=self.user_id, model=self.model, inside=task)
            else:
                raise e
        return response
    

    


    async def NOTION_GET_CHILD_BLOCKS_FOR_ROW_OR_PAGE(self, page_id):
        """
        Fetches all child blocks of a specific row or page in the Notion database. If the blocks have own children, then use this function again to get children giving specific block ID.

        Parameters:
        - page_id (str): The ID of the page or row to fetch child blocks from. It is the last part of page link, (e.g., "https://www.notion.so/ParentPage-1234567890abcdef1234567890abcdef" have page_id as "1234567890abcdef1234567890abcdef")

        Returns:
        - dict: The dictionary containing the final combined rich text
        """
        final_content=""
        response = self.tool_set.execute_action(
            action="NOTION_GET_BLOCKS",
            params={"block_id": page_id}
        )["data"]["block_child_data"]
        next_cursor= response.get("next_cursor", None)
        if next_cursor:
            while next_cursor:
                additional_results = self.tool_set.execute_action(
                    action="NOTION_QUERY_DATABASE",
                    params={"block_id": page_id, "start_cursor": next_cursor},
                )
                response["results"].extend(additional_results["data"]["block_child_data"]["results"])
                next_cursor = additional_results["data"]["block_child_data"].get("next_cursor", None)
        for i in response["results"]:
            if not i["archived"] and not i["in_trash"]:
                if "has_children" in i and i["has_children"]:
                    final_content += f"\n\nBlock ID: {i['id']}\nhas children: {i['has_children']}\n"
                inner_list=i[i["type"]]["rich_text"]
                inner_content= ""
                if inner_list:
                    for inner in inner_list:
                        inner_content+=str(inner[inner["type"]])+"\n"
                final_content += inner_content + "\n"
        final_content+="ROW DETAILS:\n"
        # Fetching row details
        properties={}
        try:
            row_details = self.tool_set.execute_action(
                action="NOTION_FETCH_ROW",
                params={"page_id": page_id}
            )["data"]["page_data"]
            if not row_details["archived"] and not row_details["in_trash"]:
                properties=row_details.get("properties", {})
                final_content+=str(properties)+"\n"
        except:
            final_content += "No row details found or row is archived/in trash.\n"
               
        return {
            "content":final_content
        }

    async def NOTION_SEARCH_BY_PAGE_OR_ROW_OR_DB_NAME(self,name):
        """ Searches for pages, rows, or databases or tables in Notion based on a given name. STRICTLY Use this when the page link or database ID is not available
        When user have not given the page link or database ID, then this function is used to search for the page, row, or database by name.
example: "Show me all tasks of aryan from Sigmoyd table" -->  Here "Sigmoyd" is the name of the database to search for.

        Parameters:
        - name (str): The part of name of the page, row, or database to search for.
        Returns:
        - dict: The search results containing matching pages, rows, or databases.
        """
        response= self.tool_set.execute_action(
            action="NOTION_FETCH_DATA",
            params={"query": name, "page_size": 100,'get_all': True}
        )["data"]["values"]

        return {
            "results": response
        }

    async def execute(self):
       
        action = self.kwargs["action"]
        del self.kwargs["action"]

        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")
        

class API_TOOL:
    def __init__(self,api_key,kwargs,model, llm, user_id):
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.kwargs = kwargs
        self.user_id = user_id
    
    def clean_llm_json_response(self,raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)
            
    def API_TOOL(self,api:str,input:str,input_param_format:dict,request_type:str):
        '''Makes a dynamic API call (GET or POST) by filling the provided input values into the specified parameter format.

This function enables users to define the exact structure of the API request body (including nested keys),
and automatically fills in the appropriate values based on user-provided input.

Parameters:
----------
api : str
    The URL of the target API endpoint.

input_values : dict
    A dictionary of actual values to be substituted into the request payload.
    The keys in this dictionary should correspond to the descriptions defined in the `input_param_format`.

input_param_format : dict
    A dictionary that defines the exact key structure expected by the target API.
    - The key names must exactly match the keys the API expects (including nesting).
    - The values should be short descriptions of what each key represents, which will be used to match against `input_values`.

    Example:
    --------
    input_param_format = {
        "aayush": {
            "name": "name of the person",
            "email": "email id of the person"
        }
    }

    input_values = {
        "name of the person": "John Doe",
        "email id of the person": "john@example.com"
    }

    Final payload sent to the API:
    {
        "aayush": {
            "name": "John Doe",
            "email": "john@example.com"
        }
    }

request_type : str
    The HTTP method to use for the API call. Must be either 'GET' or 'POST'.

Returns:
-------
response : requests.Response or str
    The response object returned by the API call, or an error message if the request_type is invalid.
'''
        task=f'''you are an API tool that makes API input : {input} in format {input_param_format}.
            expected output is a json with key as in {input_param_format} and values as the input for the API call.
            no preamble , no postamble , no explainantion just the json '''
        response=self.llm.invoke(task)
        print("llm response :",response.content)
        params=self.clean_llm_json_response(response.content)
        print("params :",params)
        try:
            request_type_lower = request_type.lower()

            if request_type_lower == 'post':
                response = requests.post(url=api, json=params, timeout=10)
            elif request_type_lower == 'get':
                response = requests.get(url=api, params=params, timeout=10)
            else:
                return "❌ Invalid request type. Please use 'POST' or 'GET'."

            response.raise_for_status()  # Raises HTTPError for bad HTTP responses (4xx/5xx)
            return response

        except requests.exceptions.Timeout:
            return "❌ Request timed out. The server might be too slow or unreachable."

        except requests.exceptions.ConnectionError:
            return "❌ Failed to connect to the server. Check the API URL or your internet connection."

        except requests.exceptions.HTTPError as http_err:
            return f"❌ HTTP error occurred: {http_err}. Response content: {response.text}"

        except requests.exceptions.JSONDecodeError:
            return "❌ Failed to parse JSON in the response. The API might not be returning valid JSON."

        except requests.exceptions.RequestException as e:
            return f"❌ An unexpected error occurred during the API request: {e}"

        except Exception as e:
            return f"❌ A general error occurred: {e}"




    async def execute(self):
        action = self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")
            


class CSV:
    async def execute(self):
        """
        Executes the specified action dynamically.

        Parameters:
        - action (str): The name of the method to execute.

        Returns:
        - The result of the executed method or an error message if the method is not found.
        """
        action = self.kwargs["action"]
        del self.kwargs["action"]

        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")
    
    def __init__(self, api_key, kwargs,model,llm,user_id):
        
        """
        Initializes the NOTION class with API key and additional parameters.

        Parameters:
        - api_key (str): The API key for authentication.
        - kwargs (dict): Additional parameters for the class.
        """
        self.kwargs = kwargs
        self.model=model
        self.llm=llm
        self.user_id = user_id
        self.tool_set = ComposioToolSet(api_key=api_key)
        self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)
    async def CSV_ANALYSER_AND_REPORT_MAKER(self,query: str, csv_path: str="name of one csv file's s3 path (amazon s3 key. ex: s3://workflow-files-2709/your_file.csv)"):
        """
            Analyzes a CSV file based on a natural language query and generates a comprehensive report.
            this function interprets the user's natural language query about the data, performs appropriate analysis using pandas and visualization 
            libraries, and generates a PDF report with the results.
            Parameters:
            ----------
            query : str
                Natural language query about the data (e.g., "Show me sales trends by month", 
                "What's the correlation between price and quantity?")
            csv_path : str
                Path to the CSV file : An S3 path in format "s3://bucket-name/key" 
                
            Returns:
            -------
                - "s3_path_of_pdf_analysis_report": S3 path to the generated PDF report
                - "answer_to_the_query": Text summary of the analysis results
"""
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = "workflow-files-2709"
        temp_file = None
        
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv' if not csv_path.endswith('.xlsx') else '.xlsx')
            temp_file_path = temp_file.name
            
            try:
                # If the csv_path is an S3 path (e.g., "s3://bucket/key"), parse and download
                if csv_path.startswith("s3://"):
                    s3_parts = csv_path.replace("s3://", "").split("/", 1)
                    if len(s3_parts) == 2:
                        bucket_name, key = s3_parts
                        s3_client.head_object(Bucket=bucket_name, Key=key)
                        s3_client.download_file(bucket_name, key, temp_file_path)
                        csv_path = temp_file_path
                    else:
                        raise ValueError("Invalid S3 path format for csv_path")
                else:
                    # Try to get the object from S3 using default bucket
                    s3_client.head_object(Bucket=bucket_name, Key=csv_path)
                    # Download the file from S3
                    s3_client.download_file(bucket_name, csv_path, temp_file_path)
                    # Update csv_path path to local path
                    csv_path = temp_file_path
            except Exception as e:
                # If error, assume csv_path is a local file path
                print(f"S3 download failed, using csv_path as local path: {e}")
            
            # Load the CSV with encoding error handling
            if csv_path.endswith('.xlsx'):
                df = pd.read_excel(csv_path)
            else:
                try:
                    df = pd.read_csv(csv_path)
                except UnicodeDecodeError:
                    # Try different encodings if UTF-8 fails
                    encodings_to_try = ['latin1', 'ISO-8859-1', 'cp1252']
                    for encoding in encodings_to_try:
                        try:
                            df = pd.read_csv(csv_path, encoding=encoding)
                            print(f"Successfully read file using {encoding} encoding")
                            break
                        except Exception as e:
                            print(f"Failed with encoding {encoding}: {e}")
                    else:
                        # If all encodings fail, try with error handling parameter
                        df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
            
            columns = df.columns.tolist()
            # Get sample data for each column
            sample_data = {col: df[col].dropna().unique()[:5].tolist() for col in columns}
            # Add necessary imports for enhanced functionality
            import matplotlib.pyplot as plt

            # Step 1: First, let's understand what the user is asking
            structured_query_prompt = f"""
            Given a CSV file with the following column names and sample values:

            Column names: {columns}
            Sample data: {sample_data}

            The user has asked this question: "{query}"

            Your task is to interpret this query and convert it into a clear, technical specification for data analysis.
            1. Identify what columns are relevant to this query
            2. Determine what type of analysis is needed (e.g., statistical summary, trends, comparisons)
            3. Specify if any data visualizations would be helpful
            4. Convert the user's query into a technical, precise question for a data analyst

            Provide only the rephrased technical query without any explanations.
            """

            # Get technical query interpretation
            try:
                tech_query_response = await generate_con(uid=self.user_id, model=self.model, inside=structured_query_prompt)
            except Exception as e:
                error_str = str(e)
                print(f"Error occurred: {error_str}")
                if "429" in error_str and "retry_delay" in error_str:
                    # Extract retry delay seconds
                    retry_seconds = 60  # Default fallback
                    # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    # if retry_match:
                    #     retry_seconds = int(retry_match.group(1))
                    
                    print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                    # Retry the request
                    tech_query_response = await generate_con(uid=self.user_id, model=self.model, inside=structured_query_prompt)
                else:
                    raise e
            technical_query = tech_query_response.strip()

            # Create a unique directory for temporary files
            temp_dir = tempfile.mkdtemp()
            print("Technical Query:", technical_query)
            # Construct the prompt for the LLM focused on analysis only
            prompt = f"""
            Given a CSV file with the following column names and sample values (note: these are just examples and not the complete dataset):
            
            Column names: {columns}
            Sample data: {sample_data}
            
            dont use sample data . instead use the full csv available in the csv_path : {csv_path}
            Write Python pandas, matplotlib/seaborn/plotly code to answer this analysis query: "{technical_query}"
            
    
            Your code MUST:
            0. absolutely Error free
            1. First read the complete CSV file from '{csv_path}'
            2. print the analysis in structured manner. it should also save the images generated by matplotlib and print the temp paths of that images.
            3. Convert to a numpy array before indexing always, in the following type of lines : plt.plot(df['month_number'], df[product], marker='o')
            4. don't create a function. instead just write code that can be run directly
            5. the print statements should be clear and informative, explaining the analysis results as well as image paths.
            save images in directory: {temp_dir}
            
            The code should be structured as follows:
            ```
            # import all necessary libraries
            
            ...
            # Read the complete dataset
            if csv_path.endswith('.xlsx'):
                df = pd.read_excel(csv_path)
            else:
                try:
                    df = pd.read_csv(csv_path)
                except UnicodeDecodeError:
                    # Try different encodings
                    encodings_to_try = ['latin1', 'ISO-8859-1', 'cp1252']
                    for encoding in encodings_to_try:
                        try:
                            df = pd.read_csv(csv_path, encoding=encoding)
                            break
                        except Exception:
                            pass
                    else:
                        df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
            
            # Your analysis code here
            
            ```
            
            The print statements should be a clear, informative , explaining the analysis results.
            Return only executable Python code with no explanations (comments are allowed).
            """
            
            # Send to LLM
            try:
                response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
            except Exception as e:
                error_str = str(e)
                print(f"Error occurred: {error_str}")
                if "429" in error_str and "retry_delay" in error_str:
                    # Extract retry delay seconds
                    retry_seconds = 60  # Default fallback
                    # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    # if retry_match:
                    #     retry_seconds = int(retry_match.group(1))
                    
                    print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                    # Retry the request
                    response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
                else:
                    raise e
            
            # Extract the generated code
            code = response
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            # Execute the code safely with output capturing
            
            # Prepare to capture output
            # output_buffer = io.StringIO()
            
            # local_vars = {"pd": pd, "csv_path": csv_path, "os": os, "plt": plt, "temp_dir": temp_dir}
            print("Executing code...",code)
            try:
                # Execute the code and capture all printed output
                # with redirect_stdout(output_buffer):
                    # Write code to a temporary file
                temp_code_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
                temp_code_file.write(f"""
{code}
                """)
                temp_code_file.close()

                # Run the code as a separate process
                result = subprocess.run(
                    f"python3 {temp_code_file.name}", 
                    shell=True, 
                    check=False,  # Don't raise exception on non-zero exit
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Get analysis text or error message
                if result.returncode == 0:
                    analysis_text = result.stdout
                    print("Code executed successfully.")
                    if not analysis_text.strip():
                        print("Warning: Empty stdout despite successful execution.")
                        # Try to read print statements directly from the file as fallback
                        with open(temp_code_file.name, 'r') as f:
                            file_content = f.read()
                            analysis_text = f"Code executed but produced no output.\nCode content: \n{file_content}"
                else:
                    print(f"Code execution failed with return code {result.returncode}")
                    analysis_text = result.stdout
                    analysis_text += f"\nError in analysis:\n{result.stderr}\n\nPartial output (if any):\n{result.stdout}"

                print("Analysis Text:", analysis_text)
                

                # Clean up the temp file
                os.remove(temp_code_file.name)
                # print("Code executed successfully.")
                # Get the printed output
                

                # print("Analysis Text:", analysis_text)
                
                # Find all image paths in the output
                # image_pattern = re.compile(r'(?:Image saved at|Saved plot to|Saved|saved)(?:\s+(?:to|at))?\s*:?\s*((?:/[^\s]+)?/[^\s]+\.(?:png|jpg|jpeg|pdf))')
                # image_matches = image_pattern.findall(analysis_text)
                # print("Image matches found:", image_matches)
                
                # # If no images were found by the pattern, look for full paths in the output
                # if not image_matches:
                full_path_pattern = re.compile(r'/tmp/[^\s]+(?:\.png|\.jpg|\.jpeg|\.pdf)')
                image_matches = full_path_pattern.findall(analysis_text)
                print("Additional image paths found:", image_matches)
            
                # Create PDF with analysis text and images
                pdf_path = os.path.join(temp_dir, "analysis_report.pdf")
                doc = SimpleDocTemplate(pdf_path, pagesize=letter)
                styles = getSampleStyleSheet()
                elements = []
                
                # Add text content to PDF, split by lines for better formatting
                for line in analysis_text.split('\n'):
                    if line.strip():  # Skip empty lines
                        # Check if this line contains an image path, if so skip it (we'll add the image directly)
                        if not any(img_path in line for img_path in image_matches):
                            elements.append(Paragraph(line, styles["Normal"]))
                            elements.append(Spacer(1, 6))  # Small spacing between lines
                
                # Add all found images to the PDF in order
                for img_path in image_matches:
                    print(f"Found image path: {img_path}")
                    if os.path.exists(img_path):
                        # Add a spacer before each image
                        elements.append(Spacer(1, 12))
                        # Scale image to fit page width (width=450 is good for letter size paper)
                        img = Image(img_path, width=450)
                        elements.append(img)
                        elements.append(Spacer(1, 12))
            except Exception as e:
                print(f"Error executing code: {e}")
                return f"Error analyzing data: {e}"
            try:
                # Build PDF with better image handling
                for i, element in enumerate(elements):
                # Check if element is an Image and adjust its size if needed
                    if isinstance(element, Image):
                        # Get the original image dimensions
                        img_width = element.imageWidth
                        img_height = element.imageHeight
                        
                        # Calculate the available space in the frame (with margins)
                        available_width = 450  # Slightly smaller than the page width
                        max_height = 600       # Limit height to avoid overflow
                        
                        # Calculate the scaling factor while preserving aspect ratio
                        if img_width > available_width:
                            scale_factor = available_width / img_width
                            new_width = available_width
                            new_height = img_height * scale_factor
                        
                        # If height is still too large, scale further
                        if new_height > max_height:
                            scale_factor = max_height / new_height
                            new_width = new_width * scale_factor
                            new_height = max_height
                        
                        # Update element dimensions
                        element.drawWidth = new_width
                        element.drawHeight = new_height
                
                # Build the PDF with our properly sized elements
                doc.build(elements)
                
                # Upload the PDF to S3
                s3_client = boto3.client('s3', region_name='us-east-1')
                output_filename = f"analysis_{os.path.basename(csv_path).replace('.', '_')}.pdf"
                s3_client.upload_file(pdf_path, bucket_name, output_filename)
                
                # Return S3 path and brief analysis summary
                pdf_s3_path = f"s3://{bucket_name}/{output_filename}"
                print("analysis text:", analysis_text)
                short_summary =  analysis_text
                # Create a download link for the PDF
                # Create a presigned URL for downloading the PDF from S3
                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': output_filename
                    },
                    ExpiresIn=604800  # URL expires in 1 week (7 days)
                )

                # The presigned URL can be used to download the file without authentication
                download_link = presigned_url
                to_ret={
                    "s3_path_of_pdf_analysis_report": pdf_s3_path,
                    "answer_to_the_query": short_summary,
                    "download_link":download_link
                }
                return to_ret

            except Exception as e:
                print(f"Error creating PDF: {e}")
                return f"Error analyzing data: {e}"
            # finally:
            #     # Clean up any image files
            #     for match in image_pattern.findall(output_buffer.getvalue()):
            #         if os.path.exists(match):
            #             try:
            #                 os.remove(match)
            #             except:
            #                 pass
                
        finally:
            # Clean up - delete temporary file if it exists
            if temp_file:
                temp_file.close()
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

    async def CSV_MODIFY(self,query: str, csv_path: str="name of one csv file's s3 path (amazon s3 key. ex: s3://workflow-files-2709/your_file.csv)"):
        """
            Modifies a CSV file using natural language query instructions.
            This method allows users to transform and manipulate CSV or Excel files through plain English
            commands. 
            Parameters:
            -----------
            query : str
                Natural language description of the modifications to perform on the CSV file.
                Examples: "Remove all rows with missing values", "Add a new column that multiplies price by quantity"
            csv_path : str
                Path to the CSV file to modify : S3 path in format "s3://bucket-name/path/to/file.csv"
            Returns:
            --------
            str
                Summary of modifications made and the S3 path where the modified file was saved
            
        """
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = "workflow-files-2709"
        temp_file = None
        
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv' if not csv_path.endswith('.xlsx') else '.xlsx')
            temp_file_path = temp_file.name
            
            # Download file from S3 or use local path
            key = None
            try:
                if csv_path.startswith("s3://"):
                    s3_parts = csv_path.replace("s3://", "").split("/", 1)
                    if len(s3_parts) == 2:
                        bucket_name, key = s3_parts
                        s3_client.head_object(Bucket=bucket_name, Key=key)
                        s3_client.download_file(bucket_name, key, temp_file_path)
                        csv_path = temp_file_path
                    else:
                        raise ValueError("Invalid S3 path format")
                else:
                    # Assume it's a key in the default bucket
                    key = csv_path
                    s3_client.head_object(Bucket=bucket_name, Key=key)
                    s3_client.download_file(bucket_name, key, temp_file_path)
                    csv_path = temp_file_path
            except Exception as e:
                print(f"S3 download failed, using as local path: {e}")
            
            # Load the CSV with encoding error handling
            if csv_path.endswith('.xlsx'):
                df = pd.read_excel(csv_path)
            else:
                try:
                    df = pd.read_csv(csv_path)
                except UnicodeDecodeError:
                    encodings_to_try = ['latin1', 'ISO-8859-1', 'cp1252']
                    for encoding in encodings_to_try:
                        try:
                            df = pd.read_csv(csv_path, encoding=encoding)
                            break
                        except Exception:
                            pass
                    else:
                        df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
            
            columns = df.columns.tolist()
            sample_data = {col: df[col].dropna().unique()[:5].tolist() for col in columns}
            
            # Create updated file path
            if key:
                original_filename = os.path.basename(key)
                filename_without_ext, ext = os.path.splitext(original_filename)
                updated_filename = f"{filename_without_ext}_updated{ext}"
            else:
                updated_filename = f"modified_data_{os.path.basename(csv_path)}"
            
            updated = tempfile.NamedTemporaryFile(delete=False, suffix=ext if 'ext' in locals() else '.csv')
            updated_file_path = updated.name
            
            prompt = f"""
            Given a CSV file with the following column names and sample values:
            
            Column names: {columns}
            Sample data: {sample_data}
            
            Write Python pandas code to modify this dataset according to this query: "{query}"
            
            Your code MUST:
            0. Import all necessary libraries
            1. Read the complete CSV file from '{csv_path}' :-
            # Read the complete dataset
            if csv_path.endswith('.xlsx'):
                df = pd.read_excel(csv_path)
            else:
                try:
                    df = pd.read_csv(csv_path)
                except UnicodeDecodeError:
                    # Try different encodings
                    encodings_to_try = ['latin1', 'ISO-8859-1', 'cp1252']
                    for encoding in encodings_to_try:
                        try:
                            df = pd.read_csv(csv_path, encoding=encoding)
                            break
                        except Exception:
                            pass
                    else:
                        df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
            2. Perform the requested modifications on the dataframe
            3. Save the modified dataframe to '{updated_file_path}'
            4. Store a summary string of what was modified in a variable named 'result'
            
            Return only executable Python code with no explanations.
            """
            
            # Send to LLM
            try:
                response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
            except Exception as e:
                error_str = str(e)
                print(f"Error occurred: {error_str}")
                if "429" in error_str and "retry_delay" in error_str:
                    # Extract retry delay seconds
                    retry_seconds = 60  # Default fallback
                    # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    # if retry_match:
                    #     retry_seconds = int(retry_match.group(1))
                    
                    print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                    # Retry the request
                    response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
                else:
                    raise e
            
            # Extract the generated code
            code = response
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            # Execute the code safely
            local_vars = {"pd": pd, "csv_path": csv_path, "updated_file_path": updated_file_path}
            try:
                exec(code, {"pd": pd}, local_vars)
                
                # Upload the updated file to S3
                try:
                    s3_client.upload_file(updated_file_path, bucket_name, updated_filename)
                    
                    # Add S3 file location to result
                    modification_summary = local_vars.get("result", "File was modified")
                    result = f"{modification_summary}\nModified file saved to S3: s3://{bucket_name}/{updated_filename}"
                    return result
                except Exception as e:
                    print(f"Error uploading file to S3: {e}")
                    return f"Error uploading file to S3: {e}"
            except Exception as e:
                print(f"Error executing code: {e}")
                return f"Error modifying data: {e}"
        
        finally:
            # Clean up temporary files
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            if 'updated' in locals() and os.path.exists(updated.name):
                os.remove(updated.name)

    async def CSV_READER_FOR_ITERATOR(self,csv_path: str = "s3 path of one csv file (amazon s3 path. ex: s3://workflow-files-2709/your_file.csv)"):
        
        """
        Reads a CSV or Excel file from  S3 path and returns the content as a list of dictionaries.
        This function is designed to be used when processing CSV data row by row or needing the entire CSV file 
        as a list at once. 
        Args:
            csv_path (str): Path to the CSV/Excel file : S3 path in format "s3://bucket-name/key"
        Returns:
            list[dict]: List of dictionaries where each dictionary represents one row of the CSV file,
                        with column names as keys. Returns None if there's an error reading the file.
        
        """

        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = "workflow-files-2709"
        temp_file = None
        
        try:
            # key = csv_path

            # For Excel/CSV files that need to be read by pandas, we need to download them
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv' if not csv_path.endswith('.xlsx') else '.xlsx')
            temp_file_path = temp_file.name
            
            bucket_name = "workflow-files-2709"
                    
            try:
                # If the csv_path is an S3 path (e.g., "s3://bucket/key"), parse and download
                if csv_path.startswith("s3://"):
                    s3_parts = csv_path.replace("s3://", "").split("/", 1)
                    if len(s3_parts) == 2:
                        bucket_name, key = s3_parts
                        s3_client.head_object(Bucket=bucket_name, Key=key)
                        s3_client.download_file(bucket_name, key, temp_file_path)

                        csv_path = temp_file_path
                    else:
                        raise ValueError("Invalid S3 path format for csv_path")
                else:
                    # Try to get the object from S3 using default bucket

                    s3_client.head_object(Bucket=bucket_name, Key=csv_path)
                
                    # Download the file from S3
                    s3_client.download_file(bucket_name, csv_path, temp_file_path)
                    
                    # Update csv_path path to local path
                    csv_path = temp_file_path
            except Exception as e:
                # If error, assume csv_path is a local file path
                print(f"S3 download failed, using csv_path as local path: {e}")

            
            
            # Read the file
            if csv_path.endswith('.xlsx'):
                df = pd.read_excel(csv_path)
            else:
                df = pd.read_csv(csv_path)
            
            # Convert DataFrame to list of dictionaries (each dict represents a row)
            rows_as_json = df.to_dict(orient='records')
            return rows_as_json
        
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
        
        finally:
            # Clean up - delete temporary file if it exists
            temp_file.close()
            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)
        




class WEB_SEARCH_AGENT:
    def __init__(self,api_key,kwargs,model, llm,user_id):
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.kwargs = kwargs
        self.user_id = user_id
        if api_key:
            self.tool_set = ComposioToolSet(api_key=api_key)
            self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)

    def clean_llm_json_response(self,raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)

    async def searchfinder(self,query:str):
        task = f'''
    You are a Task Classification Agent. Your role is to analyze the following user query:

        {query}

    Your job is to determine the nature of the task requested in the query by following these steps:

    1. Identify whether the query is asking to **scrape information from a specific webpage** (i.e., it includes or refers to a URL or website), or to **perform a general web search** for a topic or question.
    2. If it is a scraping task, extract and return the **URL of the webpage** to be scraped.
    3. If it is a search task, return the **search query** as it is.

    Your output must be a valid JSON object with the following format:

    ```json
    {{
    "type": "scraping or searching",
    "URL": "URL or none if not applicable",
    "task" : "what exactly the user wants to do with the URL or search query"
    }}

            
        no preambles , no postambles, no explanations, just return the json object as described above'''
        try:
            # response = self.llm.invoke(task)

            response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)
            else:
                raise e
        print(response.content)
        response_json = self.clean_llm_json_response(response.content)
        return response_json
        
    async def search(self,metadata):
        task= metadata['task']
        prompt=f'''you are a search agent, your job is to search for the {task} and return the appropritate output in string format
        no preambles, no postambles, no explanations, just return the string as described above'''
        try:
            response = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)
            else:
                raise e
        print("llm response :",response.content)
        return response.content
       
    async def scrape(self, metadta):
        url = metadta['URL']
        task = metadta['task']

        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()  # Raise error if bad status
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all visible text from the page
            text = soup.get_text(separator='\n', strip=True)
            print(text)  # prints ALL text found on the page

            # If you want to save it or process, you can return text
            prompt=f'''you are a content extraction agent, your job  is to extract the information regarding the given task from the given text and return it in a string format 
            TASK : {task}
            ####################
            TEXT : {text}
            ####################
            no preambles, no postambles, no explanations, just return the string as described above'''
            try:
                response = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and "retry_delay" in error_str:
                    # Extract retry delay seconds
                    retry_seconds = 60  # Default fallback
                    # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    # if retry_match:
                    #     retry_seconds = int(retry_match.group(1))
                    
                    print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                    time.sleep(retry_seconds + 1)  # Add 1 second buffer
                    # Retry the request
                    response = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)

                else:
                    raise e
            print("llm response :",response.content)
            return response.content

        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")

    async def WEB_ALL_ACTIONS(self,query:str):
        """
        Processes web-related actions based on the query type and executes corresponding operations.
        This function takes a query string, determines the appropriate web action type
        (scraping or searching) and performs the relevant operation using helper methods.
        Args:
            query (str): The search query or URL to process
        Returns:
            str: The response from either the scraping or searching operation.
                Returns error message if task type is invalid.
        """



        response_json = await self.searchfinder(query=query)
        if response_json['type'] == 'scraping':
            response=await self.scrape(metadta=response_json)
        elif response_json['type'] == 'searching':
            response=await self.search(metadata=response_json)
        else:
            response = "Invalid task type. Please provide a valid query."
        return response

    async def execute(self):
        action = self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")


# add content nicely
# block, block children
# NOTION_QUERY_DATABASE
# NOTION_FETCH_ROW
# NOTION_UPDATE_ROW_DATABASE
# unresolved comment
# create comment



# GMAIL_ADD_LABEL_TO_EMAIL
# GMAIL_CREATE_LABEL



"""
create a class WEB with the initializer and must have method execute :-

class GITHUB:
    def __init__(self, api_key=None, kwargs={}):
       
        self.api_key = api_key
        self.kwargs = kwargs
        self.tool_set = ComposioToolSet(api_key=api_key)
        self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)

    def execute(self):
       
        action = self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")

apart from these methods, the web class should have functions like scrape link, find websites,  
"""



# class GITHUB:
#     def __init__(self, api_key, kwargs,model, llm,user_id):
#         """
#         Initializes the GITHUB class with API key and additional parameters.

#         Parameters:
#         - api_key (str, optional): The API key for authentication. Example: "your_github_api_key".
#         - kwargs (dict, optional): Additional parameters for the class. Example: {"action": "download_and_extract_repo"}.
#         """
#         self.model=model
#         self.llm=llm
#         self.api_key = api_key
#         self.kwargs = kwargs
#         self.user_id = user_id
#         self.tool_set = ComposioToolSet(api_key=api_key)
#         self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)

#     async def execute(self):
#         """
#         Executes the specified action dynamically.

#         Parameters:
#         - action (str): The name of the method to execute. Example: "download_and_extract_repo".

#         Returns:
#         - The result of the executed method or an error message if the method is not found.
#         """
#         action = self.kwargs["action"]
#         del self.kwargs["action"]
        
#         if hasattr(self, action):  # Check if function exists
#             return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
#         else:
#             return print(f"Method {action} not found")

#     def DOWNLOAD_AND_EXTRACT_REPO(self, owner, repo, ref="main", extract_to="extracted_repo"):
#         """
#         Downloads and extracts a GitHub repository.

#         Parameters:
#         - owner (str): The owner of the repository. Example: "octocat".
#         - repo (str): The name of the repository. Example: "Hello-World".
#         - ref (str, optional): The branch. Default is "main"

#         Returns:
#         - str: The full path to the extracted repository.

#         Raises:
#         - Exception: If the repository download fails.
#         """
#         headers = {
#             "Accept": "application/vnd.github+json",
#             "X-GitHub-Api-Version": "2022-11-28",
#         }
#         url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{ref}"
#         response = requests.get(url, headers=headers)

#         if response.status_code == 200:
#             with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
#                 zip_ref.extractall(extract_to)
#             subfolder = next((p for p in os.listdir(extract_to) if os.path.isdir(os.path.join(extract_to, p))), None)
#             full_path = os.path.join(extract_to, subfolder)
#             print(f"Repo extracted to '{full_path}/'")
#             return full_path
#         else:
#             raise Exception(f"Failed to download repo: {response.status_code} - {response.text}")

#     def LIST_GITHUB_ISSUES(self):
#         """
#         Lists GitHub issues assigned to the authenticated user.

#         Returns:
#         - dict: A dictionary containing the list of issues assigned to the authenticated user.

#         Example:
#         {
#             "issues": [
#                 {"title": "Bug fix", "url": "https://github.com/octocat/Hello-World/issues/1"},
#                 ...
#             ]
#         }
#         """
#         composio_toolset = ComposioToolSet(self.api_key)
#         return composio_toolset.execute_action(
#             action="GITHUB_LIST_ISSUES_ASSIGNED_TO_THE_AUTHENTICATED_USER",
#             params={}
#         )










# SUPPORTED_EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c'}

# class CODE_SUMMARY:
#     def __init__(self, api_key, kwargs,model, llm,user_id):
#         """
#         Initializes the CODE_SUMMARY class with API key and additional parameters.

#         Parameters:
#         - api_key (str, optional): The API key for authentication. Example: "your_api_key".
#         - kwargs (dict, optional): Additional parameters for the class. Example: {"action": "summarise_codebase"}.
#         """
#         self.model=model
#         self.llm=llm
#         self.api_key = api_key
#         self.kwargs = kwargs
#         self.user_id = user_id
#         self.tool_set = ComposioToolSet(api_key=api_key)
#         self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)

#     def extract_code_chunks(self, code, language):
#         """
#         Extracts code chunks (functions, classes, etc.) from the provided code based on the programming language.

#         Parameters:
#         - code (str): The source code to extract chunks from. Example: "def my_function():\n    pass".
#         - language (str): The programming language of the code. Example: "python", "javascript".

#         Returns:
#         - list: A list of tuples containing the code chunk, start line, and end line. Example: [("def my_function():\n    pass", 1, 2)].
#         """
#         if language == "python":
#             pattern = re.compile(r"^(def |class ).+", re.MULTILINE)
#         elif language in {"javascript", "typescript"}:
#             pattern = re.compile(r"^(function |const |let |class ).+", re.MULTILINE)
#         else:
#             pattern = re.compile(r"^.+", re.MULTILINE)

#         matches = list(pattern.finditer(code))
#         chunks = []

#         for i, match in enumerate(matches):
#             start = match.start()
#             end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
#             chunk = code[start:end].strip()
#             line_start = code[:start].count("\n") + 1
#             line_end = code[:end].count("\n") + 1
#             if chunk:
#                 chunks.append((chunk, line_start, line_end))
#         return chunks

#     def summarise_code(self, code_chunk, model):
#         """
#         Summarizes a given code chunk into natural language for documentation purposes.

#         Parameters:
#         - code_chunk (str): The code snippet to summarize. Example: "def add(a, b):\n    return a + b".
#         - model (object): The generative AI model to use for summarization. Example: genai.GenerativeModel("gemini-2.0-flash-lite").

#         Returns:
#         - str: A natural language summary of the code snippet. Example: "This function adds two numbers and returns the result."
#         """
#         prompt = (
#             "You're an expert developer. "
#             "Summarise the functionality of the following code snippet in clear, natural language for documentation purposes and be descriptive:\n\n"
#             f"{code_chunk}\n\n"
#             "Be brief but informative, and do not miss anything."
#         )
#         try:
#             try:
#                 response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
#             except Exception as e:
#                 error_str = str(e)
#                 print(f"Error occurred: {error_str}")
#                 if "429" in error_str and "retry_delay" in error_str:
#                     # Extract retry delay seconds
#                     retry_seconds = 60  # Default fallback
#                     # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
#                     # if retry_match:
#                     #     retry_seconds = int(retry_match.group(1))
                    
#                     print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
#                     await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
#                     # Retry the request
#                     response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
#                 else:
#                     raise e
#             return response.strip()
#         except Exception as e:
#             print(f"LLM error: {e}")
#             return "Summary unavailable."

#     def SUMMARISE_CODEBASE(self, root_folder, rate_limit=30, output_file="json_path_containing_summarized_code_chunks"):
#         """
#         Summarizes all code files in a given directory into natural language descriptions.

#         Parameters:
#         - root_folder (str): The root directory containing the codebase. Example: "/path/to/codebase".
#         - rate_limit (int, optional): The maximum number of code chunks to process per minute. Default is 30.
#         - output_file (str): The path to save the summarized code chunks as a JSON file. Example: "summaries.json".

#         Returns:
#         - str: The path to the output JSON file containing the summaries. Example: "/path/to/summaries.json".
#         """
#         root_folder = root_folder.rstrip("/") + "/"
#         genai.configure(api_key="AIzaSyAjXLKudWgHxVaSdi7UmzL7CtYQpQV4Nt8")
#         model = genai.GenerativeModel("gemini-2.0-flash-lite")

#         all_summaries = []
#         chunk_count = 0
#         start_time = time.time()

#         for filepath in Path(root_folder).rglob("*"):
#             if filepath.suffix in SUPPORTED_EXTENSIONS:
#                 try:
#                     code = filepath.read_text(encoding="utf-8", errors="ignore")
#                     ext = filepath.suffix
#                     language = (
#                         "python" if ext == ".py"
#                         else "javascript" if ext in [".js", ".ts", ".tsx", ".jsx"]
#                         else "cpp"
#                     )
#                     chunks = self.extract_code_chunks(code, language)

#                     for chunk, start, end in chunks:
#                         chunk_count += 1
#                         if chunk_count % rate_limit == 0:
#                             elapsed = time.time() - start_time
#                             time.sleep(max(0, 60 - elapsed))
#                             start_time = time.time()

#                         explanation = self.summarise_code(chunk, model)
#                         all_summaries.append({
#                             "file": str(filepath.relative_to(root_folder)),
#                             "lines": f"{start}-{end}",
#                             "summary": explanation
#                         })

#                 except Exception as e:
#                     print(f"Error parsing {filepath}: {e}")

#         output_path = Path(output_file).resolve()
#         file_path = os.path.join(os.path.dirname(__file__), output_path)
#         with open(file_path, "w", encoding="utf-8") as wb:
#             json.dump(all_summaries, wb, indent=2)

#         print(f"Summary saved to: {output_path}")
#         return str(output_path)

#     def execute(self):
#         """
#         Executes the specified action dynamically.

#         Parameters:
#         - action (str): The name of the method to execute. Example: "summarise_codebase".

#         Returns:
#         - The result of the executed method or an error message if the method is not found.
#         """
#         action = self.kwargs["action"]
#         del self.kwargs["action"]

#         if hasattr(self, action):  # Check if function exists
#             return getattr(self, action)(**self.kwargs)  # Call the function dynamically
#         else:
#             return print(f"Method {action} not found")










# class VECTORISER:
#     def __init__(self, api_key, kwargs,model, llm,user_id):
#         """
#         Initializes the VECTORISER class with API key and additional parameters.

#         Parameters:
#         - api_key (str, optional): The API key for authentication. Example: "your_api_key".
#         - kwargs (dict, optional): Additional parameters for the class. Example: {"action": "vectorise_codebase"}.
#         """
#         self.model=model
#         self.llm=llm
#         self.api_key = api_key
#         self.kwargs = kwargs
#         self.user_id = user_id
#         self.tool_set = ComposioToolSet(api_key=api_key)
#         self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)

#     def get_embedding(self, text):
#         """
#         Generates an embedding for the given text using the specified model.

#         Parameters:
#         - text (str): The text content to embed. Example: "This is a sample code summary."

#         Returns:
#         - list: A list of floats representing the embedding vector, or None if embedding fails.
#         """
#         try:
#             response = genai.embed_content(
#                 model="models/gemini-embedding-exp-03-07",
#                 content=text,
#                 task_type="RETRIEVAL_DOCUMENT",
#                 title="Code Summary"
#             )
#             return response['embedding']
#         except Exception as e:
#             print(f"Embedding failed: {e}")
#             return None

#     def embed_chunks_with_limit(self, summaries, delay=12):
#         """
#         Embeds a list of code summaries with a delay between each embedding request.

#         Parameters:
#         - summaries (list): A list of dictionaries containing code summaries. Example: [{"file": "file1.py", "summary": "This is a summary."}].
#         - delay (int, optional): Time in seconds to wait between embedding requests. Default is 12.

#         Returns:
#         - list: A list of dictionaries containing the original summaries with their embeddings.
#         """
#         embedded_chunks = []
#         for idx, entry in enumerate(summaries):
#             print(f"Embedding {idx + 1}/{len(summaries)}: {entry['file']}")
#             embedding = self.get_embedding(entry["summary"])
#             if embedding:
#                 embedded_chunks.append({**entry, "embedding": embedding})
#             else:
#                 print(f"Skipping chunk {idx+1}")
#             time.sleep(delay)
#         return embedded_chunks

#     def save_embeddings(self, embedded_chunks, index_path="code_chunks.index", metadata_path="code_chunks_metadata.json"):
#         """
#         Saves the embeddings to a FAISS index and metadata JSON file.

#         Parameters:
#         - embedded_chunks (list): A list of dictionaries containing embeddings and metadata. Example: [{"file": "file1.py", "embedding": [0.1, 0.2, ...]}].
#         - index_path (str): Path to save the FAISS index file. Default is "code_chunks.index". Example: "path/to/index_file.index".
#         - metadata_path (str): Path to save the metadata JSON file. Default is "code_chunks_metadata.json". Example: "path/to/metadata.json".

#         Returns:
#         - dict: A dictionary containing paths to the saved index and metadata files, or None if no valid vectors are found.
#         """
#         vectors = np.array([x["embedding"] for x in embedded_chunks]).astype("float32")
#         if vectors.size > 0:
#             index = faiss.IndexFlatL2(len(vectors[0]))
#             index.add(vectors)
#             faiss.write_index(index, index_path)
#             file_path = os.path.join(os.path.dirname(__file__), metadata_path)
#             with open(file_path, "w") as f:
#                 json.dump([{k: v for k, v in x.items() if k != "embedding"} for x in embedded_chunks], f, indent=2)
#             print("FAISS index and metadata saved.")
#             return {"vector_db_path": index_path, "metadata_json_path": metadata_path}
#         else:
#             print("No valid vectors to save.")
#             return None

#     def VECTORISE_CODEBASE(self, summaries_json="output path of summary json", delay=12, index_path="path of vector database (.index) file", metadata_path="json_path_containing_metadata_of_summarized_code"):
#         """
#         Vectorizes a codebase by embedding code summaries and saving them to a FAISS index and metadata file.

#         Parameters:
#         - summaries_json (str): Path to the JSON file containing code summaries. Example: "path/to/summaries.json".
#         - delay (int, optional): Time in seconds to wait between embedding requests. Default is 12.
#         - index_path (str): Path to save the FAISS index file. Example: "path/to/index_file.index".
#         - metadata_path (str): Path to save the metadata JSON file. Example: "path/to/metadata.json".

#         Returns:
#         - dict: A dictionary containing paths to the saved index and metadata files.
#         """
#         genai.configure(api_key="AIzaSyAjXLKudWgHxVaSdi7UmzL7CtYQpQV4Nt8")
#         file_path = os.path.join(os.path.dirname(__file__), summaries_json)
#         with open(file_path) as f:
#             summaries = json.load(f)
#         embedded_chunks = self.embed_chunks_with_limit(summaries, delay=delay)
#         return self.save_embeddings(embedded_chunks, index_path, metadata_path)

#     def execute(self):
#         """
#         Executes the specified action dynamically.

#         Parameters:
#         - action (str): The name of the method to execute. Example: "vectorise_codebase".

#         Returns:
#         - The result of the executed method or an error message if the method is not found.
#         """
#         action = self.kwargs["action"]
#         del self.kwargs["action"]

#         if hasattr(self, action):  # Check if function exists
#             return getattr(self, action)(**self.kwargs)  # Call the function dynamically
#         else:
#             return print(f"Method {action} not found")







# class VECTORQUERY:
#     def __init__(self, api_key, kwargs,model, llm,user_id):
#         self.model=model
#         self.llm=llm
#         self.api_key = api_key
#         self.kwargs = kwargs
#         self.user_id = user_id
#         self.tool_set = ComposioToolSet(api_key= api_key)
#         self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)

#     def execute(self):
#         action=self.kwargs["action"]
#         del self.kwargs["action"]
        
#         if hasattr(self, action):  # Check if function exists
#             return getattr(self, action)(**self.kwargs)  # Call the function dynamically
#         else:
#             return print(f"Method {action} not found")

#     def load_index_and_metadata(self, index_path, metadata_path):
#         genai.configure(api_key="AIzaSyAjXLKudWgHxVaSdi7UmzL7CtYQpQV4Nt8")
#         index = faiss.read_index(index_path)
#         file_path = os.path.join(os.path.dirname(__file__), metadata_path)
#         with open(file_path, "r") as f:
#             metadata = json.load(f)
#         return index, metadata

#     def get_similar_chunks(self):
#         index_path = self.kwargs["index_path"]
#         metadata_path = self.kwargs["metadata_path"]
#         query = self.kwargs["query"]
#         top_k = self.kwargs.get("top_k", 3)

#         try:
#             index, metadata = self.load_index_and_metadata(index_path, metadata_path)
#             response = genai.embed_content(
#                 model="models/gemini-embedding-exp-03-07",
#                 content=query,
#                 task_type="RETRIEVAL_QUERY"
#             )
#             embedding = np.array(response['embedding'], dtype="float32").reshape(1, -1)
#             _, indices = index.search(embedding, top_k)
#             return [metadata[i] for i in indices[0]]
#         except Exception as e:
#             return f"Similarity search failed: {e}"

#     def ANSWER_QUESTION(self,user_question,index_path="path of vector database (.index) file", metadata_path="json_path_containing_metadata_of_summarized_code"):
#         """
#         Answers a user-provided question by retrieving relevant code summaries and generating a response.
#         Parameters:
#         ----------
#         user_question : str
#             The question provided by the user that needs to be answered.
#         index_path : str
#             The file path to the vector database (.index) file used for searching relevant embeddings.
#             Example: "/home/aryan/BABLU/Agentic/data/code_embeddings.index"
#         metadata_path : str
#             The file path to the JSON file containing metadata of summarized code.
#             Example: "/home/aryan/BABLU/Agentic/data/code_metadata.json"
#         Returns:
#         -------
#         str
#             The generated answer to the user's question based on the retrieved code summaries.
        
#         """
        

#         try:
#             index, metadata = self.load_index_and_metadata(index_path, metadata_path)

#             response = genai.embed_content(
#                 model="models/gemini-embedding-exp-03-07",
#                 content=user_question,
#                 task_type="RETRIEVAL_QUERY"
#             )
#             embedding = np.array(response['embedding'], dtype="float32").reshape(1, -1)
#             _, indices = index.search(embedding, 3)
#             chunks = [metadata[i] for i in indices[0]]

#             if not chunks:
#                 return "No relevant code found."

#             context = "\n\n".join(
#                 f"File: {c['file']} (lines {c['lines']})\nSummary: {c['summary']}" for c in chunks
#             )

#             prompt = f"""You are a coding assistant. Use the following summaries to answer the question:

# {context}

# Question: {user_question}
# Answer:"""

#             model = genai.GenerativeModel("gemini-2.0-flash")
#             return await generate_con(uid=self.user_id, model=model, inside=prompt).strip()
#         except Exception as e:
#             return f"Failed to generate answer: {e}"






class GMAIL:
    def __init__(self, api_key, kwargs,model,llm,user_id):
        
        """
        Initializes the GMAIL class with API key and additional parameters.

        Parameters:
        - api_key (str): The API key for authentication. Example: "your_gmail_api_key".
        - kwargs (dict): Additional parameters for the class. Example: {"action": "GMAIL_SEND_EMAIL"}.
        """
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.kwargs = kwargs
        self.user_id = user_id
        self.tool_set = ComposioToolSet(api_key=api_key)
        self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)
    async def GMAIL_SEND_EMAIL(self, subject, body, recipient_email, is_html=False, attachment=None):
        """
        Sends an email to the specified recipient.

        Parameters:
        - subject (str): The subject of the email. Example: "Meeting Reminder".
        - body (str): The body content of the email. Example: "This is a reminder for our meeting tomorrow at 10 AM."
        - is_html (bool, optional): Whether the email body is in HTML format. Default is False. Example: True.
        - recipient_email (str): The recipient's email address. One address only. Example: "recipient@example.com".
        - attachment (str, optional):  It is the s3 file path. Default is None..

        Returns:
        - dict: A dictionary containing the status of the email sending operation.
        """
        local_files = []
        try:
            if attachment:
                # Check if the attachment is in S3
                
                bucket_name = "workflow-files-2709"

                try:
                    # If the attachment is an S3 path (e.g., "s3://bucket/key"), parse and download
                    if attachment.startswith("s3://"):
                        s3_parts = attachment.replace("s3://", "").split("/", 1)
                        if len(s3_parts) == 2:
                            bucket_name, key = s3_parts
                            s3_client.head_object(Bucket=bucket_name, Key=key)
                            temp_dir = tempfile.gettempdir()
                            local_file_path = os.path.join(temp_dir, os.path.basename(key))
                            s3_client.download_file(bucket_name, key, local_file_path)
                            local_files.append(local_file_path)
                            attachment = local_file_path
                        else:
                            raise ValueError("Invalid S3 path format for attachment")
                    else:
                        # Try to get the object from S3 using default bucket
                        s3_client.head_object(Bucket=bucket_name, Key=attachment)
                    
                        # Create a temporary file to store the downloaded content
                        temp_dir = tempfile.gettempdir()
                        local_file_path = os.path.join(temp_dir, attachment)
                        
                        # Download the file from S3
                        s3_client.download_file(bucket_name, attachment, local_file_path)
                        
                        # Add to list of files to clean up
                        local_files.append(local_file_path)
                        
                        # Update attachment path to local path
                        attachment = local_file_path
                except Exception as e:
                    # If error, assume attachment is a local file path
                    print(f"S3 download failed, using attachment as local path: {e}")

            
            # Send the email
            result = self.tool_set.execute_action(
                action="GMAIL_SEND_EMAIL",
                params={"recipient_email": recipient_email, "subject": subject, "body": body, "is_html": is_html, "attachment": attachment},
            )
            
            return result
        finally:
            # Clean up any downloaded files
            for file_path in local_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete temporary file {file_path}: {e}")

    async def GMAIL_CREATE_EMAIL_DRAFT(self, subject, body, recipient_email, is_html=False, attachment=None):
        """
        Creates an email draft with the specified details.

        Parameters:
        - subject (str): The subject of the draft email. Example: "Project Update".
        - body (str): The body content of the draft email. Example: "Please find the project update attached."
        - recipient_email (str): The recipient's email address. One address only. Example: "recipient@example.com".
        - is_html (bool, optional): Whether the email body is in HTML format. Default is False. Example: True.
        - attachment (str, optional):  It is the s3 file path. Default is None..

        Returns:
        - dict: A dictionary containing the status of the draft creation operation.
        """
        local_files = []
        try:
            if attachment:
                # Check if the attachment is in S3
                
                
                bucket_name = "workflow-files-2709"
                
                try:
                    ## If the attachment is an S3 path (e.g., "s3://bucket/key"), parse and download
                    if attachment.startswith("s3://"):
                        s3_parts = attachment.replace("s3://", "").split("/", 1)
                        if len(s3_parts) == 2:
                            bucket_name, key = s3_parts
                            s3_client.head_object(Bucket=bucket_name, Key=key)
                            temp_dir = tempfile.gettempdir()
                            local_file_path = os.path.join(temp_dir, os.path.basename(key))
                            s3_client.download_file(bucket_name, key, local_file_path)
                            local_files.append(local_file_path)
                            attachment = local_file_path
                        else:
                            raise ValueError("Invalid S3 path format for attachment")
                    else:
                        # Try to get the object from S3 using default bucket
                        s3_client.head_object(Bucket=bucket_name, Key=attachment)
                    
                        # Create a temporary file to store the downloaded content
                        temp_dir = tempfile.gettempdir()
                        local_file_path = os.path.join(temp_dir, attachment)
                        
                        # Download the file from S3
                        s3_client.download_file(bucket_name, attachment, local_file_path)
                        
                        # Add to list of files to clean up
                        local_files.append(local_file_path)
                        
                        # Update attachment path to local path
                        attachment = local_file_path
                except Exception as e:
                    # If error, assume attachment is a local file path
                    print(f"S3 download failed, using attachment as local path: {e}")
            
            # Create the email draft
            result = self.tool_set.execute_action(
                action="GMAIL_CREATE_EMAIL_DRAFT",
                params={"recipient_email": recipient_email, "subject": subject, "body": body, "is_html": is_html, "attachment": attachment},
            )
            
            return result
        finally:
            # Clean up any downloaded files
            for file_path in local_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete temporary file {file_path}: {e}")

    async def GMAIL_REPLY_TO_THREAD(self, thread_id, body, recipient_email,is_html=False, attachment=None):
        """
        Replies to an email thread if and only if an email thread already exists. 

        Parameters:
        - thread_id (str): The ID of the email thread to reply to. Example: "1234567890abcdef".
        - body (str): The body content of the reply email. Example: "Thank you for the reminder."
        - is_html (bool, optional): Whether the email body is in HTML format. Default is False. Example: True.
        - attachment (str, optional): It is the s3 file path. Default is None.
        - recipient_email (str) : the email id to whome the email should be send . 

        Returns:
        - dict: A dictionary containing the status of the reply operation.
        """
        local_files = []
        try:
            if attachment:
                # Check if the attachment is in S3
                
               
                bucket_name = "workflow-files-2709"
                
                try:
                    # If the attachment is an S3 path (e.g., "s3://bucket/key"), parse and download
                    if attachment.startswith("s3://"):
                        s3_parts = attachment.replace("s3://", "").split("/", 1)
                        if len(s3_parts) == 2:
                            bucket_name, key = s3_parts
                            s3_client.head_object(Bucket=bucket_name, Key=key)
                            temp_dir = tempfile.gettempdir()
                            local_file_path = os.path.join(temp_dir, os.path.basename(key))
                            s3_client.download_file(bucket_name, key, local_file_path)
                            local_files.append(local_file_path)
                            attachment = local_file_path
                        else:
                            raise ValueError("Invalid S3 path format for attachment")
                    else:
                        # Try to get the object from S3 using default bucket
                        s3_client.head_object(Bucket=bucket_name, Key=attachment)
                    
                        # Create a temporary file to store the downloaded content
                        temp_dir = tempfile.gettempdir()
                        local_file_path = os.path.join(temp_dir, attachment)
                        
                        # Download the file from S3
                        s3_client.download_file(bucket_name, attachment, local_file_path)
                        
                        # Add to list of files to clean up
                        local_files.append(local_file_path)
                        
                        # Update attachment path to local path
                        attachment = local_file_path
                except Exception as e:
                    # If error, assume attachment is a local file path
                    print(f"S3 download failed, using attachment as local path: {e}")
            
            # Reply to the thread
            result = self.tool_set.execute_action(
                action="GMAIL_REPLY_TO_THREAD",
                params={"thread_id": thread_id, "message_body": body, "is_html": is_html, "attachment": attachment,"recipient_email":recipient_email},
            )
            
            return result
        finally:
            # Clean up any downloaded files
            for file_path in local_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete temporary file {file_path}: {e}")




    async def GMAIL_FETCH_OLD_EMAILS(self, query=None):
        """
        Fetches old emails based on the specified query.

        Parameters:
        - query (str): The search query to filter emails. IT is a natural language query. Example: "fetch all emails from aryan22102@iiitnr.edu.in till today".
        Returns:
        - list: A list of dictionaries containing email details.
        """
      
        current_date = datetime.now().date()
        prompt = f"""
            You are a Gmail search specialist. Convert this natural language request into a proper Gmail search query:
            "{query}"
            
            Use Gmail's search operators like:
            - from: (sender)
            - to: (recipient)
            - subject: (email subject)
            - label: (Gmail label)
            - has:attachment
            - is:unread
            - after:YYYY/MM/DD
            - before:YYYY/MM/DD
            - logical operators (AND, OR, NOT or -)
            - "exact phrase" (in quotes)
            
            Return ONLY a JSON object with these fields:
            {{
              "query": "the properly formatted Gmail search query",
              "label_ids": [] (list of label IDs or empty list if none specified),
            }}
            
            some common label ids :
            'UNREAD', 'STARRED'

            today's date is {current_date}
            Do not include any explanation, just the JSON.
            """
        #
        
        try:
            llm_response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
        except Exception as e:
            error_str = str(e)
            print(f"Error occurred: {error_str}")
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                llm_response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
            else:
                raise e

        if llm_response[0]=="`":
            llm_response=json.loads(llm_response[7:-4])
        else:
            llm_response=json.loads(llm_response)


        result = self.tool_set.execute_action(
                action="GMAIL_FETCH_EMAILS",
                params={"query": llm_response['query'], "label_ids": llm_response['label_ids'], "max_results": 500},
            )
        

        # tools = self.prompt_toolset.get_tools(actions=['GMAIL_FETCH_EMAILS'])
        # agent = create_openai_functions_agent(llm, tools, prompt)
        # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        # result = agent_executor.invoke({"input": query})
        # agent_executor
        final = []
        # print("result",result)
        for i in result['data']["messages"]:
            # Create a copy of the dict and remove the payload
            # message_dict = i.copy()
            # extracted=extract_email(message_dict)
            attachments=i["attachmentList"]
            text=i["messageText"]
            timestamp=i["messageTimestamp"]
            sender=i["sender"]
            subject=i["subject"]
            thread= i["threadId"]
            # Extract useful text from the message - handle potential HTML content

            # Check if text is HTML and extract plain text if needed
            if "<html" in text or any(tag in text for tag in ["<div", "<p", "<body", "<table", "<span", "<a", "<ul", "<ol", "<li", "<h1", "<h2", "<h3", "<img", "<br", "<hr"]):
                soup = BeautifulSoup(text, 'html.parser')
                
                # Extract all links before removing tags
                links = []
                for a_tag in soup.find_all('a', href=True):
                    link_text = a_tag.get_text(strip=True)
                    href = a_tag['href']
                    links.append(f"{link_text} ({href})")
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text(separator=' ', strip=True)
                
                # Append links to the text
                if links:
                    text += "\n\nLinks:\n" + "\n".join(links)
                            
                          
            extracted = {
                
                "sender": sender,
                "subject": subject,
                "text": text,
                "timestamp": timestamp,
                "attachments": attachments,
                "threadId": thread
            }


            # if "payload" in message_dict:
            #     del message_dict["payload"]
            final.append(extracted)
        return final


    

    

    async def execute(self):
        """
        Executes the specified action dynamically.

        Parameters:
        - action (str): The name of the method to execute. Example: "GMAIL_SEND_EMAIL".

        Returns:
        - The result of the executed method or an error message if the method is not found.
        """
        action = self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")
   

class GOOGLESHEETS:

    def __init__(self,api_key,kwargs,model, llm, user_id):
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)
        self.kwargs = kwargs
        self.user_id = user_id
        
    def get_id_from_url(self,url):
        match = re.search(r'/d/([a-zA-Z0-9_-]+)/', url)
        return match.group(1) if match else None  
   
    def clean_llm_json_response(self,raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)


    async def GOOGLESHEETS_BATCH_GET(self,spreadsheetID,sheet_name=None):
        
        params={'spreadsheet_id':spreadsheetID,"ranges":[sheet_name]}
        
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_GET')
        
        values = response['data']['valueRanges'][0].get('values', [])
        header = values[0] if values else []
        rows = values[1:] if len(values) > 1 else []
        
        mapped_data = [dict(zip(header, row)) for row in rows]
        print("mapped data:",mapped_data)

        return {'response':mapped_data,"headers":header}

    async def get_sheet_header(self,spreadsheetID, sheet_name=None):

        response=await self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        headers = response['headers']
        return headers

    async def header_description(self,spreadsheetID,sheet_name=None):

        # header=self.get_sheet_header(spreadsheetID=spreadsheetID, sheet_name=sheet_name)
        data=await self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        header = data['headers']  # Use the headers from the response
        description={}
        # Collect all column data samples in one dictionary
        column_samples = {}
        for j in header:
            col_data = []
            
            for i in data['response']:
                col_data.append(i.get(j, ''))
                
            column_samples[j] = col_data

        # Create a single prompt for all columns
        all_columns_prompt = "You are analyzing columns from a table. For each column, provide a brief description of the data and its usage in the table.\n\n"
        for column_name, samples in column_samples.items():
            all_columns_prompt += f"Column: {column_name}\nSample data: {samples[0] if samples else 'No samples available'}\n\n"
        all_columns_prompt += "Return a parsable JSON object where each key is a column name string and each value is its description string. Be concise and direct without reasoning or explanation. Please keep all the columns in the response json, no matter samples available or not. please keep all strings in double quotes. No preambles or postambles, just the JSON object.\n\n"

        # Make a single LLM call
        try:
            response = get_chain(uid=self.user_id,prompt=self.llm,inside=all_columns_prompt,mo="gemini-2.0-flash",path=1)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = get_chain(uid=self.user_id,prompt=self.llm,inside=all_columns_prompt,mo="gemini-2.0-flash",path=1)
            else:
                raise e
            
        # Parse the JSON response
        try:
            description = self.clean_llm_json_response(response.content.strip())
            # Convert descriptions to lowercase for consistency
            description = {k: v.lower() for k, v in description.items()}
        except json.JSONDecodeError:
            # Fallback in case response isn't valid JSON
            description = {}
            for j in header:
                # Extract descriptions using regex pattern matching if possible
                pattern = rf"{j}.*?:.*?\"(.*?)\""
                match = re.search(pattern, response.content, re.IGNORECASE | re.DOTALL)
                if match:
                    description[j] = match.group(1).lower()
                else:
                    description[j] = "Description unavailable"

        return description
            # print(j," :- ",response1.content.strip().lower())
    async def add_new_data(self,spreadsheetID,data:list,sheet_name=None):
        response=await self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        new_row=len(response['response'])+2
        print(new_row)
        # response1=composio_toolset.execute_action(params={'spreadsheet_id':spreadsheetID,'ranges':[sheet_name]},action = 'GOOGLESHEETS_BATCH_GET' )
        # sheet_name=response1['data']['valueRanges']['range'].split('!')[0]
        params={'spreadsheet_id':spreadsheetID,'sheet_name':sheet_name,'first_cell_location':f'A{new_row}','values':[data],'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        response3=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_UPDATE' )
        return {'response':response3['successfull']} 
    async def update_cell(self,spreadsheetID,data:str,column_name,old_value,sheet_name=None):
        response=await self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        #print(response['response'][1])
        k=1
        for i in response['response']:
            #print(i)
            if i[column_name]==old_value:
                row=k
                header=response['headers']
                column_number=header.index(column_name)+1
                column_letter=chr(64+column_number)
                cell_location=f'{column_letter}{row+1}'
                params={'spreadsheet_id':spreadsheetID,'sheet_name':sheet_name,'first_cell_location':cell_location,'values':[[data]],'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
                try:
                    composio_toolset = ComposioToolSet(api_key=self.api_key)
                    response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_UPDATE' )
                    return {'response':response['successfull']}
                except Exception as e:
                    return {'error':str(e)}
            else:
                k+=1
    async def update_cell_by_row_condition(self,spreadsheetID,data:str,column_name,condition:dict,sheet_name=None):
        response=await self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        #print(response['response'][1])
        k=1
        for i in response['response']:
            #print(i)
            if all(item in i.items() for item in condition.items()):
                row=k
                header=response['headers']
                column_number=header.index(column_name)+1
                column_letter=chr(64+column_number)
                cell_location=f'{column_letter}{row+1}'
                params={'spreadsheet_id':spreadsheetID,'sheet_name':sheet_name,'first_cell_location':cell_location,'values':[[data]],'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
                try:
                    composio_toolset = ComposioToolSet(api_key=self.api_key)
                    response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_UPDATE' )
                    return {'response':response['successfull']}
                except Exception as e:
                    return {'error':str(e)}
            else:
                k+=1
    # def update_condition_divider(self,spreadsheetID,query:str,sheet_name=None):
    #     details=self.header_description(spreadsheetID=spreadsheetID, sheet_name=sheet_name)
    #     header=self.get_sheet_header(spreadsheetID=spreadsheetID, sheet_name=sheet_name)

    #     task1=f'''for the given query : {query} \nthere are two types of data present. one is the data which is to be updated and other is the condition on which the data is to be updated.
    #     now find the data which is to be updated , which means the new data which will replace the older one and return just the data.
    #     the final response will be containing a single entity as the data which is to be updated.
        
    #     STRICTLY FOLLOW THE ABOVE TASK GIVEN AND EXTRACT INFORMATION FROM THE QUERY CONTENT NO REASONING NO EXPLANATION NO APPROCH IS REQUIRED JUST THE DATA '''
    #     response1=self.llm.invoke(task1)
    #     data= response1.content.strip()
    #     print(data)
    #     task2=f'''from the given query :{query}\n it has been understood that we need to update some value in a table with data :{data}\n but yet don't know in which column . 
    #      now look into column_detail where every key is the column name and the value is the detail of the column :\n {details}\n and find out in which column the updation is required 
    #      the final response will be containing a single entity as the column name where we have to update. following  are the column name so the entity should be from this list: {header}\n
    #     STRICTLY FOLLOW THE ABOVE TASK GIVEN AND EXTRACT INFORMATION FROM THE QUERY CONTENT NO REASONING NO EXPLANATION NO APPROCH IS REQUIRED JUST THE DATA'''
    #     response2=self.llm.invoke(task2)
    #     column_name= response2.content.strip()
    #     print(column_name)

    #     task3=f'''from the user's query : {query} \nit has been understood that we need to update at column : {column_name} with data : {data} \nbut yet we don't know that in which row .
    #     the row requirement has been in the query . now we have to find out the row condition then follow these steps:
    #     step 1 : in query there must be some value which will be acting as the condition for the row and its not data
    #     step 2 : once value has been found look into these column details where every key is taken from available columns and every column value depicts the following column description : {details}
    #     step 3 : make the dictonary of the column name found in the step 2 and value in step 1 in the format : 'column_name':value . also there can be be more than one conditional columns
        
    #     the final response must contain a single entity as the json of format 
    #     column name : value
    #     NO PRIEMBLES AND POSTAMBLES ARE REQUIRED , AND KEEP ALL STRINGS IN DOUBLE QUOTES'''
    #     response3=self.llm.invoke(task3)
    #     print("bhej dia hai guru")
    #     print(response3.content.strip())
    #     condition=self.clean_llm_json_response(response3.content.strip())
    #     print(condition)
    #     print(type(condition))

    #     return {'data':data,'column_name':column_name,'condition':condition}
    async def standardize_methodology_update(self,spreadsheet_url,query:str,sheet_name:str):
        spreadsheetID=self.get_id_from_url(spreadsheet_url)
        make_df = await self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        df=pd.DataFrame(make_df['response'])

        # Ask LLM to return structured update instruction
        task = f'''
        Given this user query: "{query}", return a JSON object describing the update.
        

        Column descriptions: {await self.header_description(spreadsheetID=spreadsheetID, sheet_name=sheet_name)}
        Column data types: {df.dtypes.astype(str).to_dict()}
        
        Return only this format:
        {{
        "column": "column_to_update",
        "value": "new_value",
        "condition": {{
            "column": "condition_column",
            "value": condition_value
        }}
        }}

         - The column names and condition_value  should be exactly same as in the DataFrame.
        - Use string values in double quotes.
        - Use correct Python types (e.g. strings in double quotes, numbers as-is).
        - Do not include any preambles and postambles.
        '''

        try:
            response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1).content.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1).content.strip()
            else:
                raise e
       
        update_instruction=self.clean_llm_json_response(response)
        print(update_instruction)
        # Extract and apply the update
        col_to_update = update_instruction['column']
        new_value = update_instruction['value']
        condition = update_instruction['condition']
        cond_col = condition['column']
        cond_val = condition['value']

        # Apply the update
        try:
            # Convert condition value to match DataFrame column type if needed
            try:
                # Add these debug prints
                print(f"Column type check: {pd.api.types.is_numeric_dtype(df[cond_col])}")
                print(f"Original cond_val: {cond_val}, type: {type(cond_val)}")

                if pd.api.types.is_numeric_dtype(df[cond_col]):
                    # Handle numeric column
                    original_cond_val = cond_val
                    cond_val = pd.to_numeric(cond_val, errors='coerce')
                    print(f"After conversion cond_val: {cond_val}, type: {type(cond_val)}")
                    
                    # Check if value was coerced to NaN
                    if pd.isna(cond_val):
                        print("Using isna() for comparison since cond_val is NaN")
                        df.loc[pd.isna(df[cond_col]), col_to_update] = new_value
                    else:
                        df.loc[df[cond_col] == cond_val, col_to_update] = new_value
                else:
                    # For string columns, try more flexible matching
                    print("String column detected. Trying case-insensitive and trimmed comparison.")
                    # Clean both the condition value and the dataframe values
                    clean_cond_val = str(cond_val).strip().lower()
                    # Create a mask with cleaned values
                    mask = df[cond_col].str.strip().str.lower() == clean_cond_val
                    print(f"Values in column: {df[cond_col].unique()}")
                    print(f"Cleaned condition value: '{clean_cond_val}'")
                    print(f"Number of rows matching cleaned condition: {mask.sum()}")
                    # Apply the update using the mask
                    df.loc[mask, col_to_update] = new_value
                    
                # Check how many rows were updated
                matching_rows = df[df[col_to_update] == new_value].shape[0]
                print(f"Number of rows updated: {matching_rows}")
            except Exception as e:
                print(f"Error updating value: {e}")
                
            data_list = []
            for i in df.values.tolist():
                # Replace None values with empty string
                i = ["" if val is None or pd.isna(val) else val for val in i]
                data_list.append(list(i))
                print("i:",list(i))

            params={'spreadsheet_id':spreadsheetID,'sheet_name':sheet_name,'first_cell_location':'A2','values':data_list,'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
            try:
                composio_toolset = ComposioToolSet(api_key=self.api_key)
                response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_UPDATE' )
                return response['successfull']
            except Exception as e:
                return response['successfull']
        except KeyError as e:
            return f"Error: {e}. Column '{col_to_update}' or '{cond_col}' not found in DataFrame."
          # Debug print

        # Optional: update Google Sheet back here with updated df
    async def standardize_methedology_read(self , spreadsheet_url , query:str,sheet_name:str):
        spreadsheetID=self.get_id_from_url(spreadsheet_url)
        # Load and clean sheet data into a DataFrame
        #print(self.header_description(spreadsheetID=spreadsheetID))
        make_df=await self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        df = pd.DataFrame(make_df['response'])
        df.columns = [re.sub(r'\W+', '', col).lower() for col in df.columns]
        #print(df.dtypes)

        # Construct LLM prompt
        task = f'''
            Given this user query: "{query}", return a JSON object describing the read operation on a pandas DataFrame.

            Available columns: {df.columns.tolist()}
            Column descriptions: {await self.header_description(spreadsheetID=spreadsheetID, sheet_name=sheet_name)}
            Column data types: {df.dtypes.astype(str).to_dict()}

            Return only this format:
            {{
            "select_columns": ["col1", "col2", ...],
            "condition": {{
                "column": "condition_column",
                "operator": "comparison_operator",   # e.g., ">", "<", "==", ">=", "<=", "!="
                "value": condition_value
            }}
            }}
            the above made format is basically used for reading specific data , but in case of read complete data the above condition will have all null . Or in case of reading a column the operator and the value will be null. 

            - Use string values in double quotes.
            - Use correct Python types (e.g. strings in quotes, numbers as-is).
            - Do not include any preambles or postambles.
            '''


        ops = {
            "==": operator.eq,
            "!=": operator.ne,
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le
        }
        try:
            response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)
            else:
                raise e
        spec=self.clean_llm_json_response(response.content.strip())
        #print(spec)
        select_cols = spec["select_columns"]
        cond = spec["condition"]
        
        if cond['operator']==None or "None":
            
            if cond['column']==None or "None":

                if cond['value']==None or "None":
                    return df.to_dict(orient='records')
            
            else:
                return df[select_cols].to_dict(orient='records')
        
        else:
            cond_col = cond["column"]
            cond_op = cond["operator"]
            cond_val = cond["value"]

            # Safe results list
            results = []

            # Loop through rows
            for _, row in df.iterrows():
                cell_val = row[cond_col]

                try:
                    # Convert types for safe comparison
                    if isinstance(cond_val, (int, float)):
                        cell_val = float(cell_val)
                    elif isinstance(cond_val, bool):
                        cell_val = str(cell_val).strip().lower() in ['true', '1', 'yes']
                    else:
                        cell_val = str(cell_val)

                    if ops[cond_op](cell_val, cond_val):
                        results.append({col: row[col] for col in select_cols})
                        return results
                except Exception as e:
                    print(f"Skipping row due to error: {e}")
                    continue

            # Output
            print(results)
    async def standardize_methodology_add(self,spreadsheet_url,query:str,sheet_name:str='Sheet1'):
        spreadsheetID=self.get_id_from_url(spreadsheet_url)
        header_description=await self.header_description(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        task= f'''you are a query to data maker agent that looks into the user's query and try to find the appropriate data that suits the header and header description of the google sheet
               
                expected output is a list of jsons with each json having-
                1. keys as headers
                2. values corresponding to the header that you have extracted from user's query 
                3. if for any in headers no appropriate value is found then make its value as empty string
                
                no preambles , postambles and explainantion needed just the list of jsons is required

                
                header description : {header_description}
                query : {query}
                
                '''
        try:
            response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)
            else:
                raise e
        data=self.clean_llm_json_response(response.content)
        rows=await self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        new_data_location='A'+str(len(rows['response'])+2)
        final_data=[]
        for d in data:
            data_list = ['' if v is None else str(v) for v in d.values()]
            final_data.append(data_list)
        print("final data : ",final_data)
        params={'spreadsheet_id':spreadsheetID,'sheet_name':sheet_name,'first_cell_location':new_data_location,'values':final_data,'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
        try:
            composio_toolset = ComposioToolSet(api_key=self.api_key)
            response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_UPDATE' )
            return {'response':response['successfull']}
        except Exception as e:
            return {'error':str(e)}
    
    
    async def CREATE_SHEET_FROM_SCRATCH_AND_ADD_DATA(self,query:str):
        """
        Processes a natural language query to extract unstructured data entries and creates a Google Sheet with relevant columns from them.
        Args:
            query (str): A natural language string containing data entries in unstructured format, to be extracted and tabulated.
        Returns:
            dict: 
                - On success: {'sheet_url': <URL of the created Google Sheet>}
                - On failure: {'error': <error message>}
        Note : This tool will execute successfully only if the query contains data entries that can be tabulated, If there are no data entries in the query, it will return an error.
        
        """
        task = f'''You are a Google Sheets agent. Given the following query: \n"{query}",\n extract all data entries that should become rows in a sheet.
                    For each entry in the query, create a JSON object where:
                    - The keys are the column names (as inferred from the query, e.g., "name", "product", "quantity", etc.).
                    - The values are the corresponding values extracted from the query.
                    
                    If the query contains multiple data entries, return a list of JSON objects (one per row). 
                    If column names are not explicitly provided then infer them from the data entries.
                    If a value for a column is missing in an entry, set its value to null.
                    
                    Return only the list of JSON objects, no explanation or extra text. Example:
                    [
                      {{"name": "Alice", "product": "chips", "quantity": 3, "amount": 60}},
                      {{"name": "Bob", "product": "soda", "quantity": 2, "amount": 40}}
                    ]
                    '''
        try:
            json_response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1).content
            print("json response : ",json_response)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                json_response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1).content
            else:
                raise e
        print(json_response)
        try:
            list_composio=self.clean_llm_json_response(json_response)
            print("cleaned response :",list_composio ,type(list_composio))
            try:
                task_new=f'''you are a google sheets agent that looks into the {list_composio} and try to find the appropriate title that suits the google sheet 
                expected output is just a small string which will be title that suits the google sheet and no preambles , postambles and explainantion needed just the string'''
                try:
                    title = get_chain(uid=self.user_id,prompt=self.llm,inside=task_new,mo="gemini-2.0-flash",path=1).content
                    print("title : ",title)
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str and "retry_delay" in error_str:
                        # Extract retry delay seconds
                        retry_seconds = 60  # Default fallback
                        # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                        # if retry_match:
                        #     retry_seconds = int(retry_match.group(1))
                        
                        print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                        time.sleep(retry_seconds + 1)  # Add 1 second buffer
                        # Retry the request
                        title = get_chain(uid=self.user_id,prompt=self.llm,inside=task_new,mo="gemini-2.0-flash",path=1).content
                        print("title : ",title)
                    else:
                        raise e
                params={'title':title,'sheet_name':'Sheet1','sheet_json':list_composio}
                composio_toolset = ComposioToolSet(api_key=self.api_key)
                response3=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_SHEET_FROM_JSON')
                #https://docs.google.com/spreadsheets/d/1JbboZxp5tzVSEWSCcbZllH288dcgj5xy7c1ciSuOlpE
                return {'sheet_url':"https://docs.google.com/spreadsheets/d/"+str(response3['data']['response_data']['spreadsheetId'])}
            except Exception as e:
                print("Error creating sheet:", e)
                return {'error': str(e)}
        except Exception as e:
            print("Error cleaning response:", e)


    async def GOOGLESHEETS_ALL_ACTIONS(self,spreadsheet_url : str,query: str, sheet_name: str = 'Sheet1'):
        """
        Processes Google Sheets operations based on the user's query intent.
        This function uses an LLM to classify the user's intent and routes to appropriate
        handler methods for different Google Sheets operations.
        Args:
            spreadsheet_url (str): The URL of the Google Sheet to operate on
            query (str): A natural language description of the operation to perform,
                        which might involve updating data, reading data, etc. 
                        The query must contain data to be updated or modified.
            sheet_name (str) : The name of the sheet to operate on. If not provided, defaults to 'Sheet1'.
        Returns:
            either True for update query or list of the data for read query 

        """
        
        task = f"""
                    You are an AI assistant classifying actions for Google Sheets operations.
                    Given the following query:
                    "{query}"
                    Classify whether the intent is to 'create', 'update', 'read' or 'add a new row or column' the data.
                    Respond with only one word: create,update,read,add.
                    """
        # Call the model
        try:
            response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)

        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)
            else:
                raise e

        # If using Langchain's return object
        classifies=response.content.strip().lower()

        print(classifies)
        if classifies=='update':
            try:    
                response=await self.standardize_methodology_update(spreadsheet_url=spreadsheet_url,query=query,sheet_name=sheet_name)
                return response
            except Exception as e:
                return {'error':str(e)}
            # If using Langchain's return object
                # print(col_data)
        elif classifies=='read':
            try:
                response=await self.standardize_methedology_read(spreadsheet_url=spreadsheet_url,query=query,sheet_name=sheet_name)
                return response
            except Exception as e:
                return {'error':str(e)}
        elif classifies=='add':
            try:
                response=await self.standardize_methodology_add(spreadsheet_url=spreadsheet_url,query=query,sheet_name=sheet_name)
                return response
            except Exception as e:
                return {'error':str(e)}

        

    async def execute(self):
        action = self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")
            
    
from tools.whatsapp import send_whatsapp_message, s3_to_whatsapp
class WHATSAPP:    
    def __init__(self, api_key, kwargs,model,llm,user_id):
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.kwargs = kwargs
        self.user_id = user_id


    async def SEND_WHATSAPP_MESSAGE(self, phone_number: str, message: str):
        """
        Sends a WhatsApp message to the specified phone number from SIGMOYD WHATSAPP

        Parameters:
        - phone_number (str): The phone number to send the message to (with country code, without +). Example: "917000978867"
        - message (str): The content of the message to send. Example: "Hello, this is a test message."

        Important: The message cannot be very long (it should strictly be less than 500 words) (make it under 500 words if message is long)
        

        Returns:
        - dict: A dictionary containing the success state of the message sending.
        """
        
        # If message is longer than 4000 characters, split it into chunks
        if len(message) > 4000:
            chunks = []
            current_chunk = ""
            
            # Split by sentences to keep chunks coherent
            sentences = message.replace('\n', ' \n').split('. ')
            
            for sentence in sentences:
                # Add period back except for lines that end with newline
                if not sentence.endswith('\n'):
                    sentence += '.'
                
                # If adding this sentence would exceed chunk size, start new chunk
                if len(current_chunk) + len(sentence) > 4000:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += ' ' + sentence if current_chunk else sentence
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
            
            # Send each chunk as a separate message
            results = []
            for chunk in chunks:
                result = send_whatsapp_message(chunk, phone_number)
                results.append(result)
            
            # Return success if all messages were sent successfully
            return results
        else:
            # For regular-sized messages, just send
            return send_whatsapp_message(message, phone_number)
        


    async def HUMAN_IN_THE_LOOP(self, question: str):
        """
        Tool used to get manual validation/feedback from user between the workflow steps.
        Note: This tool is intended to use only between the workflow steps. no independent utilisation of this tool is allowed.

        Parameters:
        - question (str): The question or prompt to get feedback on. Example: "can i send the mail, or some corrections are needed?"



        """

        

    async def execute(self):
        action = self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")

# #####
# class GOOGLEDRIVE:
#     def __init__(self, api_key,kwargs,model, llm,user_id):
#         self.model=model
#         self.llm=llm
#         self.api_key = api_key
#         self.kwargs=kwargs
#         self.user_id = user_id

#     def GOOGLEDRIVE_GET_FILE_CONTENT_FROM_URL(self,url:str='this is the url of the file'):
#         pass

class LINKEDIN:
    def __init__(self, api_key, kwargs,model,llm,user_id):
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.kwargs = kwargs
        self.user_id = user_id


    async def CREATE_LINKEDIN_POST(self, post_content, is_resharable=True):
        """
        Creates a LinkedIn post with the specified content and visibility settings.

        Parameters:
        - post_content (str): The content of the LinkedIn post. Example: "Excited to share my latest project!"
        - is_resharable (bool, optional): Whether the post can be reshared by others. Default is True. Example: True

        Returns:
        - dict: A dictionary containing the success state of the post creation.
        """
        toolset = ComposioToolSet(api_key=self.api_key)
        linkedin_internal = LINKEDIN(api_key=self.api_key, kwargs={"action": "linkedin_get_my_info"})
        result_internal = linkedin_internal.execute()
        author_id = result_internal['data']['response_dict']['author_id']
        return {"state": toolset.execute_action(
            action=Action.LINKEDIN_CREATE_LINKED_IN_POST,
            params={
            "author": author_id,
            "commentary": post_content,
            "visibility": "PUBLIC",
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": is_resharable
            },
        )["successfull"]}

    async def LINKEDIN_JOBS_SEARCH(self, query="linkedin search query", pages=1):
        """
        Searches for LinkedIn jobs based on the provided query.

        Parameters:
        - query (str): The search query for LinkedIn jobs. Example: "Data Scientist".
        - pages (int, optional): The number of pages of results to fetch. Default is 1. Example: 2

        Returns:
        - list: A list of dictionaries containing job details such as job ID, position, company name, etc.
        """
        url = "https://api.scrapingdog.com/linkedinjobs/"
        params = {
            "api_key": self.api_key,
            "field": query,
            "geoid": "102713980",
            "page1234567": pages
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            extracted_data = [
            {
                "job_id": item.get("job_id"),
                "job_position": item.get("job_position"),
                "company_name": item.get("company_name"),
                "company_profile": item.get("company_profile"),
                "job_location": item.get("job_location"),
                "job_posting_date": item.get("job_posting_date"),
                "job_link": item.get("job_link"),
                "company_logo_url": item.get("company_logo_url")
            }
            for item in data
            ]
            return extracted_data
        else:
            return []

    async def LINKEDIN_GET_PECIFIC_JOB_INFO(self, query="job title", pages=1, limit=5):
        """
        Retrieves detailed information about specific LinkedIn jobs based on a search query.

        Parameters:
        - query (str): The job title or keyword to search for. Example: "Software Engineer".
        - pages (int, optional): The number of pages of results to fetch. Default is 1. Example: 2
        - limit (int, optional): The maximum number of job details to retrieve. Default is 5. Example: 3

        Returns:
        - list: A list of dictionaries containing detailed job information such as position, location, description, etc.
        """
        result = await self.LINKEDIN_JOBS_SEARCH(query=query, pages=pages)
        all_responses = []

        for x in range(min(limit, len(result))):
            url = "https://api.scrapingdog.com/linkedinjobs"
            api_key = self.api_key
            job_id = result[x]['job_id']
            params = {
            "api_key": api_key,
            "job_id": job_id
            }
            response = requests.get(url, params=params)

            if response.status_code == 200:
                try:
                    job_data = response.json()

                    # If API returns a list, take the first dictionary
                    if isinstance(job_data, list) and job_data:
                        job_data = job_data[0]  # Extract the first job result

                    if isinstance(job_data, dict):
                        required_details = {
                        "job_position": job_data.get("job_position"),
                        "job_location": job_data.get("job_location"),
                        "company_name": job_data.get("company_name"),
                        "company_linkedin_id": job_data.get("company_linkedin_id"),
                        "job_posting_time": job_data.get("job_posting_time"),
                        "job_description": job_data.get("job_description"),
                        "Seniority_level": job_data.get("Seniority_level"),
                        "Employment_type": job_data.get("Employment_type"),
                        "Job_function": job_data.get("Job_function"),
                        "Industries": job_data.get("Industries"),
                        "job_apply_link": job_data.get("job_apply_link"),
                        "recruiter_details": job_data.get("recruiter_details", [])
                    }
                        all_responses.append(required_details)
                    else:
                        all_responses.append({"error": "Unexpected response format", "job_id": job_id})

                except json.JSONDecodeError:
                    all_responses.append({"error": "Invalid JSON response", "job_id": job_id})
            else:
                all_responses.append({
                "error": f"Request failed with status code {response.status_code}",
                "job_id": job_id
            })

        return all_responses

    async def LINKEDIN_GET_RECRUITER_EMAILS(self, job_title, quantity=10):
        """
        Retrieves recruiter emails from LinkedIn posts related to a specific job title.

        Parameters:
        - job_title (str): The job title to search for in LinkedIn posts. Example: "Data Scientist".
        - quantity (int, optional): The number of posts to extract emails from. Default is 10. Example: 5

        Returns:
        - list: A list of strings containing recruiter emails or post content.
        """

        try:
            url = "https://cb12-2401-4900-51dd-60f8-54be-8d77-d471-a4d4.ngrok-free.app/recruiter-emails"
            payload = {
                "job_title": job_title,
                "quantity": quantity
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return [f"Error: {response.status_code}, {response.text}"]
        except Exception as e:
            return f"Failed to retrieve recruiter emails: {str(e)}"
            



        options=webdriver.ChromeOptions()
       #uncomment for cloud
        # options.add_argument("--headless")
        # options.add_argument("--no-sandbox")
        # options.add_argument("--disable-dev-shm-usage")
        # options.add_argument("--disable-gpu")
        # options.add_argument("--disable-software-rasterizer")
        # service = Service(ChromeDriverManager().install())





        # options.add_argument("--headless")
        # options.add_argument("--no-sandbox")
        # options.add_argument("--disable-dev-shm-usage")

        # options.add_argument("--disable-gpu")
        # options.add_argument("--disable-software-rasterizer")
        # options.add_argument("--disable-extensions")
        # options.add_argument("--disable-infobars")
        # options.add_argument("--disable-browser-side-navigation")
        # options.add_argument("--disable-application-cache")
        # options.add_argument("--disable-popup-blocking")
        # options.add_argument("--disable-translate")
        # options.add_argument("--disable-default-apps")
        # options.add_argument("--disable-client-side-phishing-detection")
        # options.add_argument("--start-maximized")


        #uncomment these for ccookie session
        options.add_argument("user-data-dir=/home/aryan/selenium-profile")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")
        
        driver = webdriver.Chrome(options=options)
        driver.get(f"https://www.linkedin.com/search/results/content/?keywords=%22hiring%22%20%26%20%22remote%22%20%26%20%22AI%22%20%26%20%22{job_title}%22&origin=GLOBAL_SEARCH_HEADER&sid=E_J")

        # Login to LinkedIn (Manually recommended for 2FA)
        try:
            username = driver.find_element(By.ID, "username")
            password = driver.find_element(By.ID, "password")
            wait = WebDriverWait(driver, 10)
        
            
          
            password.send_keys(Keys.RETURN)
            time.sleep(5)  # Wait for login to complete

            # Wait for home page to load
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(3)
        except:
            pass

        # # Click the search bar to activate it
        # search_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "search-global-typeahead__collapsed-search-button")))
        # search_button.click()

        # # Wait for search input to appear
        # search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.search-global-typeahead__input")))
        # search_box.send_keys("Data Science")
        # search_box.send_keys(Keys.RETURN)

        # # Wait for results to load
        # time.sleep(5)

        # driver.get(driver.current_url + "&type=posts")
        # time.sleep(5)

        # Scroll down multiple times to load more posts
        for _ in range(5):  
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

        # Expand all "See more" buttons to get full posts
        see_more_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'more')]")
        for button in see_more_buttons:
            try:
                driver.execute_script("arguments[0].click();", button)
                time.sleep(1)  # Give time to expand
            except:
                pass  # Ignore if already expanded

        # Extract full post contents
        posts = driver.find_elements(By.CSS_SELECTOR, ".feed-shared-update-v2__description")
        to_ret = []
        for index, post in enumerate(posts[:quantity]):  
            print(f"\n🔹 Post {index + 1}:\n{post.text}\n{'-'*50}")
            to_ret.append(post.text)

        driver.quit()
        return to_ret

    async def execute(self):
        action = self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")
        

from tools.reminders import Reminder, create_reminder,delete_reminder

class GOOGLECALENDAR:
    def __init__(self, api_key, kwargs,model,llm,user_id):
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)
        self.user_id= user_id
        self.kwargs = kwargs

    def clean_llm_json_response(self,raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)

    async def find_free_slot(self, start_time: str, end_time: str, matching_repeating_events: list):
        busy_response = await self.get_events(timeMin=start_time, timeMax=end_time)

        busy_intervals = []
        for event in busy_response:
            start = event['start time']
            end = event['end time']
            busy_intervals.append((start, end))
        for event in matching_repeating_events:
            start = event['start time']
            end = event['end time']
            busy_intervals.append((start, end))
        # Sort busy intervals by start time (first convert string times to datetime objects)
        busy_intervals_dt = []
        to_remove={}
        for start, end in busy_intervals:
            try:
                # Check if start/end are dictionaries and extract dateTime if they are
                if isinstance(start, dict):
                    start = start.get('dateTime', '')
                if isinstance(end, dict):
                    end = end.get('dateTime', '')
                    
                # Try isoparse first, fall back to fromisoformat if that fails
                try:
                    start_dt = parser.isoparse(start)
                except:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+05:30'))
                
                try:
                    end_dt = parser.isoparse(end)
                except:
                    end_dt = datetime.fromisoformat(end.replace('Z', '+05:30'))


                # Check if this event spans multiple days (crosses midnight)
                if start_dt.date() != end_dt.date():
                    to_remove = {"start": start_time, "end": (end_dt - timedelta(days=1)).isoformat()}  # Mark events that cross midnight
                
                busy_intervals_dt.append((start_dt, end_dt))
            except Exception as e:
                print(f"Error parsing date: {e}")
                continue
        # Sort by start time
        busy_intervals_dt.sort(key=lambda x: x[0])
        
        # Convert input times to datetime objects
        start_time_dt = parser.isoparse(start_time)
        end_time_dt = parser.isoparse(end_time)
        
        # Find free slots
        free_slots = []
        current = start_time_dt
        
        # Add free slots between busy intervals
        for busy_start, busy_end in busy_intervals_dt:
            if current < busy_start:
                free_slots.append({
                "start": current.isoformat(),
                "end": busy_start.isoformat()
            })
            current = max(current, busy_end)
            
            
        
        # Add final free slot if there's time after the last busy interval
        if current < end_time_dt:
            free_slots.append({
            "start": current.isoformat(),
            "end": end_time_dt.isoformat()
            })

        if to_remove:
            print("Removing slot that crosses midnight:", to_remove)

            free_slots = [slot for slot in free_slots if slot!= to_remove]
            
        return free_slots

    async def get_events(self,timeMax:str,timeMin:str):
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        
        response1 = composio_toolset.execute_action(params={
            "timeMin": timeMin,
            "timeMax": timeMax,
            "max_results":100
        }, action='GOOGLECALENDAR_FIND_EVENT')
        events=[]
        for i in response1['data']['event_data']['event_data']:
            print(i.keys())
            data={}
            data['summary'] = i.get('summary', '')
            data['start time']= i['start']['dateTime']
            data['end time']= i.get('end', {}).get('dateTime', i['start'].get('dateTime'))
            data['calendar id'] = i['id']
            events.append(data)
        print(events)
        return events
    
    async def ALL_ACTIONS(self, query: str):
        """
        Processes a user's calendar-related query and performs the required action: creating an event and reminders, deleting events, retrieving events or finding free time slots.
        Args:
            query (str): The user's input query describing their calendar-related intent. (please include current time and date in the query, also include the time range if user mentions any)
        Returns:
            str: Output or status message based on the action performed.
        """

        TASK="""
You are a calendar agent that processes user queries related to calendar events.
You will be given a query and you need to classify the action to be performed.
The actions can be one of the following:
1. create_event - Create a new event based on the user's query.
2. delete_event - Delete an existing event based on the user's query.
3. reschedule_or_update_event - Reschedule or update an existing event based on the user's query.
4. fetch_events - Fetch events based on the user's query.
5. find_free_time_slots - Find free time slots based on the user's query.

If user say to convert a non repetitive event to repetitive event, then route the user to create_event

Return only one string representing the action to be performed out of the above options - create_event, delete_event, reschedule_or_update_event, fetch_events, find_free_time_slots.
Do not include any preambles or postambles.
"""
        TASK += f"\nUser Query: {query}\n"
        try:
            response = await generate_con(uid=self.user_id, model=self.model, inside=TASK)
        except Exception as e:
            error_str = str(e)
            print(f"Error occurred: {error_str}")
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = await generate_con(uid=self.user_id, model=self.model, inside=TASK)
            else:
                raise e
            
        response = response.strip()
        print("Response from LLM:", response)


        EVENTS=f'''you are a calender agent that looks into user's query and find out the time range for which the user is interested in , and want to perform any action on calendar like create, delete, reschedule, update, fetch events.
            expected output is a json as
            {{
            "timeMin": start time with format string datetime in ISO 8601 format  with +05:30 for indian time zone,
            "timeMax": end time with format string datetime in ISO 8601 format  with +05:30 for indian time zone
            }}
            no preamble , no postamble , no explainantion just the json
            
            IMPORTANT :     
            Current indian date and time (YYYY-MM-DDTHH:MM:SS) is : {(datetime.now(pytz.timezone('Asia/Kolkata'))).isoformat()}
            IF USER HAS NOT MENTIONED ANY TIME RANGE IN THE QUERY, THEN KEEP THE RANGE "STARTING FROM START OF JUST PREVIOUS DAY" AND "ENDING TO 2 DAYS FROM NOW" , 
            IF THE USER HAS MENTIONED A TIME RANGE IN THE QUERY, "TODAY" OR "TOMORROW" OR "YESTERDAY" OR SOME_SPECIFIC_RANGE,  THEN USE THAT TIME RANGE AS IT IS            
            IF USER WANT TO FIND FREE TIME SLOTS, THEN TIME RANGE MUST BE ONE FULL DAY . USER MAY MENTION TODAY, TOMORROW, YESTERDAY AND RESPECTIVELY THE TIME RANGE WILL BE STARTING FROM START OF THE DAY AND ENDING AT END OF THE DAY (no matter what time range user has mentioned in the day)


            QUERY : {query}
            '''
        
        try:
            response_events = get_chain(uid=self.user_id,prompt=self.llm,inside=EVENTS,mo="gemini-2.0-flash",path=1)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response_events = get_chain(uid=self.user_id,prompt=self.llm,inside=EVENTS,mo="gemini-2.0-flash",path=1)
            else:
                raise e
        params=self.clean_llm_json_response(response_events.content)
        print(params)
        
        schedule=await self.get_events(timeMax=params['timeMax'], timeMin=params['timeMin'])
        response_data = "SINGLE EVENTS (NON REPEATING):\n"
        for i in schedule:
            try:
                # Extract only the time part in HH:MM:SS format from ISO 8601 datetime
                start_time = i['start time']
                end_time = i['end time']
                calendar_id = i['calendar id']
                # Handle cases where dateTime might not be present (e.g., all-day events)
                # def extract_time(dt):
                #     if 'T' in dt:
                #         t = dt.split('T')[1]
                #         return t.split('+')[0].split('Z')[0]  # Remove timezone if present
                #     return dt
                # start_time_str = extract_time(start_time)[:8]
                # end_time_str = extract_time(end_time)[:8]
                i["summary"]=i['summary']+"(Calendar_id:"+calendar_id+")"
                response_data += f"{i['summary']}, Start Time: {start_time}, End Time: {end_time}\n"
            except :
                response_data+= "due to some technical event the query was not processed successfully, please try again"
        
        
        try:
            # Fetch user's repeating events from the database
            user_data = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': self.user_id}}
            )
            
            repeating_events = []
            if 'Item' in user_data and 'repeating_events' in user_data['Item']:
                # Parse the repeating events JSON
                repeating_events = json.loads(user_data['Item']['repeating_events']['S'])
            
            # Convert time range parameters to datetime objects for comparison
            # Parse input times and convert to Python datetime objects
            # For India timezone (IST = UTC+5:30)
            start_range = datetime.fromisoformat(params['timeMin'].replace('Z', '+05:30'))
            end_range = datetime.fromisoformat(params['timeMax'].replace('Z', '+05:30'))
            
            # Find repeating events that fall within the given time range
            matching_repeating_events = []
            for event in repeating_events:
                # Parse the event start time
                event_start = datetime.fromisoformat(event['start_datetime'].replace('Z', '+05:30'))
                event_duration = timedelta(
                    hours=event.get('event_duration_hour', 0),
                    minutes=event.get('event_duration_minutes', 0)
                )
                interval_hours = event.get('interval', 24)  # Default to daily if not specified
                
                # Calculate how many repetitions have occurred since the event started
                hours_since_start = (start_range - event_start).total_seconds() / 3600
                
                # If the event starts after our range begins, adjust hours_since_start
                if hours_since_start < 0:
                    first_occurrence_in_range = event_start
                else:
                    # Calculate the first occurrence in range
                    repetitions_before_range = int(hours_since_start / interval_hours)
                    first_occurrence_in_range = event_start + timedelta(hours=repetitions_before_range * interval_hours)
                    
                    # If we overshot the start of our range, go to the next occurrence
                    if first_occurrence_in_range < start_range:
                        first_occurrence_in_range += timedelta(hours=interval_hours)
                
                # Now check all occurrences within the range
                current_occurrence = first_occurrence_in_range
                while current_occurrence < end_range:
                    event_end = current_occurrence + event_duration
                    
                    # Add this occurrence to the results
                    matching_repeating_events.append({
                        'summary': event['name']+f" SiGmOyD(Reminder ID: {event.get('reminder_id', '')})",
                        'start time': {'dateTime': current_occurrence.isoformat()},
                        'end time': {'dateTime': event_end.isoformat()},
                        "interval": interval_hours
                  
                    })
                    break
                    # Move to the next occurrence
                    current_occurrence += timedelta(hours=interval_hours)
            
            # Append the matching repeating events to the response
            response_data += "\n\nRepeating Events:\n"
            for event in matching_repeating_events:
                start_time = event['start time']['dateTime']
                end_time = event['end time']['dateTime']
                # start_time_str = start_time.split('T')[1][:8] if 'T' in start_time else start_time
                # end_time_str = end_time.split('T')[1][:8] if 'T' in end_time else end_time
                response_data += f"{event['summary']}, Start Time: {start_time}, End Time: {end_time}\ninterval_hours_between_repetitions: {event['interval']}\n"
        except Exception as e:
            print(f"Error fetching repeating events: {e}")

        if response_data == "SINGLE EVENTS (NON REPEATING):\n\n\nRepeating Events:\n":
            response_data="No events found for the given time range."
        

        print("existing events found", response_data)
        if response == "create_event":
            TASK=f'''you are a calender agent that looks into user's query and returns a list of events user wants to create.
            Each event should be a dictionary with the following keys:
            - "start_datetime": The start time of the event in ISO 8601 format with timezone.
            - "summary": A brief description of the event. (The summary must contain relatable emojis too).
            - "event_duration_hour": The integer duration of the event in hours (0-24).
            - "event_duration_minutes": The integer duration of the event in minutes (0-59).
            - "interval": The interval in hours (integer) for repeating events (optional. default is 0).
            
            if user don't specify the duration for any task then that dictionary should not contain the keys "event_duration_hour" and "event_duration_minutes".
            If user mentions words like, perform any task every monday, or everyday or every 2 hours, then you should add the key "interval" with the value as the interval in hours (integer) for repeating events. interval should be 0  if the event is not repeating.


            Return a list of dictionaries, each representing an event to be created.
            Example output:
            [
                {{
                    "start_datetime": "2023-10-01T10:00:00+05:30",  (default time zone is Indian Standard Time)
                    "summary": "🏋🏻GYM",  (should contain an emoji)
                    "event_duration_hour": 1,
                    "event_duration_minutes": 30,
                    "interval": 0  
                    
                }},
                ....
            ]

            no preamble , no postamble , no explainantion just the list of dictionaries as described above, keep all strings in double quotes.
            QUERY : {query}
            '''
            try:
                response_list = await generate_con(uid=self.user_id, model=self.model, inside=TASK)
            except Exception as e:
                error_str = str(e)
                print(f"Error occurred: {error_str}")
                if "429" in error_str and "retry_delay" in error_str:
                    # Extract retry delay seconds
                    retry_seconds = 60  # Default fallback
                    # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    # if retry_match:
                    #     retry_seconds = int(retry_match.group(1))
                    
                    print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                    # Retry the request
                    response_list = await generate_con(uid=self.user_id, model=self.model, inside=TASK)
                else:
                    raise e

            analysis_text = response_list.strip()
            parsed_events = self.clean_llm_json_response(analysis_text)
            print("Parsed Events:", parsed_events)
            for event in parsed_events:
                

                if not (event.get("interval",None)) or event["interval"]==0:
                    if event.get("interval",None):
                        del event["interval"]
                    start_time_dt = datetime.fromisoformat(event["start_datetime"].replace('Z', '+00:00'))
                    reminder_time = start_time_dt - timedelta(minutes=5)
                    rem=Reminder(
                        user_id=self.user_id,
                        start_time=reminder_time.isoformat(),
                        message=event["summary"],
                    )
                    rid=create_reminder(rem)["reminder_id"]
                    event["summary"]+=" SiGmOyD(Reminder ID: "+rid+")"
                    event["timezone"]="IST"
                    print(event)
                    composio_toolset = ComposioToolSet(api_key=self.api_key)
                    response1 = composio_toolset.execute_action(params=event, action='GOOGLECALENDAR_CREATE_EVENT')
                    if response1['successfull'] == True or 'true':
                        print("event added successfully")
                    else:
                        print("event not added successfully, please try again later or check the query you provided")
                else:
                    start_time_dt = datetime.fromisoformat(event["start_datetime"].replace('Z', '+00:00'))
                    reminder_time = start_time_dt - timedelta(minutes=5)
                    rem=Reminder(
                        user_id=self.user_id,
                        start_time= reminder_time.isoformat(),
                        message=event["summary"] ,
                        repeat_interval_hours= event.get("interval", 24),
                    )
                    rid=create_reminder(rem)["reminder_id"]
                    event["event_duration_hour"] = int(event.get("event_duration_hour", 0))
                    event["event_duration_minutes"] = int(event.get("event_duration_minutes", 0))
                    event["interval"] = int(event.get("interval", 24))  # Default to daily if not specified
                    event["name"] = event["summary"] +"\n" +event.get("description", "")
                    event["reminder_id"] = rid

                    # Get current user data
                    user_data = db_client.get_item(
                        TableName='users',
                        Key={'clerk_id': {'S': self.user_id}}
                    )

                    # Initialize repeating events if not exists
                    repeating_events = []
                    if 'Item' in user_data and 'repeating_events' in user_data['Item']:
                        try:
                            repeating_events = json.loads(user_data['Item']['repeating_events']['S'])
                        except:
                            repeating_events = []

                    # Add the new repeating event
                    repeating_events.append({
                        'name': event["name"],
                        'start_datetime': event["start_datetime"],
                        'event_duration_hour': event["event_duration_hour"],
                        'event_duration_minutes': event["event_duration_minutes"],
                        'interval': event["interval"],
                        'reminder_id': rid
                    })

                    # Update the user record in DynamoDB
                    db_client.update_item(
                        TableName='users',
                        Key={'clerk_id': {'S': self.user_id}},
                        UpdateExpression='SET repeating_events = :re',
                        ExpressionAttributeValues={
                            ':re': {'S': json.dumps(repeating_events)}
                        }
                    )

                    # # Add to Google Calendar
                    # event["summary"] += " (Reminder ID: " + rid + ")"
                    # event["timezone"] = "IST"
                    # composio_toolset = ComposioToolSet(api_key=self.api_key)
                    # response = composio_toolset.execute_action(params=event, action='GOOGLECALENDAR_CREATE_EVENT')
                    # if response['successfull'] == True or response['successfull'] == 'true':
                    #     print("Repeating event added successfully to calendar")
                    # else:
                    #     print("Failed to add repeating event to calendar")
            if response_data == "No events found for the given time range.":
                response_data = "No pre-existing repeating events found in the given time range."
                return f"New Events created successfully : {parsed_events}"
            else:
                return f"New Events created successfully : {parsed_events}"




        elif response == "fetch_events":
            # return {"events_in_given_duration":response_data}
            task="""
Your task is to arrange the events of user's calender in a chronological order based on the start time of the events.
kindly merge the repeating and non repeating events together in the same list, and sort them by start time.
Return a clean and readable string with the events arranged in chronological order, change the isoformat time to a more readable format like "3:45 pm" for start time and end time ,and show duration of the event in hours and minutes instead of end time
please remove the calendar id and reminder id from the output, and just show the event name, start time, end time and duration in hours and minutes. (show the repeating - like repeats daily, weekly, etc. if applicable)
If the events of multiple days are present, then separate them by date, and show the date in the format "DD-MM-YYYY" followed by the events of that day.
No preambles and postambles are required, just return the string with the events arranged in chronological order.
"""
            task += f"\n\nEvents in the given time range are:\n{response_data}\n\n"
            try:
                response_list = await generate_con(uid=self.user_id, model=self.model, inside=task)
            except Exception as e:
                error_str = str(e)
                print(f"Error occurred: {error_str}")
                if "429" in error_str and "retry_delay" in error_str:
                    # Extract retry delay seconds
                    retry_seconds = 60  # Default fallback
                    # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    # if retry_match:
                    #     retry_seconds = int(retry_match.group(1))
                    
                    print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                    # Retry the request
                    response_list = await generate_con(uid=self.user_id, model=self.model, inside=task)
                else:
                    raise e
            analysis_text = response_list.strip()
            print("Response from LLM for fetch_events:", analysis_text)
            return analysis_text
        elif response == "delete_event":
            TASK=f"""you are a calender agent that looks into user's query and the already existing reminders/events and returns a list of reminder ids and calendar ids that user wants to delete.
            Each id should be a string.
            Return a list of jsons, each representing an event that the user wants to delete.
            Example output:
            [
                {{"name":"Event 1","type":"repeating","reminder_id": "reminder_id_1"}},
                {{"name":"Event 2","type":"simple","reminder_id": "reminder_id_2", "calendar_id": "calendar_id_2"}},
                ...
            ]
            no preamble , no postamble , no explainantion just the list of jsons as described above, keep all strings in double quotes.

            Already existing reminders/events in the given time range are:
            {response_data}

            IMPORTANT : 
            FOR REPEATING EVENTS, THERE IS NO CALENDAR ID, SO DON'T INCLUDE CALENDAR ID FOR REPEATING EVENTS IN THE OUTPUT JSON. BUT MUST INCLUDE THEM FOR SIMPLE EVENTS.
            REMINDER ID IS AVAILABLE FOR BOTH REPEATING AND SIMPLE EVENTS, SO INCLUDE THEM IN EACH OUTPUT JSON.

            QUERY : {query}
            """

            try:
                response_list = await generate_con(uid=self.user_id, model=self.model, inside=TASK)
            except Exception as e:
                error_str = str(e)
                print(f"Error occurred: {error_str}")
                if "429" in error_str and "retry_delay" in error_str:
                    # Extract retry delay seconds
                    retry_seconds = 60  # Default fallback
                    # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    # if retry_match:
                    #     retry_seconds = int(retry_match.group(1))
                    
                    print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                    # Retry the request
                    response_list = await generate_con(uid=self.user_id, model=self.model, inside=TASK)
                else:
                    raise e
            print("Response from LLM for delete_event:", response_list.strip())
            analysis_text = response_list.strip()
            parsed_events = self.clean_llm_json_response(analysis_text)
            print("Parsed Events to Delete:", parsed_events)
            to_ret=""
            for event in parsed_events:
                if event["type"]=="repeating":
                    # Delete the repeating event from the database
                    reminder_id = event.get("reminder_id")
                    if reminder_id:
                        delete_reminder(reminder_id)
                        # Remove the repeating event from the user's repeating events in the database
                        user_data = db_client.get_item(
                            TableName='users',
                            Key={'clerk_id': {'S': self.user_id}}
                        )
                        
                        if 'Item' in user_data and 'repeating_events' in user_data['Item']:
                            repeating_events = json.loads(user_data['Item']['repeating_events']['S'])
                            repeating_events = [e for e in repeating_events if e['reminder_id'] != reminder_id]
                            print(f"Updated repeating events: {repeating_events}")
                            db_client.update_item(
                                TableName='users',
                                Key={'clerk_id': {'S': self.user_id}},
                                UpdateExpression='SET repeating_events = :re',
                                ExpressionAttributeValues={
                                    ':re': {'S': json.dumps(repeating_events)}
                                }
                            )
                    print(f"Repeating event {event['name']} deleted successfully.")
                    to_ret+=f"Repeating event {event['name']} deleted successfully.\n"
                elif event["type"]=="simple":
                    # Delete the simple event from the database
                    reminder_id = event.get("reminder_id")
                    if reminder_id:
                        delete_reminder(reminder_id)
                    # Delete the event from Google Calendar using the calendar ID
                    calendar_id = event.get("calendar_id")
                    if calendar_id:
                        composio_toolset = ComposioToolSet(api_key=self.api_key)
                        response = composio_toolset.execute_action(params={"event_id": calendar_id}, action='GOOGLECALENDAR_DELETE_EVENT')
                        if response['successfull'] == True or response['successfull'] == 'true' or response["successful"]==True or response["successful"]=='true':
                            print(f"Simple event {event['name']} deleted successfully from Google Calendar.")
                            to_ret+=f"Simple event {event['name']} deleted successfully from Google Calendar.\n"
                        else:
                            print(f"Failed to delete simple event {event['name']} from Google Calendar.")
                            to_ret+=f"Failed to delete simple event {event['name']} from Google Calendar.\n"
            return to_ret if to_ret else "No events deleted successfully. Please check the query and try again."
        elif response == "reschedule_or_update_event":
            TASK=f"""
- you are a calender agent that looks into user's query and the already existing reminders/events and returns a list of reminder ids and calendar ids that user wants to reschedule or update.
- You also need to determine the new start time, duration, description, and interval (if any) for each of the event user wants to reschedule or update.
- Find out what parameters user wants to update, and what all parameters already present in the event. Update only those parameters which user wants to update, and keep the rest of the parameters same as they are already present in the event.
- Find out event duration by looking at start time and end time of the event

Each event should be a dictionary with the following keys:
    - "reminder_id" : reminder id of event
    - "calendar_id" : calendar id of event  (only for non-repeating events)
    - "start_datetime": The start time of the event in ISO 8601 format with timezone.
    - "summary": A brief description of the event.
    - "event_duration_hour": The integer duration of the event in hours (0-24).
    - "event_duration_minutes": The integer duration of the event in minutes (0-59).
    - "interval": The interval in hours (integer) for repeating events (optional. default is 0).
    

    If user mentions words like, perform any task every monday, or everyday or every 2 hours, then you should add the key "interval" with the value as the interval in hours (integer) for repeating events. interval should be 0  if the event is not repeating.
    

    Return a list of dictionaries, each representing an event to be updated or rescheduled.
    Example output:
    [
        {{
            "reminder_id": "reminder_id_1",
            "calendar_id": "calendar_id_1",  (only for non-repeating events),
            "start_datetime": "2023-10-01T10:00:00+05:30",  (default time zone is Indian Standard Time)
            "summary": "Team Meeting",
            "event_duration_hour": 1,
            "event_duration_minutes": 30,
            "interval": 0  
            
        }},
        ....
    ]

    no preamble , no postamble , no explainantion just the list of dictionaries as described above, keep all strings in double quotes.

    Already existing reminders/events in the given time range are:
    {response_data}

    USER'S QUERY : {query}

"""
            try:
                response_list = await generate_con(uid=self.user_id, model=self.model, inside=TASK)
            except Exception as e:
                error_str = str(e)
                print(f"Error occurred: {error_str}")
                if "429" in error_str and "retry_delay" in error_str:
                    # Extract retry delay seconds
                    retry_seconds = 60  # Default fallback
                    # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    # if retry_match:
                    #     retry_seconds = int(retry_match.group(1))
                    
                    print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                    # Retry the request
                    response_list = await generate_con(uid=self.user_id, model=self.model, inside=TASK)
                else:
                    raise e

            analysis_text = response_list.strip()
            parsed_events = self.clean_llm_json_response(analysis_text)
            print("Parsed Events to Update or Reschedule:", parsed_events)
            for event in parsed_events:
                

                if event["interval"]==0:
                    del event["interval"]
                    delete_reminder(event["reminder_id"])
                    event["event_id"] = event["calendar_id"]
                    del event["calendar_id"]
                    start_time_dt = datetime.fromisoformat(event["start_datetime"].replace('Z', '+00:00'))
                    reminder_time = start_time_dt - timedelta(minutes=5)
                    rem=Reminder(
                        user_id=self.user_id,
                        start_time= reminder_time.isoformat(),
                        message=event["summary"] ,
                    )
                    rid=create_reminder(rem)["reminder_id"]
                    event["summary"]+=" SiGmOyD(Reminder ID: "+rid+")"
                    # event["timezone"]="IST"
                    print(event)
                    composio_toolset = ComposioToolSet(api_key=self.api_key)
                    response1 = composio_toolset.execute_action(params=event, action='GOOGLECALENDAR_UPDATE_EVENT')
                    if response1['successfull'] == True or 'true':
                        print("event updated successfully")
                    else:
                        print("event not updated successfully, please try again later or check the query you provided")
                else:
                    delete_reminder(event["reminder_id"])
                    prev_rid=event["reminder_id"]
                    start_time_dt = datetime.fromisoformat(event["start_datetime"].replace('Z', '+00:00'))
                    reminder_time = start_time_dt - timedelta(minutes=5)
                    rem=Reminder(
                        user_id=self.user_id,
                        start_time= reminder_time.isoformat(),
                        message=event["summary"],
                        repeat_interval_hours= event.get("interval", 24),
                    )
                    rid=create_reminder(rem)["reminder_id"]
                    event["event_duration_hour"] = int(event.get("event_duration_hour", 0))
                    event["event_duration_minutes"] = int(event.get("event_duration_minutes", 0))
                    event["interval"] = int(event.get("interval", 24))  # Default to daily if not specified
                    event["name"] = event["summary"] +"\n" +event.get("description", "")
                    event["reminder_id"] = rid

                    # Get current user data
                    user_data = db_client.get_item(
                        TableName='users',
                        Key={'clerk_id': {'S': self.user_id}}
                    )

                    # Initialize repeating events if not exists
                    repeating_events = []
                    if 'Item' in user_data and 'repeating_events' in user_data['Item']:
                        try:
                            repeating_events = json.loads(user_data['Item']['repeating_events']['S'])
                        except:
                            repeating_events = []

                    # Remove the old repeating event if it exists
                    repeating_events = [e for e in repeating_events if e['reminder_id'] != prev_rid]
                    # Add the new repeating event
                    repeating_events.append({
                        'name': event["name"],
                        'start_datetime': event["start_datetime"],
                        'event_duration_hour': event["event_duration_hour"],
                        'event_duration_minutes': event["event_duration_minutes"],
                        'interval': event["interval"],
                        'reminder_id': rid
                    })

                    # Update the user record in DynamoDB
                    db_client.update_item(
                        TableName='users',
                        Key={'clerk_id': {'S': self.user_id}},
                        UpdateExpression='SET repeating_events = :re',
                        ExpressionAttributeValues={
                            ':re': {'S': json.dumps(repeating_events)}
                        }
                    )
            return "events updated successfully"
        elif response == "find_free_time_slots":

            return await self.find_free_slot(params['timeMin'], params['timeMax'],matching_repeating_events)


    async def execute(self):
        action = self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")

from tools.md_to_pdf import markdown_to_pdf

class REPORT_MAKER:
    def __init__(self, api_key, kwargs,model,llm,user_id):
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.kwargs=kwargs
        self.user_id=user_id
    def extract_and_clean_markdown(self,text: str):
        """
        Extracts markdown from a code block (```markdown) and converts LLM-style formatting.
        
        Args:
            text (str): The raw input text from the LLM, including ```markdown.
            underline_html (bool): Whether to preserve underline with HTML.
        
        Returns:
            str: Functional, clean markdown.
        """
        
        # Step 1: Extract markdown inside the ```markdown ... ``` block
        markdown_block = re.search(r"```markdown\n(.*?)```", text, re.DOTALL)
        if markdown_block:
            markdown_content = markdown_block.group(1)
        else:
            markdown_content = text  # Fallback: assume input is raw markdown

        

        return markdown_content
      
    @staticmethod
    def get_id_from_url(url):
        match = re.search(r'/d/([a-zA-Z0-9_-]+)/', url)
        return match.group(1) if match else None  
   
        
    async def rule_maker(self,task:str,format:str=None):

        if format==None:

            prompt = f'''you are a simple style rule maker that makes the styling rules for the markdown text in order to execute the task : {task}.
                        expected output is string list of the rules.
                        the rules must contain :
                        - what section must be bold and what section must not 
                        - where should a paraph change and line ends 
                        - next line indicators 
                        - Heading
                        the rules can contain following depending upon condition :
                        - bulletien list 
                        - link and image 
                        
                        no preambles , no postambles and no explainantion is needed.  

            '''
        else:
            prompt=f'''find the markdown styling rules in the for {format} and list them one by one.
                        expected output is the string list of rules 
                         no preambles , no postambles and no explainantion is needed. '''

        try:
            rules_response = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                rules_response = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)
            else:
                raise e
        rules = rules_response.content
        return rules


    async def google_doc_markdown_maker(self,rules:str,content:str,document_id:str=None):
        prompt=f'''you are a markdown content maker that makes the markdow text for the {content} by following the rules as {rules}
                expected output is the markdown of the {content} which is following the rules : {rules}
                no preambles , no postambles and no explainantion is needed '''
        try:
            json_maker = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)
            print("json maker : ",json_maker)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                json_maker = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)
            else:
                raise e
        print(self.extract_and_clean_markdown(json_maker.content))

        if document_id==None:
            params={'markdown_text':self.extract_and_clean_markdown(json_maker.content),'title':'USER INPUT'}
            composio_toolset = ComposioToolSet(api_key=self.api_key)
            response=composio_toolset.execute_action(params=params,action = 'GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN' )
            return {'document_url': "https://docs.google.com/document/d/"+str(response['data']['document_id'])}
        
        elif document_id!=None:
            param={'document_id':document_id,'new_markdown_text':self.extract_and_clean_markdown(json_maker.content)}
            composio_toolset = ComposioToolSet(api_key='wfaixhni71caogru03zu7a')
            response=composio_toolset.execute_action(params=param,action = 'GOOGLEDOCS_UPDATE_DOCUMENT_MARKDOWN' )
            return response['successfull']
    
    async def ALL_ACTION(self,task:str,content:str,format:str=None,document_url:str=None):
        """
        report generator that generates the report / invoices / documentation on google docs. In beautiful markdown format.
        Parameters:
        ----------
        task : str
            The type of processing task to be performed on the content
        content : str
            The text content to be processed for the report 
        format : str, optional
            Specified styling and formating template ( if any , default is none)
        document_url : str, optional
            URL of a Google Document to be used as reference (default is None)
        Returns:
        -------
        boolean true or false 
   
        """
        if document_url==None:
            rules=await self.rule_maker(task=task,format=format)
            return await self.google_doc_markdown_maker(rules=rules,content=content)
        else:
            id=self.get_id_from_url(document_url)
            rules=await self.rule_maker(task=task)
            return await self.google_doc_markdown_maker(rules=rules,document_id=id,content=content)
        

    async def PDF_REPORT_MAKER(self,md_text):
        
        """
        Generate a PDF report from markdown text.
        This method converts markdown text into a PDF document, suitable for creating reports, invoices, or documentation with professional formatting.
        Parameters:
            md_text (str): The markdown text content to be converted into a PDF.
        Returns:
            Pdf link (str): The downloadable link of generated pdf
        """
        
        try:
            # Create a temporary file for the PDF
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_path = temp_file.name
            temp_file.close()
            
            # Convert markdown to PDF
            markdown_to_pdf(md_text, temp_path)
            
            # Upload the PDF to S3
            bucket_name = "gmail-attachments-2709"
            output_filename = f"report_{uuid.uuid4()}.pdf"
            
            s3_client.upload_file(temp_path, bucket_name, output_filename)
            
            # Generate a presigned URL for downloading the PDF
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': output_filename
                },
                ExpiresIn=604800  # URL expires in 1 week (7 days)
            )
            
            # Return both the S3 path and the download link
            return {
                "s3_path": f"s3://{bucket_name}/{output_filename}",
                "download_link": presigned_url
            }
        
        except Exception as e:
            print(f"Error creating PDF report: {e}")
            return {"error": str(e)}
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)


    async def execute(self):                       #LLM will only call execute
        action=self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")
        

######

######
class GOOGLEDOCS:
    
    def __init__(self,api_key,kwargs,model,llm,user_id):
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.kwargs=kwargs
        self.model=model
        self.user_id=user_id
    @staticmethod
    def get_id_from_url(url):
        match = re.search(r'/d/([a-zA-Z0-9_-]+)/', url)
        return match.group(1) if match else None  


    def extract_and_clean_markdown(text: str, underline_html=False):
        """
        Extracts markdown from a code block (```markdown) and converts LLM-style formatting.
        
        Args:
            text (str): The raw input text from the LLM, including ```markdown.
            underline_html (bool): Whether to preserve underline with HTML.
        
        Returns:
            str: Functional, clean markdown.
        """
        
        # Step 1: Extract markdown inside the ```markdown ... ``` block
        markdown_block = re.search(r"```markdown\n(.*?)```", text, re.DOTALL)
        if markdown_block:
            markdown_content = markdown_block.group(1)
        else:
            markdown_content = text  # Fallback: assume input is raw markdown

        

        return markdown_content
        
                
    async def GOOGLEDOCS_CREATE_DOCUMENT(self,title:str='this is the title of the document',text:str='this is the text of the document'):
        """
        Creates a new Google Document with the specified title and text content.
        Parameters:
        ----------
        title : str
            The title of the Google Document.
            Example: "Project Proposal"
        text : str
            The text content to be included in the document.
            Example: "This is a sample project proposal document."
        Returns:
        -------
        dict: A dictionary containing the success status of the document creation.  

        """
        param={"title":title,"text":text}
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        
        
        response=composio_toolset.execute_action(params=param,action = 'GOOGLEDOCS_CREATE_DOCUMENT' )
        return {'success': response['successfull'] , 'document url':"https://docs.google.com/document/d/"+response['data']['document_id']+"/edit?usp=sharing"}


    async def content(self,task:str,orininal_content:str):
        prompt=f'''you are a precise content creator agent , you job is to create a content for the {task} based on the {orininal_content} where every /n represent the next line so while creating content make a new line for every /n .
        maybe you might have to add , update or remove the content from {orininal_content} to create a content for the {task} . 

        expected output is a markdown format and make sure the content must be presentable , professional and easy to read . make the markdown in the best way possible . 
        no preambles, no postambles, no explanations, just return the string as described above'''
        try:
            response = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                response = get_chain(uid=self.user_id,prompt=self.llm,inside=prompt,mo="gemini-2.0-flash",path=1)
            else:
                raise e
        print("llm response :",response.content)
        return self.extract_and_clean_markdown(response.content.strip())


    async def GOOGLEDOCS_UPDATE_EXISTING_DOCUMENT(self, document_url: str = 'this is the url of the document', task: str = 'task which is supposed to be done on the document'):
        """
            Updates an existing Google Document by replacing its content with the provided text.

            Parameters:
            - document_url (str): The URL of the Google Document to update. Example: "https://docs.google.com/document/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ/edit".
            - task (str): A natural language description of how to update the content. if content needed to be added , then This should also contain the content to be inserted or modified in the document.
            

            Returns:
            - dict: A dictionary containing the success status of the update operation.
            """

        #content creation
        original_content = await self.GOOGLEDOCS_GET_DOCUMENT_BY_URL(document_url)
        editDocs = await self.content(task, original_content['data'])

        

        id = self.get_id_from_url(document_url)
        param = {"id": id}
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        response1 = composio_toolset.execute_action(params=param, action='GOOGLEDOCS_GET_DOCUMENT_BY_ID')
        





        response = await self.GOOGLEDOCS_GET_DOCUMENT_BY_URL(document_url)
        end_index_data = response1['data']['response_data']['body']['content'][-1]
        end_index = end_index_data['endIndex']
        document_id = self.get_id_from_url(document_url)
        if end_index-1==1:
            editdocs_content = [
                
                {
                    "insertText": {
                        "text": editDocs,
                        "location": {
                            
                            "index": 1
                        }
                    }
                }
            ]
        else:
            editdocs_content = [
                {
                    "deleteContentRange": {
                        "range": {
                            "startIndex": 1,
                            "endIndex": end_index -1
                        }
                    }
                },
                {
                    "insertText": {
                        "text": editDocs,
                        "location": {
                            "index": 1
                        }
                    }
                }
            ]

        param = {"document_id": document_id, "editDocs": editdocs_content}
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        response = composio_toolset.execute_action(params=param, action='GOOGLEDOCS_UPDATE_EXISTING_DOCUMENT')
        print(response)
        return {'success': response['successfull']}

    async def GOOGLEDOCS_GET_DOCUMENT_BY_URL(self, url: str = 'this is the url of the document'):
        """
        Retrieves the content of a Google Document using its URL.

        Parameters:
        - url (str): The URL of the Google Document to retrieve. Example: "https://docs.google.com/document/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ/edit".

        Returns:
        - dict: A dictionary containing the text content of the document.
        """
        id = self.get_id_from_url(url)
        param = {"id": id}
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        response = composio_toolset.execute_action(params=param, action='GOOGLEDOCS_GET_DOCUMENT_BY_ID')
        content = response['data']['response_data']['body']['content']
        data = ""
        for i in content:
            
            if i == 0:
                pass
            else:
                try:
                    data = data + i['paragraph']['elements'][0]['textRun']['content'] + "\n"
                except:
                    pass

        return {'data': data}
    
    async def REPORT_MAKER(self, task: str = 'this is the task for which report is to be made', content: str = 'this is the content of the report', format: str = None, document_url: str = None):
        """
         report generator that generates the report / invoices / documentation on google docs in beautiful markdown format, given some text or analysis or knowledge.
        Parameters:
        ----------
        task : str
            The type of processing task to be performed on the content
        content : str
            The text content to be processed for the report 
        format : str, optional
            Specified styling and formating template ( if any , default is none)
        document_url : str, optional
            URL of a Google Document to be used as reference (default is None)
        Returns:
        -------
        googledoc url of the documentation created  
        """
        report= REPORT_MAKER(api_key=self.api_key, kwargs={"action": "ALL_ACTION", "task": task, "content": content, "format": format, "document_url": document_url}, model=self.model, llm=self.llm,user_id=self.user_id)
        return await report.execute()
    async def execute(self):                       #LLM will only call execute
        action=self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")



# class YOUTUBE:
#     def __init__(self, api_key,kwargs):
#         self.api_key = api_key
#         self.kwargs = kwargs

#     def YOUTUBESEARCH(self, query, max_results=10, search_type="video"):
#         toolset = ComposioToolSet(api_key=self.api_key, entity_id="default")
#         data = toolset.execute_action(
#             action=Action.YOUTUBE_SEARCH_YOU_TUBE,
#             params={"q": query, "part": "snippet", "maxResults": max_results, "type": search_type},
#             entity_id="default"
#         )

#         items = data.get("data", {}).get("response_data", {}).get("items", [])
#         extracted_data = {
#             "status_code": 200,
#             "data": [
#                 {
#                     "video_id": item.get("id", {}).get("videoId"),
#                     "channel_id": item.get("snippet", {}).get("channelId"),
#                     "title": item.get("snippet", {}).get("title"),
#                     "description": item.get("snippet", {}).get("description"),
#                     "video_link": f"https://www.youtube.com/watch?v={item.get('id', {}).get('videoId')}",
#                     "publishedAt": item.get("snippet", {}).get("publishedAt"),
#                     "channelTitle": item.get("snippet", {}).get("channelTitle"),
#                     "thumbnail": item.get("snippet", {}).get("thumbnails", {}).get("high", {}).get("url"),
#                 }
#                 for item in items
#             ]
#         }
#         return json.dumps(extracted_data, indent=4)

#     def YOUTUBEVIDEODETAILS(self, video_id):
#         toolset = ComposioToolSet(api_key=self.api_key)
#         data = toolset.execute_action(
#             action=Action.YOUTUBE_VIDEO_DETAILS,
#             params={"id": video_id, "part": "snippet,contentDetails,statistics"},
#         )

#         items = data.get("data", {}).get("response_data", {}).get("items", [])
#         extracted_data = {
#             "status_code": 200,
#             "data": [
#                 {
#                     "id": item.get("id"),
#                     "title": item.get("snippet", {}).get("title"),
#                     "description": item.get("snippet", {}).get("description"),
#                     "video_link": f"https://www.youtube.com/watch?v={item.get('id')}",
#                     "publishedAt": item.get("snippet", {}).get("publishedAt"),
#                     "channelTitle": item.get("snippet", {}).get("channelTitle"),
#                     "thumbnail": item.get("snippet", {}).get("thumbnails", {}).get("high", {}).get("url"),
#                     "duration": item.get("contentDetails", {}).get("duration"),
#                     "viewCount": item.get("statistics", {}).get("viewCount"),
#                     "likeCount": item.get("statistics", {}).get("likeCount"),
#                     "commentCount": item.get("statistics", {}).get("commentCount"),
#                 }
#                 for item in items
#             ]
#         }
#         return json.dumps(extracted_data, indent=4)

#     def YOUTUBELISTUSERSUBSCRIPTIONS(self, max_results=10):
#         toolset = ComposioToolSet(api_key=self.api_key, entity_id="default")
#         data = toolset.execute_action(
#             action=Action.YOUTUBE_LIST_USER_SUBSCRIPTIONS,
#             params={"part": "snippet,contentDetails", "maxResults": max_results},
#             entity_id="default",
#         )

#         items = data.get("data", {}).get("response_data", {}).get("items", [])
#         extracted_data = {
#             "status_code": 200,
#             "data": [
#                 {
#                     "subscription_id": item.get("id"),
#                     "title": item.get("snippet", {}).get("title"),
#                     "description": item.get("snippet", {}).get("description"),
#                     "channel_id": item.get("snippet", {}).get("resourceId", {}).get("channelId"),
#                     "publishedAt": item.get("snippet", {}).get("publishedAt"),
#                     "thumbnail": item.get("snippet", {}).get("thumbnails", {}).get("high", {}).get("url"),
#                     "totalItemCount": item.get("contentDetails", {}).get("totalItemCount"),
#                     "newItemCount": item.get("contentDetails", {}).get("newItemCount"),
#                     "activityType": item.get("contentDetails", {}).get("activityType"),
#                 }
#                 for item in items
#             ]
#         }
#         return json.dumps(extracted_data, indent=4)

#     def YOUTUBELISTUSERPLAYLISTS(self, max_results=10):
#         toolset = ComposioToolSet(api_key=self.api_key)
#         response = toolset.execute_action(
#             action=Action.YOUTUBE_LIST_USER_PLAYLISTS,
#             params={"part": "snippet", "maxResults": max_results},
#         )

#         if response.get("successfull") and response.get("data"):
#             playlists = response["data"].get("response_data", {}).get("items", [])
#             useful_data = [
#                 {
#                     "id": playlist["id"],
#                     "title": playlist["snippet"]["title"],
#                     "description": playlist["snippet"]["description"],
#                     "thumbnail": playlist["snippet"]["thumbnails"].get("high", {}).get("url"),
#                 }
#                 for playlist in playlists
#             ]
#             return json.dumps(useful_data, indent=4)

#         return json.dumps([])

#     def YOUTUBESUBSCRIBECHANNEL(self, channel_id):
#         toolset = ComposioToolSet(api_key=self.api_key)
#         return json.dumps(toolset.execute_action(
#             action=Action.YOUTUBE_SUBSCRIBE_CHANNEL,
#             params={"channelId": channel_id},
#         ), indent=4)

#     def connect_youtube(self):
#         toolset = ComposioToolSet(api_key=self.api_key)
#         connection_request = toolset.initiate_connection(entity_id="default", app=App.YOUTUBE)
#         return json.dumps({"redirect_url": connection_request.redirectUrl}, indent=4)

#     def execute(self):
#         action=self.kwargs["action"]
#         del self.kwargs["action"]
        
#         if hasattr(self, action):  # Check if function exists
#             return getattr(self, action)(**self.kwargs)  # Call the function dynamically
#         else:
#             return print(f"Method {action} not found")

class GOOGLEMEET:

    def __init__(self,api_key,kwargs,model,llm,user_id):
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.kwargs=kwargs
        self.model=model
        self.user_id=user_id
        

    async def GOOGLEMEET_CREATE_MEET(self, start_datetime: str, event_duration_hour: int, summary: str):
        """
        Creates a Google Meet meeting with the specified start time and duration.

        Parameters:
        - start_datetime (str): The start time of the meeting in ISO 8601 format with timezone. Example: "2023-10-01T10:00:00+05:30" (default is indian standard time).
        - event_duration_hour (int): The duration of the meeting in hours. Example: 2.
        - summary (str): The summary or title of the meeting. Example: "Team Sync".

        Returns:
        - str: The Google Meet link for the created meeting.
        """
        
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        param = {"start_datetime": start_datetime, "create_meeting_room": True, "event_duration_hour": event_duration_hour,"timezone":"IST","summary":summary}    
        response = composio_toolset.execute_action(params=param, action='GOOGLECALENDAR_CREATE_EVENT')
        link = response['data']['response_data']['hangoutLink']
        start_time_dt = datetime.fromisoformat(start_datetime.replace('Z', '+00:00'))
        reminder_time = start_time_dt - timedelta(minutes=10)
        rem=Reminder(
            user_id=self.user_id,
            start_time=reminder_time.isoformat(),
            message=summary+"\nmeet link : "+link,
        )
        rid=create_reminder(rem)["reminder_id"]
        return {
            "summary": summary,
            "link": link
        }
        
        

    async def GOOGLEMEET_GET_CONFERENCE_RECORD_FOR_MEET(self, meeting_code: str = "The meeting code of the Google Meet space"):
        """
        Retrieves the conference record (e.g., transcripts) for a specific Google Meet meeting.

        Parameters:
        - meeting_code (str): The unique meeting code of the Google Meet space. Example: "abc-defg-hij".

        Returns:
        - dict: A dictionary containing the conference record data.
        """
        params = {"meeting_code": meeting_code}
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        response = composio_toolset.execute_action(params=params, action='GOOGLEMEET_GET_TRANSCRIPTS_BY_CONFERENCE_RECORD_ID')
        return response['data']
    async def execute(self):                       #LLM will only call execute
        action=self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")
            


# test_cases = [
#     {"action": "connect_youtube"},
#     {"action": "youtube_search", "query": "react native", "max_results": 2, "search_type": "video"},
#     {"action": "video_details", "video_id": "gvkqT_Uoahw"},
#     {"action": "list_user_subscriptions", "max_results": 2},
#     {"action": "list_user_playlists", "max_results": 5},
#     {"action": "subscribe_channel", "channel_

# api_key = "8o15k780p434dhgxyrdjs"
# youtube = Youtube(api_key=api_key)

# for test in test_cases:
#     action = test.pop("action")  
#     try:
#         result = youtube.execute(action, **test)
#         print(f"Test case {action}:", result)
#     except Exception as e:
#         print(f"Error in test case {action}: {e}")
#     print("---")

class RetrievalResult(BaseModel):
    content: str
    summary: str
    section_title: str
    section_index: int
    score: float
    source_file: str
    original_file: str
    s3_key: str
    project: str

class QueryResponse(BaseModel):
    results: List[Any]
    total_files_searched: int
    query_time_ms: float

class QueryRequest(BaseModel):
    query: str
    top_k_per_file: int = 5
    final_top_k: int = 10
    similarity_threshold: float = 0.0

async def query_project(
    project: str,
    query_request: QueryRequest,
    user_id: str    
):
    """
    Query all files in a user's project and return ranked results
    """
    import time


    start_time = time.time()


    try:
        # Convert query to embedding
        query_embedding = get_embedding_from_query(query_request.query)
        
        # Get all files in the project
        file_bases = await list_project_files(user_id, project)
        
        if not file_bases:
            return QueryResponse(
                results=[],
                total_files_searched=0,
                query_time_ms=round((time.time() - start_time) * 1000, 2)
            )
       
        # Load all indices and metadata sequentially
        loaded_data = []
        for file_base in file_bases:
            result = await load_file_index_and_metadata(user_id, project, file_base)
            loaded_data.append(result)
        print("Loaded data:", loaded_data)
        # Search each file sequentially
        search_tasks = []
        for index, metadata, file_base in loaded_data:
            if index is not None:
                print(f"Searching file: {file_base} with index: {index}")
                task = await search_single_file(
                    index, metadata, file_base, query_embedding,
                    query_request.top_k_per_file, project
                )
                search_tasks.append(task)

        print("Search tasks:", search_tasks)

        # Execute all searches
        # file_results = await asyncio.gather(*search_tasks)
       
        # Flatten results from all files
        all_results = []
        for results in search_tasks:
            all_results.extend(results)
       
        # Filter by similarity threshold
        filtered_results = [
            result for result in all_results
            if result.score >= query_request.similarity_threshold
        ]
        print("Filtered results:", filtered_results)
        # Rerank across all files
        final_results = rerank_results(filtered_results, query_request.final_top_k, query_request.query)
       
        query_time = round((time.time() - start_time) * 1000, 2)
        print("Final results:", final_results)
        return QueryResponse(
            results=final_results,
            total_files_searched=len([data for data in loaded_data if data[0] is not None]),
            query_time_ms=query_time
        )
   
    except Exception as e:
        print(f"Error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


class RAG:
    def __init__(self, api_key, kwargs,model,llm,user_id):
        
        self.model=model
        self.llm=llm
        self.api_key = api_key
        self.user_id= user_id
        self.kwargs = kwargs
        self.tool_set = ComposioToolSet(api_key=api_key)
        self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)

    async def execute(self):                       #LLM will only call execute
        action=self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")

    async def RETRIEVE_INFO_FROM_COLLECTION(self, collection_name: str , query:str):
        """
        Retrieves information from a specified collection based on a query.

        Parameters:
        - collection_name (str): The name of the collection to search in. Example: "my_collection".
        - query (str): The query to search for in the collection. Example: "What is the capital of France?".

        Returns:
        - list: A list of dictionaries containing the retrieved information.
        """
        req=QueryRequest(
            query=query,
            top_k_per_file=5,
            final_top_k=10,
            similarity_threshold=0.0
        )
        response= await query_project(
            project=collection_name,
            query_request=req,
            user_id=self.user_id
        )
        print("total files searched:", response.total_files_searched, "query time:", response.query_time_ms, "ms")
        
        prompt= f"""
        summarize the following information retrieved from the collection {collection_name} based on the query: {query}
        Results:
        {response.results}


        """
        
        
        try:
            llm_response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
        except Exception as e:
            error_str = str(e)
            print(f"Error occurred: {error_str}")
            if "429" in error_str and "retry_delay" in error_str:
                # Extract retry delay seconds
                retry_seconds = 60  # Default fallback
                # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                # if retry_match:
                #     retry_seconds = int(retry_match.group(1))
                
                print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                llm_response = await generate_con(uid=self.user_id, model=self.model, inside=prompt)
            else:
                raise e
        return llm_response.strip() if llm_response else "No relevant information found."


        

class PERSONAL_MEMORY:
    def __init__(self, api_key, kwargs, model, llm, user_id):
        """
        Initializes the PersonalMemory class with API key and additional parameters.

        Parameters:
        - api_key (str): The API key for authentication.
        - kwargs (dict): Additional parameters for the class.
        - model (str): The model name to use for embeddings and generation.
        - llm: The language model to use for processing.
        - user_id (str): The ID of the user to keep memories separated.
        """
        self.model = model
        self.llm = llm
        self.api_key = api_key
        self.kwargs = kwargs
        self.user_id = user_id
        if api_key:
            self.tool_set = ComposioToolSet(api_key=api_key)
            self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)
        # # Create directory for ChromaDB data if it doesn't exist
        # db_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        # os.makedirs(db_directory, exist_ok=True)
    
        # Create or get collection specific to this user
        self.collection = chroma_client.get_or_create_collection(
            name=f"user_memory_{self.user_id}",
            metadata={"user_id": self.user_id}
        )
        response = db_client.get_item(
        TableName='users',
        Key={'clerk_id': {'S': user_id}}
    )
        api_keys = response.get('Item', {}).get('api_key', {}).get('M', {})
        gemini_key = api_keys.get('gemini', {}).get('S') if 'gemini' in api_keys else ""
        self.gemini_key = gemini_key

    def get_embedding(self, text):
        """
        Generates an embedding for the given text using Google's embedding model.
        
        Parameters:
        - text (str): The text to generate an embedding for.
        
        Returns:
        - list: The embedding vector.
        """
        try:
            genai.configure(api_key=self.gemini_key)
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    async def SAVE_MEMORY(self, info: str, s3_paths: list = []):
        """
        Saves information with its vector embedding in the user's personal memory.
        This function has to be called every time users throw new information at the system and dont say what they want to do with it.

        Parameters:
        - info (str): The description text / information to save. Strictly include links too if provided.
        - s3_paths (list, optional): List of S3 paths to files relevant to this information.
        
        Returns:
        - dict: Status of the save operation.
        """
        try:
            # Generate a unique ID for this memory entry
            memory_id = str(uuid.uuid4())
            
            # Generate embedding for the information
            embedding = self.get_embedding(info)
            
            if not embedding:
                return {"status": False, "error": "Failed to generate embedding for information"}
            
            # Prepare metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "user_id": self.user_id
            }
            
            # Add S3 paths to metadata if provided
            if s3_paths:
                metadata["s3_paths"] = json.dumps(s3_paths)
                
                # Generate presigned URLs for each S3 path
                # presigned_urls = []
                # for path in s3_paths:
                #     if path.startswith("s3://"):
                #         s3_parts = path.replace("s3://", "").split("/", 1)
                #         if len(s3_parts) == 2:
                #             bucket_name, key = s3_parts
                #             try:
                #                 presigned_url = s3_client.generate_presigned_url(
                #                     'get_object',
                #                     Params={
                #                         'Bucket': bucket_name,
                #                         'Key': key
                #                     },
                #                     # ExpiresIn=604800  # URL expires in 1 week
                #                 )
                #                 presigned_urls.append({"path": path, "url": presigned_url})
                #             except Exception as e:
                #                 print(f"Error generating presigned URL for {path}: {e}")
                
                # metadata["presigned_urls"] = json.dumps(presigned_urls)
            
            # Add to ChromaDB collection
            self.collection.upsert(
                ids=[memory_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[info]
            )
            
            return {
                "status": True,
                "memory_id": memory_id,
                "message": "Information saved successfully to Sigmoyd AI Mmory"
            }
            
        except Exception as e:
            print(f"Error saving information: {e}")
            return {"status": False, "error": str(e)}

    def clean_llm_json_response(self,raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)

    async def FETCH_FROM_MEMORY(self, query: str, synonyms: str ):
        """
        Retrieves information from the user's personal memory based on a query.
        Uses both semantic similarity search and keyword matching for comprehensive results.
        
        Parameters:
        - query (str): The search query by user. Eg - "What is my workout plan?"
        - synonyms (str): Few contextual synonym queries for the query to enhance search. Eg . "Gym Schedule Exercise Plan" (if user's query has typos, keep corrected synonyms here.) (please only use alphabets and spaces, no special characters)

        Returns:
        - list: Search results with both semantic matches and keyword matches.
        """
        limit = 2
        similarity_threshold = 0.4
        keyword_threshold = 0.2
        query += f" {synonyms}"
        try:
            results = {"semantic_matches": [], "keyword_matches": []}
            
            # Generate embedding for the query
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                return {"status": False, "error": "Failed to generate embedding for query"}
            
            # Perform semantic similarity search
            semantic_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Process semantic results
            if semantic_results and len(semantic_results['ids']) > 0:
                for i, doc_id in enumerate(semantic_results['ids'][0]):
                    if i >= len(semantic_results['distances'][0]):
                        continue
                    
                    similarity_score = 1.0 - semantic_results['distances'][0][i]  # Convert distance to similarity
                    
                    # Apply similarity threshold
                    if similarity_score >= similarity_threshold:
                        document = semantic_results['documents'][0][i]
                        metadata = semantic_results['metadatas'][0][i]
                        
                        # Extract S3 paths and presigned URLs if available
                        s3_paths = []
                        downloadable_links = []
                        if 's3_paths' in metadata:
                            s3_paths = json.loads(metadata['s3_paths'])
                        if 'presigned_urls' in metadata:
                            downloadable_links = json.loads(metadata['presigned_urls'])
                            # results["downloadable_links"].extend(downloadable_links)
                        
                        results["semantic_matches"].append({
                            "id": doc_id,
                            "content": document,
                            "similarity_score": similarity_score,
                            "timestamp": metadata.get("timestamp"),
                            "document_links": downloadable_links,
                            "s3_paths": s3_paths
                        })
            
            # Perform keyword matching as an alternative search method
            # This helps catch relevant results that might not be semantically similar
            all_docs = self.collection.get()
            if all_docs and 'documents' in all_docs and all_docs['documents']:
                # Normalize and tokenize the query for keyword matching
                query_keywords = set(query.lower().split())
                
                for i, document in enumerate(all_docs['documents']):
                    if i >= len(all_docs['ids']):
                        continue
                    
                    # Skip documents already found in semantic search
                    doc_id = all_docs['ids'][i]
                    if any(match["id"] == doc_id for match in results["semantic_matches"]):
                        continue
                    
                    # Simple keyword matching
                    document_text = document.lower()
                    keyword_matches = [kw for kw in query_keywords if kw in document_text]
                    
                    if keyword_matches:
                        match_ratio = len(keyword_matches) / len(query_keywords)
                        
                        # Only include if a significant portion of keywords match
                        if match_ratio >= keyword_threshold:
                            metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs and i < len(all_docs['metadatas']) else {}
                            
                            # Extract S3 paths and presigned URLs if available
                            s3_paths = []
                            downloadable_links = []
                            if 's3_paths' in metadata:
                                s3_paths = json.loads(metadata['s3_paths'])
                            if 'presigned_urls' in metadata:
                                downloadable_links = json.loads(metadata['presigned_urls'])
                                # Only add unique links
                                # existing_urls = [link["url"] for link in results["downloadable_links"]]
                                # for link in downloadable_links:
                                #     if link["url"] not in existing_urls:
                                #         results["downloadable_links"].append(link)
                            
                            results["keyword_matches"].append({
                                "id": doc_id,
                                "content": document,
                                "match_ratio": match_ratio,
                                "matched_keywords": keyword_matches,
                                "timestamp": metadata.get("timestamp"),
                                "document_links": downloadable_links,
                                "s3_paths": s3_paths,
                            })
            
            # Generate a summary if needed
            if results["semantic_matches"] or results["keyword_matches"]:
                # Prepare combined results for the summary
                combined_content = []
                for match in results["semantic_matches"]:
                    combined_content.append([match["content"], match["document_links"], match["s3_paths"],match["id"]])
                for match in results["keyword_matches"]:
                    combined_content.append([match["content"], match["document_links"], match["s3_paths"],match["id"]])
                to_embed = "\n\n############\n\n".join([f"CONTENT ID : {i[3]}\n"+str(i[0]) + "\n" + str("\n".join([j["url"] for j in i[1]])) for i in combined_content])
                # for match in results["semantic_matches"]:
                #     combined_content.append([match["content"], match["s3_paths"]])
                # for match in results["keyword_matches"]:
                #     combined_content.append([match["content"], match["s3_paths"]])
                # to_embed = "\n\n##############################\n\n".join([str(i[0]) for i in combined_content])
                print(to_embed)
                # Generate a summary if there are multiple results
                # if len(combined_content) > 1:
                summary_prompt = f"""
                You are an intelligent assistant who needs to select the exact content from user's personal memory based on the query and Retrieved information.
                The retrieved information contains the Content ID, content with links to the content (if any). Your task is to select all relevant content ids which will answer the user's query.
               #############################
                User's query: {query}
                ############################
                Retrieved information: 
                {to_embed}
                
               ##############################
                STRICTLY DON'T INCLUDE ID OF UNWANTED INFORMATION. INCLUDE EXACTLY THE CONTENT IDS OF INFORMATION USER WANTS TO FETCH.
                ALSO , MAKE SURE TO INCLUDE ALL THE CONTENT IDS WHICH ARE RELEVANT TO THE QUERY.
                If you feel the retrieved information is not relevant to the query, return an empty list [].

                OUTPUT IN THE LIST FORMAT:
                ["Content ID 1","Content ID 2",....]
                """

                
                try:
                    summary = await generate_con(uid=self.user_id, model=self.model, inside=summary_prompt)
                    # results["summary"] = summary.strip()
                except Exception as e:
                    error_str = str(e)
                    print(f"Error occurred: {error_str}")
                    if "429" in error_str and "retry_delay" in error_str:
                        # Extract retry delay seconds
                        retry_seconds = 60  # Default fallback
                        # retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                        # if retry_match:
                        #     retry_seconds = int(retry_match.group(1))
                        
                        print(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                        await asyncio.sleep(retry_seconds + 1)  # Add 1 second buffer
                        # Retry the request
                        summary = await generate_con(uid=self.user_id, model=self.model, inside=summary_prompt)
                        # results["summary"] = summary.strip() if summary else "No summary could be generated."
                    else:
                        raise e
                    
                id_list = self.clean_llm_json_response(summary)
                print("List of content IDs to return:", id_list)

                filtered = "\n\n".join([str(i[0]) for i in combined_content if i[3] in id_list])
                filtered += "\nRetrieved files associated are sent above."

                # retrieving phone number using user id
                user_response = db_client.get_item(
                    TableName='users',
                    Key={'clerk_id': {'S': self.user_id}}
                )
                phone_number = user_response.get('Item', {}).get('whatsapp_verified', {}).get('S', '')
                try:
                    for match in combined_content:
                        if match[3] in id_list:
                            for path in match[2]:
                                s3_to_whatsapp(path,phone_number)
                except Exception as e:
                    print(f"Error sending files via WhatsApp: {e}")

                     


            return filtered if (results["semantic_matches"] or results["keyword_matches"]) else "No relevant information found in Sigmoyd AI Memory."

        except Exception as e:
            print(f"Error retrieving information: {e}")
            return {"status": False, "error": str(e)}

    async def execute(self):
        """
        Executes the specified action dynamically.

        Parameters:
        - action (str): The name of the method to execute.

        Returns:
        - The result of the executed method or an error message if the method is not found.
        """
        action = self.kwargs["action"]
        del self.kwargs["action"]
        
        if hasattr(self, action):  # Check if function exists
            return await getattr(self, action)(**self.kwargs)  # Call the function dynamically
        else:
            return print(f"Method {action} not found")



#TESTING
if __name__ == "__main__":
    pass
    # gmail_obj = gmail(api_key="8o15k780p434dhgxyrdjs", kwargs={"action": "gmail_send_email", "recipient_email": "




# instance = MyClass()
# func_name = "my_method"

# # Get method reference from the instance
# method = getattr(instance, func_name)

# # Get the signature of the method
# signature = inspect.signature(method)

# # Extract parameters with default values (or None if no default)
# inputs = {name: param.default if param.default is not inspect.Parameter.empty else None
#           for name, param in signature.parameters.items()}
