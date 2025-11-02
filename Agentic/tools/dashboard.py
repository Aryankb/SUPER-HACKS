from composio_langchain import ComposioToolSet
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re
import time
import uuid
from tools.dynamo import db_client
from tools.llm import get_chain


class sheet_data_fetcher:
    def __init__(self,api_key=None, llm=None,user_id=None):
        self.api_key = api_key
        self.llm=llm
        self.user_id = user_id
    def clean_llm_json_response(self,raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)

    def GOOGLESHEETS_BATCH_GET(self,spreadsheetID,sheet_name="Sheet1"):
        
        params={'spreadsheet_id':spreadsheetID,"ranges":[sheet_name]}
        
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_GET')
        
        values = response['data']['valueRanges'][0].get('values', [])
        header = values[0] if values else []
        rows = values[1:] if len(values) > 1 else []
        
        mapped_data = [dict(zip(header, row)) for row in rows]
        print("mapped data:",mapped_data)

        return {'response':mapped_data,"headers":header}

    def get_sheet_header(self,spreadsheetID, sheet_name="Sheet1"):

        response=self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
        headers = response['headers']
        return headers

    def header_description(self,spreadsheetID,sheet_name="Sheet1"):

        # header=self.get_sheet_header(spreadsheetID=spreadsheetID, sheet_name=sheet_name)
        data=self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID,sheet_name=sheet_name)
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
            all_columns_prompt += f"Column: {column_name}\nSample data: {samples[0] if samples else 'No samples available'} , type : {type(samples[0]) if samples else 'unknown'}\n\n"
        all_columns_prompt += "Return a parsable JSON object where each key is a column name string and each value is its description string. The description must contain the data type of each column. Be concise and direct without reasoning or explanation. Please keep all the columns in the response json, no matter samples available or not. please keep all strings in double quotes. No preambles or postambles, just the JSON object.\n\n"

        # Make a single LLM call
        response = get_chain(uid=self.user_id,prompt=self.llm,inside=all_columns_prompt,mo="gemini-2.0-flash",path=1)
            
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


class Dashboard_Logic_Maker:
    def __init__(self, api_key, llm,user_id):
        self.api_key = api_key
        self.user_id = user_id
        self.sheet_fetcher = sheet_data_fetcher(api_key=api_key,llm=llm,user_id=user_id)
        self.llm=llm
    def clean_llm_json_response(self,raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)
    def insights_finder(self,query:str,sheetID:str,sheet_name:str):
        headers = self.sheet_fetcher.header_description(spreadsheetID=sheetID,sheet_name=sheet_name)
        task=f'''you are a smart insight finder agent , your job is to just tell what insights can be formed from a data with its column description:-\n {headers} \nin order to accomplish the user query :\n {query}\n
            expected output is just a json of most legitimate insights that can be formed and it can be max 5 insights.
            example json : {{'insight 1' : 'description of insight 1', 'insight 2': 'description of insight 2', etc.}}
            No preambles, no postambles, no explanations, just return the json of insights as described above'''
        
        response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)
        return self.clean_llm_json_response(response.content)
    def Best_representation_finder(self,query:str,sheetID:str,sheet_name:str="Sheet1"):
        insights=self.insights_finder(query=query,sheetID=sheetID,sheet_name=sheet_name)
        main=[]
        j=list(insights.values())
        task=f'''you are expert in data representation and visualization finding . your job is to look into the following insight and find the best representation for it : \nINSIGHT: \n{j}\n,
            from the following :
            1. numeric represenation 
            2. scatter plot
            3. bar graph
            4. pie chart
            5. line graph
            6. progress bar
            7. table
            expected output list of json in format -{{
            insight: <insight description>,
            representation: <best representation for the insight>,
            }}
            expected output is just the best representation for the insight in string format.
            No preambles, no postambles, no explanations, just return the best representation as described above'''
        
        response = get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1)
        try:
            reps = self.clean_llm_json_response(response.content)
            print("response content :",reps , 'type:',type(reps))
            params={
    "numeric_representation": {
        "value": "<integer/float to be displayed prominently (e.g., total sales)>",
        
    },
    "progress_bar": {
        "value": "<progress percentage as a number between 0–100>",
        
    },
    "line_graph": {
        "labels": "<array of x-axis labels (e.g., time periods)>",
        "data": "<array of corresponding y-axis values (e.g., sales)>",
        
    },
    "bar_graph": {
        "labels": "<array of categories (e.g., product types)>",
        "data": "<array of values corresponding to each category>",
        
    },
    "scatter_plot": {
        "data": "<array of [x, y] coordinate pairs (e.g., [[1,2],[3,4]])>",
        
    },
    
    "pie_chart": {
        "labels": "<array of segments/categories for the pie chart>",
        "data": "<array of values corresponding to each segment>",
        
    },
    "scrollable_table": {
        "headers": "<array of column headers>",
        "rows": "<array of arrays, each representing a table row with values matching the headers>",
        
    }
}           
            for rep in reps:
                if rep['representation'].lower() == 'numeric representation':
                    rep['params'] = params['numeric_representation']
                elif rep['representation'].lower() == 'progress bar':
                    rep['params'] = params['progress_bar']
                elif rep['representation'].lower() == 'line graph':
                    rep['params'] = params['line_graph']
                elif rep['representation'].lower() == 'bar graph':
                    rep['params'] = params['bar_graph']
                elif rep['representation'].lower() == 'scatter plot':
                    rep['params'] = params['scatter_plot']
                elif rep['representation'].lower() == 'pie chart':
                    rep['params'] = params['pie_chart']
                elif rep['representation'].lower() == 'scrollable table':
                    rep['params'] = params['scrollable_table']
            return reps
        except json.JSONDecodeError as e:
            return "internal server error , please try again later"
            


class Dashboard_Maker:
    def __init__(self, api_key,llm,user_id):
        self.api_key = api_key
        self.sheet_fetcher = sheet_data_fetcher(api_key=api_key, llm=llm,user_id=user_id)
        self.llm=llm
        self.user_id = user_id
    def clean_llm_json_response(self,raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)
    def rules_maker(self, representations: list, sheetID: str,sheet_name:str="Sheet1"):
        representations = representations
        header_desc = self.sheet_fetcher.header_description(spreadsheetID=sheetID,sheet_name=sheet_name)
        print("user id:", self.user_id  )
        task = f'''Given a pandas DataFrame and the following PLAN: \n{representations}\n, generate Python functions (using only pandas) to compute the required outputs as described in each "params" field. 
        Columns and their descriptions:\n {header_desc}\n.
        Return a JSON list, each item:
        {{
        "function_name": "<function name>",
        "code": "<python code as string>",
        "description": "<brief function description>",
        "representation":"scatter_plot"/"pie_chart"/"data_table"/"line_graph"/"bar_graph"/"progress"/"numeric_representation"
        }}
        Rules for creating functions:
        1. Handle null values, data types, and please write error free codes. The function should not have any print statements
        2. Functions must match the order of the representations list. And each function should return the expected output as a dictionary described in the "params" field given in the representations.
        3. Use only pandas. And each function should have only one input parameter as df (the DataFrame)
        4. For categorical columns, use df['col'].unique() to get unique values.
        5. For numeric columns, use appropriate pandas numeric operations.
        6. Strictly follow the structure and requirements in each "params" field.
No extra text required, just the JSON list.'''
        response = self.clean_llm_json_response(get_chain(uid=self.user_id,prompt=self.llm,inside=task,mo="gemini-2.0-flash",path=1).content)
        print("response content :", response, 'type:', type(response))
        return response

    def final_end_process(self, representations:list, sheetID:str, sheet_name:str="Sheet1"):
        # final_list=self.rules_maker(representations=representations,sheetID=sheetID)
        final_list=self.rules_maker(representations=representations,sheetID=sheetID,sheet_name=sheet_name)
        # Convert the list of dictionaries to a string representation
        dashboard_id = str(uuid.uuid4())
        print("user id:", self.user_id  )

        # Prepare the item to insert
        item = {
            'dashboard_id': {'S': dashboard_id},
            'sheet_id': {'S': sheetID},
            'sheet_name': {'S': sheet_name if sheet_name else ''},
            'user_id': {'S': self.user_id if self.user_id else ''},
            'rules': {'S': json.dumps(final_list)},
            'created_at': {'S': time.strftime("%Y-%m-%d %H:%M:%S")},
            'api_key': {'S': self.api_key if self.api_key else ''}
        }

        # Use the imported db_client to put the item
        db_client.put_item(
            TableName='Dashboard_data',
            Item=item
        )

        # Update user's dashboard list in Users table
        try:
            # Get current user data
            user_response = db_client.get_item(
                TableName='users',
                Key={
                    'clerk_id': {'S': self.user_id}
                }
            )
            
            # Check if user exists and if they have existing dashboards
            if 'Item' in user_response:
                # If dashboard column exists, update it
                if 'dashboards' in user_response['Item']:
                    existing_dashboards = json.loads(user_response['Item']['dashboards']['S'])
                    if not isinstance(existing_dashboards, list):
                        existing_dashboards = []
                    existing_dashboards.append(dashboard_id)
                else:
                    # If dashboard column doesn't exist, create it
                    existing_dashboards = [dashboard_id]
                    
                # Update the user record
                db_client.update_item(
                    TableName='users',
                    Key={
                        'clerk_id': {'S': self.user_id}
                    },
                    UpdateExpression='SET dashboards = :dashboards',
                    ExpressionAttributeValues={
                        ':dashboards': {'S': json.dumps(existing_dashboards)}
                    }
                )
        except Exception as e:
            print(f"Error updating user's dashboard list: {str(e)}")
        return {
            'dashboard_url': "https://app.sigmoyd.in/dashboard?dashboard_id=" + dashboard_id,
        }
    

if __name__ == "__main__":

    pass
