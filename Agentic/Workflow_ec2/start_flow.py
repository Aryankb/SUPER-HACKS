""""
missing --  custom tools, deligator, logging, error handling. make sure each .execute() method returns a dict with status and message
"""

from tools.dynamo import db_client
import json
from tools.tool_classes import *  # Import all tool classes dynamically
from Workflow_ec2.oth_tools import *
from tools.llm import get_chain
# from prompts import llm_sys_chain,iterator_chain,gemini_chain
from prompts import llm_sys, iterator_flow, gemini_prompt
import inspect
import logging
from celery import Celery
import redis
import asyncio
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
celery_app = Celery("tasks", broker="redis://redis:6379/0", backend="redis://redis:6379/0")
redis_client = redis.Redis(host='redis', port=6379, db=0)

# Configure logging
logging.basicConfig(
    filename='/home/aryan/BABLU/Agentic/Workflow_ec2/workflow.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@celery_app.task
def syn(wid,workflow_json, clerk_id, trigger_output):
    
    try:
        # Increment whatsapp_count if it exists, otherwise create it with value 1
        response = db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': clerk_id}},
            UpdateExpression="SET whatsapp_count = if_not_exists(whatsapp_count, :zero) + :one",
            ExpressionAttributeValues={
                ':zero': {'N': '0'},
                ':one': {'N': '1'}
            },
            ReturnValues="UPDATED_NEW"
        )
        logging.info(f"Updated WhatsApp count for user {clerk_id}")
        updated_count = response.get("Attributes", {}).get("whatsapp_count", {}).get("N", 0)
    except Exception as e:
        logging.error(f"Error updating WhatsApp message count: {str(e)}")

    if "Following is the query from user via whatsapp.  Please identify user needs by given chat history and new query." in trigger_output:
        updated_count = int(updated_count) - 1
        try:
            # Update the DB with the corrected count
            db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': clerk_id}},
                UpdateExpression="SET whatsapp_count = :count",
                ExpressionAttributeValues={
                    ':count': {'N': str(updated_count)}
                }
            )
            logging.info(f"Adjusted WhatsApp count for user {clerk_id} to {updated_count}")
        except Exception as e:
            logging.error(f"Error updating corrected WhatsApp count: {str(e)}")
    
    user_data = db_client.get_item(
        TableName="users",
        Key={"clerk_id": {"S": clerk_id}}
    )
    
    
    # Check the user's plan
    user_plan = user_data.get('Item', {}).get('plan', {}).get('S', 'free')
    logging.info(f"User {clerk_id} with plan {user_plan} sent a whatsapp message")

    if user_plan == 'free' or user_plan == 'trial_ended':
        return 
        
        
    elif user_plan == 'free_15_pro':
        # For free plan users, check if they've exceeded their daily limit
        if int(updated_count) > 10:
            return

    elif user_plan == 'pro':
        # For pro plan users, check if they've exceeded their daily limit
        if int(updated_count) > 50:
            
            return

    elif user_plan == 'pro++':
        # For pro++ plan users, check if they've exceeded their daily limit
        if int(updated_count) > 150:
            
            return
    asyncio.run(execute_workflow(wid,workflow_json, clerk_id, trigger_output))

async def execute_workflow(wid,workflow_json, user_id, tr_o=None,dfn=None,iterr=False):
    # Get user data to fetch API keys
    response = db_client.get_item(
        TableName='users',
        Key={'clerk_id': {'S': user_id}}
    )
    api_keys = response.get('Item', {}).get('api_key', {}).get('M', {})
    gemini_key = api_keys.get('gemini', {}).get('S') if 'gemini' in api_keys else ""
    model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=gemini_key
    # other params...
)
    
    genai.configure(api_key=gemini_key)
    llm = genai.GenerativeModel(model_name="gemini-2.0-flash")
    # Configure with the appropriate key
    llm_sys_chain=llm_sys|model|StrOutputParser()
    iterator_chain=iterator_flow|model|StrOutputParser()
    gemini_chain=gemini_prompt|model|StrOutputParser()
    # genai.configure(api_key=gemini_key)
    
    print("EXECUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUTTTTTTTTTTTINGGGGGGGGGGGG")
    print("iterr status",iterr) 
    if dfn==None:
        
        data_flow_notebook = {"trigger_output": tr_o}
    else:
        data_flow_notebook = dfn
    # composio_tools = {cls.__name__.lower(): cls for cls in globals().values() if isinstance(cls, type)}
    try:
        # Get the absolute path to the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to the composio.json file
        composio_json_path = os.path.join(os.path.dirname(current_dir), 'tools', 'composio.json')
        with open(composio_json_path, 'r') as file:
            composio_tools_data = json.load(file)
            composio_tools = {}
            for key, value in composio_tools_data.items():
                try:
                    tool_class = globals()[key]
                    if inspect.isclass(tool_class):
                        composio_tools[key.lower()] = tool_class
                except (KeyError, TypeError):
                    logging.warning(f"Could not find class for tool: {key}")
        logging.info("Loaded composio_tools from file: %s", list(composio_tools.keys()))
    except Exception as e:
        logging.error("Failed to load composio_tools from file: %s", e)
        composio_tools = {}

    # logging.info("Initialized composio_tools: %s", composio_tools)
    if isinstance(workflow_json, str):
        try:
            workflow_json = json.loads(workflow_json)
        except json.JSONDecodeError as e:
            logging.error("Invalid JSON format: %s", e)
            return {"status": "error", "message": "Invalid JSON format"}
    iter_over=False
    for agent in workflow_json:
        response=""
        status="executed successfully"
        agent_id = agent["id"]
        agent_type = agent["type"]
        agent_name = agent["name"].lower()
        to_execute = agent["to_execute"]
        data_flow_inputs = agent.get("data_flow_inputs", [])
        data_flow_outputs = agent.get("data_flow_outputs", [])
        config_inputs = agent.get("config_inputs", {})
        description = agent.get("description", "")
        if not iter_over:
            pass
        elif "iter_end" in agent_name and agent_type == "connector":
            iter_over=False
            print("inside iter end : now next  tools will execute")
            pass
        else:
            print("inside iter : continuing")
            continue
        logging.info("Processing agent: %s, Type: %s, Description: %s", agent_name, agent_type, description)

        
        # Check execution conditions
        if to_execute:
            condition_key, expected_value = to_execute
            print("condition_key",condition_key)
            print("expected_value",expected_value)
            if (data_flow_notebook.get(condition_key) == expected_value ):
                pass
            else:
                logging.info("Skipping agent %s due to execution condition mismatch", agent_name)
                continue
        try:
            input_data = {key: data_flow_notebook[key] for key in data_flow_inputs}
            logging.info("Input data for agent %s: %s", agent_name, input_data)
        except KeyError as e:
            input_data = {}
            logging.error("Missing data flow inputs key: %s", e)
            continue
        
        # LLM Agent Execution
        if agent_type == "llm":
            system_prompt = agent["llm_prompt"]
            logging.info("Executing LLM agent: %s with prompt: %s", agent_name, system_prompt)
            # logging.info("Config inputs: %s, Input data: %s", config_inputs, input_data)
            # logging.info("yayyyyy %s", input_data.update(config_inputs))  
            # response = llm_sys_chain.invoke({"data": {k: v for k, v in list(config_inputs.items()) + list(input_data.items())}, "question": system_prompt, "keys": data_flow_outputs})
            try:
                
                response = get_chain(uid=user_id,prompt=llm_sys_chain,inside={"data": {k: v for k, v in list(config_inputs.items()) + list(input_data.items())}, "question": system_prompt, "keys": data_flow_outputs},mo="gemini-2.0-flash",path=1)
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
                    response = get_chain(uid=user_id,prompt=llm_sys_chain,inside={"data": {k: v for k, v in list(config_inputs.items()) + list(input_data.items())}, "question": system_prompt, "keys": data_flow_outputs},mo="gemini-2.0-flash",path=1)
                else:
                    raise e
            logging.info("LLM response: %s", response)
            try:
                if response[0]=="`":
                    response = response[7:-4]
                    print("response",response)
                    try:
                        response = json.loads(response)
                    except:
                        response = eval(response)
                else:
                    try:
                        response = json.loads(response)
                    except:
                        response = eval(response)
                
                data_flow_notebook.update(response)
            except Exception as e:
                status="failed"
                logging.error("Error processing LLM response: %s", e)
        
        # Connector Execution
        elif agent_type == "connector":
            logging.info("Executing connector agent: %s", agent_name)
            if "iterator" in agent_name:
                elements = data_flow_notebook[data_flow_inputs[0]]
                if not isinstance(elements, list):
                    try:
                        
                        # elements = iterator_chain.invoke({"data": elements+f"\nTASK : {description}\n make sure that list elements are as expected according to given task"})
                        try:
                            elements = get_chain(uid=user_id,prompt=iterator_chain,inside={"data": elements+f"\nTASK : {description}\n make sure that list elements are as expected according to given task"},mo="gemini-2.0-flash",path=1)
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
                                elements = get_chain(uid=user_id,prompt=iterator_chain,inside={"data": elements+f"\nTASK : {description}\n make sure that list elements are as expected according to given task"},mo="gemini-2.0-flash",path=1)
                            else:
                                raise e
                    except KeyError:
                        d={k: v for k, v in list(config_inputs.items()) + list(input_data.items())}
                        # elements = iterator_chain.invoke({"data": f"{d}" + f"\nTASK : {description}\n make sure that list elements are as expected according to given task"})
                        try:
                            elements = get_chain(uid=user_id,prompt=iterator_chain,inside={"data": f"{d}" + f"\nTASK : {description}\n make sure that list elements are as expected according to given task"},mo="gemini-2.0-flash",path=1)
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
                                elements = get_chain(uid=user_id,prompt=iterator_chain,inside={"data": f"{d}" + f"\nTASK : {description}\n make sure that list elements are as expected according to given task"},mo="gemini-2.0-flash",path=1)
                            else:
                                raise e


                    try:
                        if elements[0] == "`":
                            try:
                                elements = json.loads(elements[7:-4])
                            except:
                                elements = eval(elements[7:-4])
                        else:
                            try:
                                elements = json.loads(elements)
                            except:
                                elements = eval(elements)
                        for element in elements:
                            data_flow_notebook[data_flow_outputs[0]] = element
                            await execute_workflow(
                                wid=wid,
                                workflow_json=workflow_json[agent_id:],
                                user_id=user_id,
                                tr_o=tr_o,
                                dfn=data_flow_notebook,
                                iterr=True
                            )
                        iter_over=True
                    except Exception as e:
                        status="failed"
                        logging.error("Error in iterator processing: %s", e)
                else:
                    for element in elements:
                        print("element",element)
                        data_flow_notebook[data_flow_outputs[0]] = element
                        await execute_workflow(
                                wid=wid,
                                workflow_json=workflow_json[agent_id:],
                                user_id=user_id,
                                tr_o=tr_o,
                                dfn=data_flow_notebook,
                                iterr=True
                            )
                    iter_over=True
                    

                    logging.info("Iterator processing completed for agent: %s", agent_name)

            elif "validator" in agent_name:
                system_prompt = agent["validation_prompt"]
                logging.info("Executing validator agent: %s", agent_name)
                # response = llm_sys_chain.invoke({"data": {k: v for k, v in list(config_inputs.items()) + list(input_data.items())}, "question": system_prompt + "\nwrite A/B/C... as value for given key according to validation criteria and required paths", "keys": data_flow_outputs})
                try:
                    response = get_chain(uid=user_id,prompt=llm_sys_chain,inside={"data": {k: v for k, v in list(config_inputs.items()) + list(input_data.items())}, "question": system_prompt + "\nwrite A/B/C... as value for given key according to validation criteria and required paths", "keys": data_flow_outputs},mo="gemini-2.0-flash",path=1)

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
                        response = get_chain(uid=user_id,prompt=llm_sys_chain,inside={"data": {k: v for k, v in list(config_inputs.items()) + list(input_data.items())}, "question": system_prompt + "\nwrite A/B/C... as value for given key according to validation criteria and required paths", "keys": data_flow_outputs},mo="gemini-2.0-flash",path=1)

                    else:
                        raise e
                try:
                    if response[0] == "`":
                        try:
                            response = json.loads(response[7:-4])
                        except:
                            response = eval(response[7:-4])
                    else:
                        try:
                            response = json.loads(response)
                        except:
                            response = eval(response)
                    data_flow_notebook.update(response)
                except Exception as e:
                    status="failed"
                    logging.error("Error processing validator response: %s", e)
            elif "delegator" in agent_name:
                status="failed"
                logging.info("Delegator agent: %s has no action specified", agent_name)
            

            elif "iter_end" in agent_name:
                logging.info("End of iterator agent: %s", agent_name)
                print("iterr status",iterr)
               
                if iterr:
                    # put some data in merged_output
                    print("inside iterator")
                    try:
                        data_flow_notebook[data_flow_outputs[0]] += "\n"+str(input_data)
                    except:
                        data_flow_notebook[data_flow_outputs[0]] = str(input_data)
                    return
                
                # implement printing the merged data
                response=data_flow_notebook[data_flow_outputs[0]]
                pass

           
        

        # Tool Execution
        elif agent_type == "tool":
            logging.info("Executing tool agent: %s", agent_name)
            user_data = db_client.get_item(
                        TableName="users",
                        Key={"clerk_id": {"S": user_id}}
            )
            api_key = user_data.get("Item", {}).get("api_key", {}).get("M", {})
            api_keys = {k: v['S'] for k, v in api_key.items()}
            comp = api_keys["composio"]
            # gem=api_keys["gemini"]
            # if any(partial in agent_name for partial in composio_tools):
            flag=1
            for partial in composio_tools:
                if partial in agent_name:
                    flag=2
                    agent_name=partial
                    break
            if flag==2:
                try:
                    
                    method = getattr(composio_tools[agent_name], agent.get("tool_action", "").upper())
                    docstring = method.__doc__ if method.__doc__ else "No documentation available."
                    # Get function signature
                    inputs="\n"+docstring
                    print("inputs",inputs,"input_data",input_data,"config_inputs",config_inputs)
                    print(f"You are an input validator for a function. Convert the given inputs to a dictionary format, with keys as parameter names of the function and values as the corresponding input values in proper required format. strictly convert the input parameters to required format. The given data might be in natural language, but you need to make sure you are extracting exact information in proper format from given data. STRICTLY DON'T GIVE ANY OTHER KEY, OTHER THAN INPUT PARAMETERS OF FUNCTION. If a parameter is not provided, set it to most relevant value. If still cannot find most relevant value, then don't keep that key in the output. Return the dictionary in JSON format. No preambles or postambles. keep all strings in double quotes.\nInput parameter names and their explaination:{inputs}\nAIM: {description}\ndata to insert (don't skip anything. each of the following data should go into some parameter values):"+str({k: v for k, v in list(config_inputs.items()) + list(input_data.items())}))
                    # to_go = gemini_chain.invoke({"prompt": f"You are an input validator for a function. Convert the given inputs to a dictionary format, with keys as parameter names of the function and values as the corresponding input values in proper required format. strictly convert the input parameters to required format. The given data might be in natural language, but you need to make sure you are extracting exact information in proper format from given data. STRICTLY DON'T GIVE ANY OTHER KEY, OTHER THAN INPUT PARAMETERS OF FUNCTION. If a parameter is not provided, set it to most relevant value. If still cannot find most relevant value, then don't keep that key in the output. Return the dictionary in JSON format. No preambles or postambles. keep all strings in double quotes.\nInput parameter names and their explaination:{inputs}\nAIM: {description}\ndata to insert (don't skip anything. each of the following data should go into some parameter values):"+str({k: v for k, v in list(config_inputs.items()) + list(input_data.items())})})
                    try:
                        to_go = get_chain(uid=user_id,prompt=gemini_chain,inside={"prompt": f"You are an input validator for a function. Convert the given inputs to a dictionary format, with keys as parameter names of the function and values as the corresponding input values in proper required format. strictly convert the input parameters to required format. The given data might be in natural language, but you need to make sure you are extracting exact information in proper format from given data. STRICTLY DON'T GIVE ANY OTHER KEY, OTHER THAN INPUT PARAMETERS OF FUNCTION. If a parameter is not provided, set it to most relevant value. If still cannot find most relevant value, then don't keep that key in the output. Return the dictionary in JSON format. No preambles or postambles. keep all strings in double quotes.\nInput parameter names and their explaination:{inputs}\nAIM: {description}\ndata to insert (don't skip anything. each of the following data should go into some parameter values):"+str({k: v for k, v in list(config_inputs.items()) + list(input_data.items())})},mo="gemini-2.0-flash",path=1)
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
                            to_go = get_chain(uid=user_id,prompt=gemini_chain,inside={"prompt": f"You are an input validator for a function. Convert the given inputs to a dictionary format, with keys as parameter names of the function and values as the corresponding input values in proper required format. strictly convert the input parameters to required format. The given data might be in natural language, but you need to make sure you are extracting exact information in proper format from given data. STRICTLY DON'T GIVE ANY OTHER KEY, OTHER THAN INPUT PARAMETERS OF FUNCTION. If a parameter is not provided, set it to most relevant value. If still cannot find most relevant value, then don't keep that key in the output. Return the dictionary in JSON format. No preambles or postambles. keep all strings in double quotes.\nInput parameter names and their explaination:{inputs}\nAIM: {description}\ndata to insert (don't skip anything. each of the following data should go into some parameter values):"+str({k: v for k, v in list(config_inputs.items()) + list(input_data.items())})},mo="gemini-2.0-flash",path=1)

                        else:
                            raise e
                    print(to_go)
                    if to_go[0] == "`":
                        try:
                            to_go = json.loads(to_go[7:-4])
                        except:
                            to_go = eval(to_go[7:-4])
                    else:
                        try:
                            to_go = json.loads(to_go)
                        except:
                            to_go=eval(to_go)
                    
                    kwargs = {"action": agent.get("tool_action", "").upper(), **to_go}
                    print("kwargs", kwargs)
                    
                        
                    tool_obj = composio_tools[agent_name](comp, kwargs,llm,model,user_id)
                    print("tool_obj",tool_obj)
                    
                    response = await tool_obj.execute()
                    
                    logging.info("Response from composo function: %s", response)
                    # response = llm_sys_chain.invoke({"data": response, "question": agent["description"], "keys": data_flow_outputs})
                    # if response[0] == "`":
                    #     response = json.loads(response[7:-4])
                    # else:
                    #     response = json.loads(response)
                    # data_flow_notebook.update(response)
                    for keys in data_flow_outputs:
                        data_flow_notebook[keys] = response
                except Exception as e:
                    status="failed"
                    logging.error("Error executing tool agent %s: %s", agent_name, e)
            else:
                # Check if a standalone function matches the agent_name.upper()
                # print("printing globals",agent_name.upper(), globals())
                if agent_name.upper() in globals() and callable(globals()[agent_name.upper()]):
                    try:
                        standalone_function = globals()[agent_name.upper()]
                        
                        # Get function signature
                        signature = inspect.signature(standalone_function)
                        inputs = [{name: param.default if param.default is not inspect.Parameter.empty else None}
                                  for name, param in signature.parameters.items()]
                        print("inputs",inputs,"input_data",input_data,"config_inputs",config_inputs)
                        # Perform input validation
                        # to_go = gemini_chain.invoke({"prompt": f"You are an input validator for a function. Convert the given inputs to a dictionary format, with keys as parameter names of the function and values as the corresponding inputs in proper required format. strictly convert the input parameters to required format. The given data might be in natural language, but you need to make sure you are extracting exact information in proper format from given data. STRICTLY DON'T GIVE ANY OTHER KEY, OTHER THAN INPUT PARAMETERS OF FUNCTION. If a parameter is not provided, set it to most relevant value. If still cannot find most relevant value, then don't keep that key in the output . Return the dictionary in JSON format. No preambles or postambles. keep all strings in double quotes.\nInput parameter names (REQUIRED KEYS) and their explaination:{inputs}\ndata including values for given keys above:"+str({k: v for k, v in list(config_inputs.items()) + list(input_data.items())})})
                        try:
                            to_go = get_chain(uid=user_id,prompt=gemini_chain,inside={"prompt": f"You are an input validator for a function. Convert the given inputs to a dictionary format, with keys as parameter names of the function and values as the corresponding inputs in proper required format. strictly convert the input parameters to required format. The given data might be in natural language, but you need to make sure you are extracting exact information in proper format from given data. STRICTLY DON'T GIVE ANY OTHER KEY, OTHER THAN INPUT PARAMETERS OF FUNCTION. If a parameter is not provided, set it to most relevant value. If still cannot find most relevant value, then don't keep that key in the output . Return the dictionary in JSON format. No preambles or postambles. keep all strings in double quotes.\nInput parameter names (REQUIRED KEYS) and their explaination:{inputs}\ndata including values for given keys above:"+str({k: v for k, v in list(config_inputs.items()) + list(input_data.items())})},mo="gemini-2.0-flash",path=1)
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
                                to_go = get_chain(uid=user_id,prompt=gemini_chain,inside={"prompt": f"You are an input validator for a function. Convert the given inputs to a dictionary format, with keys as parameter names of the function and values as the corresponding inputs in proper required format. strictly convert the input parameters to required format. The given data might be in natural language, but you need to make sure you are extracting exact information in proper format from given data. STRICTLY DON'T GIVE ANY OTHER KEY, OTHER THAN INPUT PARAMETERS OF FUNCTION. If a parameter is not provided, set it to most relevant value. If still cannot find most relevant value, then don't keep that key in the output . Return the dictionary in JSON format. No preambles or postambles. keep all strings in double quotes.\nInput parameter names (REQUIRED KEYS) and their explaination:{inputs}\ndata including values for given keys above:"+str({k: v for k, v in list(config_inputs.items()) + list(input_data.items())})},mo="gemini-2.0-flash",path=1)

                            else:
                                raise e
                        try:
                            if to_go[0] == "`":
                                try:
                                    to_go = json.loads(to_go[7:-4])
                                except:
                                    to_go = eval(to_go[7:-4])
                            else:
                                try:
                                    to_go = json.loads(to_go)
                                except:
                                    to_go=eval(to_go)
                                
                        except:
                            to_go=eval(to_go)
                        
                        # Call the standalone function
                        logging.info("%s",to_go)
                        response = standalone_function(**to_go)
                        
                        # Perform output validation
                        logging.info("Response from standalone function: %s", response)
                        # response = llm_sys_chain.invoke({"data": response, "question": agent["description"], "keys": data_flow_outputs})
                        # if response[0] == "`":
                        #     response = json.loads(response[7:-4])
                        # else:
                        #     response = json.loads(response)
                        # data_flow_notebook.update(response)
                        for keys in data_flow_outputs:
                            data_flow_notebook[keys] = response
                    except Exception as e:
                        status="failed"
                        logging.error("Error executing standalone function %s: %s", agent_name.upper(), e)
                else:
                    status="tool unavailable"
        logging.info("Data flow notebook:%s", data_flow_notebook)
                

        # redis_client.publish(f"workflow_{user_id}", json.dumps(data_flow_notebook))
        if type(response) == str:
            response={"response": response}
        log={
                    "workflow_id": wid,
                    "node": agent_id,
                    "agent_name": agent_name,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                    "data": json.dumps(response)
                }
        redis_client.xadd(f"workflow_{user_id}", log, maxlen=1000, approximate=True)
        
       
    return {"status": "success", "data": data_flow_notebook}












