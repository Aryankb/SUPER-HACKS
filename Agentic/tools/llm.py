from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.dynamo import db_client
import asyncio
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from tools.dynamo import db_client


def get_chain(uid, prompt, inside, mo="gemini-2.0-flash", path=0):
    print(f"get_chain called with uid: {uid}, ")
    response = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': uid}}
            )
    api_keys = response.get('Item', {}).get('api_key', {}).get('M', {})
    gemini_key = api_keys.get('gemini', {}).get('S') if 'gemini' in api_keys else ""
    
    # Initialize genai with the API key for token counting
    genai.configure(api_key=gemini_key)

    # Load the token counting model
    token_counter = genai.GenerativeModel(mo)
    
    
    def count_tokens(content):
        # Use genai.count_tokens for token counting
        try:
            result = token_counter.count_tokens(content)
            print(f"Counting tokens for content: {content}")
            print(f"Token count for content: {result}")
            return result.total_tokens
        
        except:
            return 0  # Fallback if token counting fails
    
    if path == 1:
        # For path 1, where prompt is a chain or ChatGoogleGenerativeAI object
        input_tokens = 0
        if isinstance(inside, str):
            input_tokens = count_tokens(inside)
        elif isinstance(inside, dict):
            # Count tokens in all string values in the dict
            for value in inside.values():
                if isinstance(value, str):
                    input_tokens += count_tokens(value)
        
        result = prompt.invoke(inside)
        output_tokens = count_tokens(result) if isinstance(result, str) else count_tokens(result.content)
        print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        
        
        
    else:
        
        
        model = ChatGoogleGenerativeAI(
            model=mo,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=gemini_key
        )
        
        chain = prompt | model | StrOutputParser()
        
        # Count input tokens
        input_tokens = 0
        if hasattr(prompt, "template"):
            input_tokens += count_tokens(prompt.template)
        if isinstance(inside, str):
            input_tokens += count_tokens(inside)
        elif isinstance(inside, dict):
            for value in inside.values():
                if isinstance(value, str):
                    input_tokens += count_tokens(value)
        
        result = chain.invoke(inside)
        output_tokens = count_tokens(result) if isinstance(result, str) else count_tokens(result.content)

        print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
    

    # Log tokens to DynamoDB
    try:
        # First, try to update both columns
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': uid}},
            UpdateExpression="ADD input_tokens :i, output_tokens :o",
            ExpressionAttributeValues={
                ':i': {'N': str(input_tokens)},
                ':o': {'N': str(output_tokens)}
            }
        )
    except Exception as e:
        # If the above fails (columns might not exist), try individual updates
        try:
            # Try to update input_tokens
            db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': uid}},
                UpdateExpression="SET input_tokens = if_not_exists(input_tokens, :zero) + :i",
                ExpressionAttributeValues={
                    ':i': {'N': str(input_tokens)},
                    ':zero': {'N': '0'}
                }
            )
        except Exception:
            pass
        
        try:
            # Try to update output_tokens
            db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': uid}},
                UpdateExpression="SET output_tokens = if_not_exists(output_tokens, :zero) + :o",
                ExpressionAttributeValues={
                    ':o': {'N': str(output_tokens)},
                    ':zero': {'N': '0'}
                }
            )
        except Exception:
            pass
    return result



async def generate_con(uid,model,inside,file=None):
    
    
    
    def count_tokens(content):
        # Use genai.count_tokens for token counting
        try:
            result = model.count_tokens(content)
            print(f"Counting tokens for content: {content}")
            print(f"Token count for content: {result}")
            return result.total_tokens
        
        except:
            return 0  # Fallback if token counting fails
    if file:
        to_go = [inside, file]
    else:
        to_go = [inside]
    
    # Count input tokens
    input_tokens = 0
    try:
        if isinstance(inside, str):
            input_tokens += count_tokens(inside).total_tokens
        # Note: We can't count tokens in the file object
    except:
        pass  # Fallback if token counting fails
    
    def call():
        response = model.generate_content(to_go, safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        })
        return response

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, call)
    
    # Count output tokens
    output_tokens = 0
    try:
        if hasattr(response, "text"):
            output_tokens = count_tokens(response.text)
    except:
        pass  # Fallback if token counting fails
    print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
    # Log tokens to DynamoDB
    try:
        # First, try to update both columns
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': uid}},
            UpdateExpression="ADD input_tokens :i, output_tokens :o",
            ExpressionAttributeValues={
                ':i': {'N': str(input_tokens)},
                ':o': {'N': str(output_tokens)}
            }
        )
    except Exception as e:
        # If the above fails (columns might not exist), try individual updates
        try:
            # Try to update input_tokens
            db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': uid}},
                UpdateExpression="SET input_tokens = if_not_exists(input_tokens, :zero) + :i",
                ExpressionAttributeValues={
                    ':i': {'N': str(input_tokens)},
                    ':zero': {'N': '0'}
                }
            )
        except Exception:
            pass
        
        try:
            # Try to update output_tokens
            db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': uid}},
                UpdateExpression="SET output_tokens = if_not_exists(output_tokens, :zero) + :o",
                ExpressionAttributeValues={
                    ':o': {'N': str(output_tokens)},
                    ':zero': {'N': '0'}
                }
            )
        except Exception:
            pass
    return response.text


