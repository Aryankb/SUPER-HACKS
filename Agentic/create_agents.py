from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect, Form, status, BackgroundTasks, File, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import HTMLResponse
from jose import jwt, jwk
import requests
import logging
import mimetypes
from clerk_backend_api import Clerk
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
from Workflow_ec2.start_flow import syn
from Workflow_ec2.oth_tools import *
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from tools.tool_classes import *
from tools.googlE import flow, setup_watch, save_google_credentials, get_fresh_google_credentials, extract_email,stop_gmail_watch,build
from auth import check_connection, create_connection_oauth2
from urllib.parse import unquote
from prompts import ques_flow,gemini_prompt,major_tools,trigger_flow,router,initial_tools,workflow_initial_tools
# from prompts import  ques_flow_chain,gemini_chain, major_tool_chain,trigger_chain, router_chain,  initial_tool_chain, general_agent_chain
from tools.dynamo import db_client, s3_client, lambda_client , dynamodb, DYNAMODB_AVAILABLE
from tools.llm import get_chain, generate_con
import json
import urllib.parse
import uuid
from fastapi.responses import JSONResponse
from openai import OpenAI
import asyncio
from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta, timezone
import pytz
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
import tempfile
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
import inspect
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from starlette.responses import Response
import random
from boto3.dynamodb.conditions import Attr
import razorpay
from langchain_core.prompts import ChatPromptTemplate
try:
    from composio import Composio
    from composio.client.enums import Action
    COMPOSIO_AVAILABLE = True
    print("SUCCESS: Composio API successfully imported")
except ImportError as e:
    print(f"ERROR: Composio not available: {e}")
    COMPOSIO_AVAILABLE = False

# view file, delete file, add links, delete link, database setting to edit and delete

class CustomProxyHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if "x-forwarded-proto" in request.headers:
            request.scope["scheme"] = request.headers["x-forwarded-proto"]
        response = await call_next(request) 
        return response


load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK"), base_url="https://api.deepseek.com")



#for instantaneous function calling
current_dir = os.path.dirname(os.path.abspath(__file__))
composio_json_path = os.path.join(os.path.dirname(__file__), 'tools', 'composio.json')
with open(composio_json_path, 'r') as file:
    composio_tools_data = json.load(file)
    composio_tools_dict = {}
    for key, value in composio_tools_data.items():
        try:
            tool_class = globals()[key]
            if inspect.isclass(tool_class):
                composio_tools_dict[key.lower()] = tool_class
        except (KeyError, TypeError):
            logging.warning(f"Could not find class for tool: {key}")


#keep in it for authentication
composio_tools=["GMAIL","NOTION","GOOGLESHEETS","GOOGLEDOCS","GOOGLEMEET","GOOGLECALENDAR","WHATSAPP","GOOGLEDRIVE"]

class Query(BaseModel):
    query: str
    flag:int
    wid:str

class Question(BaseModel):
    question: dict
    query: str
    flag:int

class W(BaseModel):
    workflowjson: dict

class WP(BaseModel):
    workflowjson: dict
    refined_prompt : str

class ApiKeys(BaseModel):
    composio: str

class Tool(BaseModel):
    service: str
    

class General(BaseModel):
    query: str
    session_id: str

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k_per_file: int = 5
    final_top_k: int = 10
    similarity_threshold: float = 0.0

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
    results: List[RetrievalResult]
    total_files_searched: int
    query_time_ms: float

class CollectionUpdate(BaseModel):
    collection_name: str
    type: str  # "files", "url", or "databases"
    new_data: list

groq = Groq()
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

clerk_issuer = os.getenv("CLERK_ISSUER")
clerk_jwks_url = os.getenv("CLERK_JWKS_URL")
clerk_secret_key = os.getenv("CLERK_SECRET_KEY")
clerk_sdk = Clerk(bearer_auth=clerk_secret_key)





# app = FastAPI(lifespan=lifespan)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust according to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(CustomProxyHeadersMiddleware)
# app.add_middleware(HTTPSRedirectMiddleware)
security = HTTPBearer()
def get_jwks():
    response = requests.get(clerk_jwks_url)
    return response.json()
def get_public_key(kid):
    jwks = get_jwks()
    for key in jwks['keys']:
        if key['kid'] == kid:
            return jwk.construct(key)
    raise HTTPException(status_code=401, detail="Invalid token")
def decode_token(token: str):
    headers = jwt.get_unverified_headers(token)
    kid = headers['kid']
    public_key = get_public_key(kid)
    # print("public_key",public_key)
    # First decode without verification to get the audience
    unverified_claims = jwt.get_unverified_claims(token)
    # print("unverified_claims",unverified_claims)
    audience = unverified_claims.get('sub')
    token_issuer = unverified_claims.get('iss')
    # print("audience", audience)
    # print("Token issuer:", token_issuer)
    # print("Expected issuer:", clerk_issuer)
    return jwt.decode(token, public_key.to_pem().decode('utf-8'), algorithms=['RS256'], audience=audience , issuer=token_issuer)










class PaymentRequest(BaseModel):
    amount: int

@app.post("/pay_init")
async def initialize_payment(payment_request: PaymentRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # Initialize Razorpay client
        client = razorpay.Client(auth=(str(os.getenv("RZP_KEY")), str(os.getenv("RZP_SECRET"))))
        
        # Convert amount to paise (Razorpay requires amount in smallest currency unit)
        amount_in_paise = int(payment_request.amount * 100)
        # amount_in_paise = 100  # For testing, set to 100 paise = ₹1
        
        # Create unique receipt ID
        # receipt = f"receipt_{user_id}_{uuid.uuid4().hex[:6]}"
        
        # Create payment order
        data = {
            "amount": amount_in_paise,
            "currency": "INR",
            # "receipt": receipt
        }
        
        payment = client.order.create(data=data)

        # Store payment initialization data in DynamoDB payments table
        try:
            # Get user email from users table
            user_info = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}}
            )
            
            user_email = user_info.get('Item', {}).get('email', {}).get('S', '')
            
            # Save payment information in the payments table
            db_client.put_item(
                TableName='payment',
                Item={
                    'order_id': {'S': payment["id"]},
                    'user_id': {'S': user_id},
                    'email': {'S': user_email},
                    'amount': {'N': str(payment_request.amount)},
                    'currency': {'S': 'INR'},
                    'status': {'S': 'initialized'},
                    # 'receipt': {'S': receipt},
                    # 'created_at': {'S': datetime.now().isoformat()}
                }
            )
            
            logger.info(f"Payment initialization saved to payments table: {payment['id']}")
        except Exception as e:
            logger.error(f"Error saving payment information to database: {str(e)}")
            # Continue with the payment process even if saving to DB fails
        
        return {
            "status": "success",
            "order_id": payment["id"],
            "amount": amount_in_paise,
            "currency": "INR"
        }
    
    except Exception as e:
        logger.error(f"Error initializing payment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing payment: {str(e)}")


class PaymentCallback(BaseModel):
    razorpay_payment_id: str
    razorpay_order_id: str
    razorpay_signature: str
    
@app.post("/payment_callback")
async def payment_callback(
    razorpay_payment_id: str = Form(...),
    razorpay_order_id: str = Form(...),
    razorpay_signature: str = Form(...)
):
    # Create PaymentCallback object from form data
    payment_data = PaymentCallback(
        razorpay_payment_id=razorpay_payment_id,
        razorpay_order_id=razorpay_order_id,
        razorpay_signature=razorpay_signature
    )
    try:
        # Initialize Razorpay client
        client = razorpay.Client(auth=(os.getenv("RZP_KEY"), os.getenv("RZP_SECRET")))
        
        # Verify the payment signature
        params_dict = {
            'razorpay_order_id': payment_data.razorpay_order_id,
            'razorpay_payment_id': payment_data.razorpay_payment_id,
            'razorpay_signature': payment_data.razorpay_signature
        }
        
        # Verify that the payment data hasn't been tampered with
        client.utility.verify_payment_signature(params_dict)
        
        # Fetch payment details from Razorpay
        payment = client.payment.fetch(payment_data.razorpay_payment_id)
        
        
        # Extract relevant information
        # Get order ID from the payment response
        order_id = payment.get("order_id")
        
        # Fetch payment record from DynamoDB payments table using order_id
        payment_record = db_client.get_item(
            TableName='payment',
            Key={'order_id': {'S': order_id}}
        )
        
        # Extract user_id from the payment record
        user_id = payment_record.get('Item', {}).get('user_id', {}).get('S')
        plan = payment.get("notes", {}).get("plan_type")
        
        if not user_id or not plan:
            logger.error("User ID or plan missing in payment notes")
            return {"status": "error", "message": "User ID or plan missing in payment"}
        
        # Update user's plan in DynamoDB
        # Update user's plan in users table
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET #plan = :plan",
            ExpressionAttributeNames={
            '#plan': 'plan'
            },
            ExpressionAttributeValues={
            ':plan': {'S': plan}
            }
        )
        
        # Update payment status in payments table
        db_client.update_item(
            TableName='payment',
            Key={'order_id': {'S': order_id}},
            UpdateExpression="SET #status = :status, payment_id = :payment_id, completed_at = :timestamp",
            ExpressionAttributeNames={
            '#status': 'status'
            },
            ExpressionAttributeValues={
            ':status': {'S': 'completed'},
            ':payment_id': {'S': payment_data.razorpay_payment_id},
            ':timestamp': {'S': datetime.now().isoformat()}
            }
        )
        
        logger.info(f"Payment successful: Updated user {user_id} to plan {plan}")
        
        # Return a well-designed HTML success page
        success_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Payment Successful</title>
            <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f7fa;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            .container {{
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                padding: 40px;
                text-align: center;
                max-width: 500px;
                width: 90%;
            }}
            .success-icon {{
                color: #4CAF50;
                font-size: 48px;
                margin-bottom: 20px;
            }}
            h1 {{
                color: #333;
                margin-bottom: 15px;
            }}
            p {{
                color: #666;
                line-height: 1.6;
                margin-bottom: 20px;
            }}
            .plan-badge {{
                background-color: #e3f2fd;
                color: #1976d2;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                display: inline-block;
                margin-bottom: 20px;
            }}
            .button {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                font-weight: bold;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                transition: background-color 0.3s;
            }}
            .button:hover {{
                background-color: #3e8e41;
            }}
            </style>
        </head>
        <body>
            <div class="container">
            <div class="success-icon">✓</div>
            <h1>Payment Successful!</h1>
            <div class="plan-badge">{plan.upper()} Plan</div>
            <p>Your payment has been processed successfully. Thank you for upgrading to the {plan} plan!</p>
            <p>You now have access to all the premium features of Sigmoyd.</p>
            <a href="https://app.sigmoyd.in" class="button">Use Sigmoyd Whatsapp</a>
            <a href="https://app.sigmoyd.in/premade-workflows" class="button">Explore workflows</a>
            <a href="https://app.sigmoyd.in/manage-auths" class="button">Tools and integrations</a>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=success_html)
        
    except razorpay.errors.SignatureVerificationError:
        logger.error("Payment signature verification failed")
        raise HTTPException(status_code=400, detail="Payment signature verification failed")
    except Exception as e:
        logger.error(f"Error processing payment callback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing payment: {str(e)}")






bucket_name = "ragtestsigmoyd"
# Helper functions
def get_embedding_from_query(query: str) -> np.ndarray:
    """Convert query to embedding vector using Google Gemini"""
    try:
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="RETRIEVAL_QUERY"  # Use RETRIEVAL_QUERY for queries
        )["embedding"]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating query embedding: {str(e)}")

async def list_project_files(user_id: str, project: str) -> List[str]:
    """List all files in a user's project directory"""
    try:
        prefix = f"{user_id}/{project}/"
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
       
        if 'Contents' not in response:
            return []
       
        # Extract unique file base names (without _metadata.json or _index.faiss suffixes)
        file_bases = set()
        for obj in response['Contents']:
            key = obj['Key']
            filename = key.split('/')[-1]  # Get just the filename
           
            if filename.endswith('_metadata.json'):
                base_name = filename.replace('_metadata.json', '')
                file_bases.add(base_name)
            elif filename.endswith('_index.faiss'):
                base_name = filename.replace('_index.faiss', '')
                file_bases.add(base_name)
       
        return list(file_bases)
   
    except Exception as e:
        logger.error(f"Error listing project files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accessing project files: {str(e)}")

async def load_file_index_and_metadata(user_id: str, project: str, file_base: str) -> tuple:
    """Load FAISS index and metadata for a specific file"""
    try:
        # S3 keys for the file's index and metadata
        index_key = f"{user_id}/{project}/{file_base}_index.faiss"
        metadata_key = f"{user_id}/{project}/{file_base}_metadata.json"
       
        # Download files to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp_index:
            s3_client.download_fileobj(bucket_name, index_key, tmp_index)
            tmp_index_path = tmp_index.name
       
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_metadata:
            s3_client.download_fileobj(bucket_name, metadata_key, tmp_metadata)
            tmp_metadata_path = tmp_metadata.name
       
        # Load FAISS index
        index = faiss.read_index(tmp_index_path)
       
        # Load metadata
        with open(tmp_metadata_path, 'r') as f:
            metadata = json.load(f)
       
        # Cleanup temp files
        os.unlink(tmp_index_path)
        os.unlink(tmp_metadata_path)
       
        return index, metadata, file_base
   
    except Exception as e:
        logger.error(f"Error loading index/metadata for {file_base}: {str(e)}")
        return None, None, file_base

async def search_single_file(index, metadata: Dict, file_base: str, query_embedding: np.ndarray,
                           top_k: int, project: str) -> List[RetrievalResult]:
    """Search a single FAISS index and return top-k results"""
    try:
        if index is None or metadata is None:
            return []
       
        # Perform similarity search
        scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
       
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
               
            # Get metadata for this chunk
            chunk_metadata = metadata.get(str(idx), {})
           
            result = RetrievalResult(
                content=chunk_metadata.get('full_content', ''),
                summary=chunk_metadata.get('summary', ''),
                section_title=chunk_metadata.get('section_title', ''),
                section_index=chunk_metadata.get('section_index', idx),
                score=float(score),
                source_file=file_base,
                original_file=chunk_metadata.get('original_file', file_base),
                s3_key=chunk_metadata.get('s3_key', ''),
                project=project
            )
            results.append(result)
       
        return results
   
    except Exception as e:
        logger.error(f"Error searching file {file_base}: {str(e)}")
        return []

def rerank_results(all_results: List[RetrievalResult], final_top_k: int, query: str = None) -> List[RetrievalResult]:
    """Rerank and filter results across all files with enhanced scoring"""
    if not all_results:
        return []
   
    # For Gemini embeddings, higher cosine similarity = better match
    # Sort by similarity score (higher is better)
    sorted_results = sorted(all_results, key=lambda x: x.score, reverse=True)
   
    # Optional: Add additional ranking factors
    if query:
        # You could add text-based relevance scoring here
        # For example, boost results where section_title matches query terms
        query_lower = query.lower()
        for result in sorted_results:
            title_boost = 0.1 if any(word in result.section_title.lower() for word in query_lower.split()) else 0
            result.score += title_boost
       
        # Re-sort after boosting
        sorted_results.sort(key=lambda x: x.score, reverse=True)
   
    return sorted_results[:final_top_k]




# create function for fetching from database, csv AI, links too and use it inside below route
# try it by invoking from frontend
# see how async works
@app.post("/query/{project}", response_model=QueryResponse)
async def query_project(
    project: str,
    query_request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Query all files in a user's project and return ranked results
    """
    import time
    start_time = time.time()
   
    # Extract user ID from token
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("sub")
   
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
   
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
       
        # Load all indices and metadata in parallel
        with ThreadPoolExecutor(max_workers=min(len(file_bases), 10)) as executor:
            load_tasks = [
                executor.submit(asyncio.run, load_file_index_and_metadata(user_id, project, file_base))
                for file_base in file_bases
            ]
            loaded_data = [task.result() for task in load_tasks]
       
        # Search each file in parallel
        search_tasks = []
        for index, metadata, file_base in loaded_data:
            if index is not None:
                task = search_single_file(
                    index, metadata, file_base, query_embedding,
                    query_request.top_k_per_file, project
                )
                search_tasks.append(task)
       
        # Execute all searches
        file_results = await asyncio.gather(*search_tasks)
       
        # Flatten results from all files
        all_results = []
        for results in file_results:
            all_results.extend(results)
       
        # Filter by similarity threshold
        filtered_results = [
            result for result in all_results
            if result.score >= query_request.similarity_threshold
        ]
       
        # Rerank across all files
        final_results = rerank_results(filtered_results, query_request.final_top_k, query_request.query)
       
        query_time = round((time.time() - start_time) * 1000, 2)
       
        return QueryResponse(
            results=final_results,
            total_files_searched=len([data for data in loaded_data if data[0] is not None]),
            query_time_ms=query_time
        )
   
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/get_collections")
async def get_collections(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # Get the user item from DynamoDB
        response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
        
        # Check if the user exists
        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if collections attribute already exists
        if 'collections' in response['Item']:
            # Return the existing collections
            collections_map = response['Item']['collections'].get('M', {})
            collections = {}
            
            # Convert from DynamoDB format to regular JSON
            for collection_name, collection_data in collections_map.items():
                collection_obj = {}
                data = collection_data.get('M', {})
                
                # Convert files list
                if 'files' in data:
                    collection_obj['files'] = [item.get('S', '') for item in data['files'].get('L', [])]
                
                # Convert URLs list
                if 'url' in data:
                    collection_obj['url'] = [item.get('S', '') for item in data['url'].get('L', [])]
                
                # Convert databases list
                if 'databases' in data:
                    databases = []
                    for db_item in data['databases'].get('L', []):
                        db_map = db_item.get('M', {})
                        databases.append({
                            'name': db_map.get('name', {}).get('S', ''),
                            'host': db_map.get('host', {}).get('S', ''),
                            'password': db_map.get('password', {}).get('S', '')
                        })
                    collection_obj['databases'] = databases
                
                collections[collection_name] = collection_obj
            return {"collections": collections}
        
        # If collections attribute doesn't exist, create it with sample data
        default_collections = {
            # "Research Project": {
            #     "files": ["dataset.csv", "analysis.pdf", "notes.docx", "results.xlsx"],
            #     "url": ["https://www.researchgate.net/papers", "https://scholar.google.com"],
            #     "databases": [
            #         {"name": "PostgreSQL", "host": "db.research.local", "password": "••••••••"},
            #         {"name": "AWS S3", "access_key": "AKIAXXXXXXXX", "secret_key": "••••••••"}
            #     ]
            # },
            # "Marketing Content": {
            #     "files": ["branding_guide.pdf", "logo.png", "presentation.pptx"],
            #     "url": ["https://www.behance.net/gallery", "https://dribbble.com/shots"],
            #     "databases": [
            #         {"name": "MySQL", "host": "marketing-db.cloud", "password": "••••••••"}
            #     ]
            # },
            # "Financial Data": {
            #     "files": ["q1_report.pdf", "expenses.csv", "revenue.xlsx"],
            #     "url": ["https://finance.yahoo.com", "https://www.bloomberg.com/markets"],
            #     "databases": [
            #         {"name": "MongoDB", "host": "finance.mongo.cloud", "password": "••••••••"},
            #         {"name": "AWS RDS", "access_key": "AKIAXXXXXXXX", "secret_key": "••••••••"}
            #     ]
            # }
        }
        
        # Convert regular JSON to DynamoDB format
        collections_map = {}
        for collection_name, collection_data in default_collections.items():
            collection_obj = {}
            
            # Convert files list
            if 'files' in collection_data:
                collection_obj['files'] = {'L': [{'S': file} for file in collection_data['files']]}
            
            # Convert URLs list
            if 'url' in collection_data:
                collection_obj['url'] = {'L': [{'S': url} for url in collection_data['url']]}
            
            # Convert databases list
            if 'databases' in collection_data:
                databases_list = []
                for db in collection_data['databases']:
                    db_map = {
                        'name': {'S': db.get('name', '')},
                        'host': {'S': db.get('host', '')},
                        'password': {'S': db.get('password', '')}
                    }
                    if 'access_key' in db:
                        db_map['access_key'] = {'S': db['access_key']}
                    if 'secret_key' in db:
                        db_map['secret_key'] = {'S': db['secret_key']}
                    databases_list.append({'M': db_map})
                
                collection_obj['databases'] = {'L': databases_list}
            
            collections_map[collection_name] = {'M': collection_obj}
        
        # Update the user item with the new collections attribute
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET collections = :collections",
            ExpressionAttributeValues={
                ':collections': {'M': collections_map}
            }
        )
        
        return {"collections": default_collections}
        
    except Exception as e:
        logger.error(f"Error processing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing collections: {str(e)}")




def trigger_lambda(
    project: str,
    files : list,
    user_id : str
):
    
    response = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}}
            )
    api_keys = response.get('Item', {}).get('api_key', {}).get('M', {})
    final_dict = {k: v['S'] for k, v in api_keys.items()}
    try:
        lambda_payload = {
            "user_id": user_id,
            "project": project,
            "gemini_api_key" : final_dict["gemini"],
            "files" : files
        }

        response = lambda_client.invoke(
            FunctionName="lambdahandlerr", 
            InvocationType="RequestResponse", 
            Payload=json.dumps(lambda_payload).encode()
        )
        print("Lambda response:", response)
        return {"message": "Lambda invoked", "status_code": response['StatusCode']}
    except Exception as e:
        logger.error(f"Failed to invoke Lambda: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to invoke Lambda")
    



# takes collection_name, type, new_data
# call this after file_upload. new_link or new database
async def update_collection(
    collection_data: CollectionUpdate,
    user_id: str 
):

    collection_name = collection_data.collection_name
    update_type = collection_data.type
    new_data = collection_data.new_data
    
    if update_type not in ["files", "url", "databases"]:
        raise HTTPException(status_code=400, detail="Invalid type. Must be 'files', 'url', or 'databases'")
    
    try:
        # First check if the user exists and has collections
        response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
        
        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert the new_data to DynamoDB format based on type
        if update_type in ["files", "url"]:
            # For files and url, convert list of strings to DynamoDB list of strings
            dynamo_data = {'L': [{'S': item} for item in new_data]}
        elif update_type == "databases":
            # For databases, convert list of dicts to DynamoDB list of maps
            dynamo_data = {'L': []}
            for db in new_data:
                db_map = {'M': {}}
                for key, value in db.items():
                    db_map['M'][key] = {'S': str(value)}
                dynamo_data['L'].append(db_map)
        
        # Check if collections and the specific collection exist
        if 'collections' not in response['Item'] or collection_name not in response['Item']['collections'].get('M', {}):
            # Create new collection with the specified data
            update_expression = "SET collections.#cn = :collection_data"
            expression_attribute_names = {
                '#cn': collection_name
            }
            collection_map = {'M': {update_type: dynamo_data}}
            expression_attribute_values = {
                ':collection_data': collection_map
            }
        else:
            # Check if the specified type exists in the collection
            collection_data = response['Item']['collections']['M'][collection_name]
            
            if update_type in collection_data.get('M', {}):
            # Append to existing data rather than replacing it
                if update_type in ["files", "url"]:
                    # For files and urls, append to the existing list
                    update_expression = "SET collections.#cn.#type = list_append(collections.#cn.#type, :new_data)"
                    expression_attribute_names = {
                    '#cn': collection_name,
                    '#type': update_type
                    }
                    expression_attribute_values = {
                    ':new_data': dynamo_data
                    }
                elif update_type == "databases":
                    # For databases, we need to merge the two lists
                    existing_databases = collection_data['M'].get(update_type, {}).get('L', [])
                    new_databases = dynamo_data['L']
                    merged_databases = {'L': existing_databases + new_databases}
                    
                    update_expression = "SET collections.#cn.#type = :merged_data"
                    expression_attribute_names = {
                    '#cn': collection_name,
                    '#type': update_type
                    }
                    expression_attribute_values = {
                    ':merged_data': merged_databases
                    }
                else:
                    # Type doesn't exist yet, so create it
                    update_expression = "SET collections.#cn.#type = :type_data"
                    expression_attribute_names = {
                        '#cn': collection_name,
                        '#type': update_type
                    }
                    expression_attribute_values = {
                        ':type_data': dynamo_data
                    }
        
        # Update the DynamoDB item
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values
        )
        
        return {
            "status": "success", 
            "message": f"Collection '{collection_name}' updated successfully",
            "collection_name": collection_name,
            "type_updated": update_type
        }
        
    except Exception as e:
        logger.error(f"Error updating collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating collection: {str(e)}")



# what happens if i upload csv, pdf, docx, txt, json, xlsx, pptx, mp4, mp3, zip, how labmda handles alll?
@app.post("/collection_file_upload")
async def upload_user_file(
    request: Request,
    project: str = Form("default"),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("sub")    

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    form = await request.form()
    files = form.getlist("file")  # Get all files with name "file"

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    uploaded_files = []
    
    for file in files:
        file_content = await file.read()
        original_file_name = file.filename
        filename_without_ext, ext = os.path.splitext(original_file_name)
        unique_file_name = f"{filename_without_ext}_{uuid.uuid4()}{ext}"

        s3_key = f"{user_id}/{project}/{unique_file_name}"

        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=file_content
            )
            uploaded_files.append(unique_file_name)
        except Exception as e:
            logger.error(f"Error uploading file {original_file_name} to S3: {str(e)}")
            # Continue with other files even if one fails
    
    if not uploaded_files:
        raise HTTPException(status_code=500, detail="Failed to upload any files")
    
    # Update the collection with the newly uploaded files
    await update_collection(
        CollectionUpdate(
            collection_name=project,
            type="files",
            new_data=uploaded_files
        ),
        user_id
    )
        
    # Invoke lambda with all uploaded files
    return trigger_lambda(project, uploaded_files, user_id)



@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)  # Policy Violation
        return
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=1008)
            return
    except Exception as e:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    
    print("websocket endpoint accepted for user_id:", user_id)

    try:
        redis = Redis(host="redis", port=6379, db=0, decode_responses=True)

        channel = f"workflow_{user_id}"
        last_id="$"
    except Exception as e:
        print("Auth or Redis Error:", e)
        await websocket.close()

    try:
        while True:
            response = await redis.xread(
                streams={channel: last_id},
                block=5000,  # wait max 5s
                count=10     # read up to 10 messages
            )

            if response:
                for msg_id, data in response[0][1]:
                    data['data'] = json.loads(data['data'])
                    await websocket.send_json(data)

    except WebSocketDisconnect:
        print(f"❌ WebSocket disconnected: user_id={user_id}")
    finally:
        await redis.close()


@app.post("/file_upload")
async def file_upload(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    form = await request.form()
    file = form.get("file")
    
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    file_content = await file.read()
    original_file_name = file.filename
    # Extract original filename from the S3 key

    filename_without_ext, ext = os.path.splitext(original_file_name)

    # Create new filename with _updated suffix
    unique_file_name = f"{filename_without_ext}_{uuid.uuid4()}{ext}"

    
   
    
    
    # Save file information to the database
    try:
        bucket_name = "workflow-files-2709"  # Replace with your S3 bucket name
        s3_client.put_object(
            Bucket=bucket_name,
            Key=unique_file_name,
            Body=file_content
        )
        return {"file_location": f"s3://{bucket_name}/{unique_file_name}"}
    except Exception as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file to S3 or database: {str(e)}")
        
    
    



class FileDelete(BaseModel):
    file_location: str

@app.post("/file_delete")
async def file_delete(file_data: FileDelete, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    bucket_name = "workflow-files-2709"
    file_location = file_data.file_location
    
    try:
        # Check if the file exists in S3
        try:
            s3_client.head_object(Bucket=bucket_name, Key=file_location)
        except Exception as e:
            # If file doesn't exist, return success anyway
            return {"status": "success", "message": "File deleted or did not exist"}
        
        # Delete the file if it exists
        s3_client.delete_object(
            Bucket=bucket_name,
            Key=file_location
        )
        
        return {"status": "success", "message": "File deleted successfully"}
    except Exception as e:
        # Log the error but return success anyway
        logger.error(f"Error during file deletion: {str(e)}")
        return {"status": "success", "message": "File deleted or did not exist"}


@app.post("/checkuser")
async def check_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # print(f"Checking user_id: {user_id}")

        # Debugging: List all table names
        # tables = db_client.list_tables()
        # print(f"Available tables: {tables}")
        # response = db_client.describe_table(TableName="users")
        # print(response["Table"]["KeySchema"])

        response = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )



        # Ensure api_key attribute exists before trying to update gemini key
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET api_key = if_not_exists(api_key, :empty_map)",
            ExpressionAttributeValues={
                ':empty_map': {'M': {}}
            }
        )
        
        # Then update the gemini key
        gemini_key = os.getenv("GOOGLE_API_KEY")
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET api_key.#gemini = :val",
            ExpressionAttributeNames={
                '#gemini': 'gemini'
            },
            ExpressionAttributeValues={
                ':val': {'S': gemini_key}
            }
        )
        

        # print("Response from DynamoDB:", response)  # ✅ Should print response
        try  :
            clerk_sdk = Clerk(bearer_auth=clerk_secret_key)
            user_details = clerk_sdk.users.list(user_id=[user_id])[0]
            email = user_details.email_addresses[0].email_address
        except Exception as e:
            email = ""
        
        if "Item" not in response:
            db_client.put_item(
                TableName="users",
                Item={
                    "clerk_id": {"S": user_id},
                    "plan": {"S": "free"},
                    "api_key": {"M": {}},
                    "email": {"S": str(email)},
                    # "gmailtrigger": {"M": {}}
                },
            )
            print(f"User {user_id} created in DynamoDB")
            return {"status": "success", "message": "User created successfully"}
        else:
            print(f"User {user_id} already exists in DynamoDB")
            return {"status": "success", "message": "User already exists"}

        # table_name = 'custom_credentials'

        # # Check if the table exists
        # existing_tables = db_client.list_tables()['TableNames']

        # if table_name not in existing_tables:
        #     # Create table
        #     response = db_client.create_table(
        #         TableName=table_name,
        #         KeySchema=[
        #             {'AttributeName': 'mail', 'KeyType': 'HASH'}  # Primary Key (Partition Key)
        #         ],
        #         AttributeDefinitions=[
        #             {'AttributeName': 'mail', 'AttributeType': 'S'}  # String type
        #         ],
        #         ProvisionedThroughput={
        #             'ReadCapacityUnits': 5,
        #             'WriteCapacityUnits': 5
        #         }
        #     )
        #     print(f"Table '{table_name}' is being created...")
        # else:
        #     print(f"Table '{table_name}' already exists.")

    except Exception as e:
        print(f"Error: {e}")  # ✅ Print error details
        raise HTTPException(status_code=500, detail=f"Error accessing DynamoDB: {str(e)}")




@app.post("/save_workflow")
async def save_flow(w:W,credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    try:
        response = db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET workflows.#wid.#json = :new_json",
            ExpressionAttributeNames={
            '#wid': w.workflowjson["workflow_id"],
            '#json': 'json'
            },
            ExpressionAttributeValues={
            ':new_json': {'S': json.dumps(w.workflowjson, indent=2)}
            },
            ReturnValues="UPDATED_NEW"
        )
        # print("Update succeeded:", response)
    except Exception as e:
        print("Error updating item:", e)
        
    # return {"status": "success"}
    return {"json":w.workflowjson}

def clean_llm_json_response(raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)

@app.post("/public_workflow")
async def public_workflow(w: WP, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # Save the workflow to the public_workflows table
        workflow_id = w.workflowjson["workflow_id"]
        refined_prompt = w.refined_prompt
        
        # If this workflow contains FILE_UPLOAD tools, reset their config_inputs
        workflow_json = w.workflowjson
        for agent in workflow_json.get("workflow", []):
            if agent.get("type") == "tool" and agent.get("name") == "FILE_UPLOAD":
                agent["config_inputs"] = {}

        # Save the updated workflow with empty FILE_UPLOAD config_inputs
        allowed_categories = ["trading","data science","data entry", "marketing", "sales", "email automation", "customer support", "finance", "human resources", "project management", "development", "design", "operations", "education", "healthcare", "legal", "real estate", "travel", "e-commerce", "social media", "content creation", "analytics", "security", "it administration", "research", "personal productivity"]

        public_desc_and_category = get_chain(user_id, gemini_prompt,{"prompt": f"act as a workflow analyser. Give a short and simple explaination what is happening in the following workflow. dont explain each step but give overall idea of whet is happening and for whom this workflow is. also mention the uses, potential and industry it is used in. The explaination should be crisp, simple and attention grabbing. no need to explain each tool and function used. Also, provide a single, concise category for this workflow from the following list: {', '.join(allowed_categories)}. If none of the categories fit, use 'uncategorized'. Format your response as a json with keys \"description\" and \"category\"\n\nMake sure no preambles and postambles are required\nWorkflow steps: {refined_prompt}"})
        
        # Parse the LLM's response
        if public_desc_and_category[0]=="`":
            public_desc_and_category=json.loads(public_desc_and_category[7:-4])
        else:
            public_desc_and_category=json.loads(public_desc_and_category)

        public_desc = public_desc_and_category.get("description", "")
        best_category = public_desc_and_category.get("category", "").lower() if public_desc_and_category.get("category", "").lower() in allowed_categories else "uncategorized" # Default to uncategorized if LLM fails to provide or provides an invalid category
        
        print(f"Generated public_desc: {public_desc}")
        print(f"Assigned category: {best_category}")

        db_client.put_item(
            TableName='public_workflows',
            Item={
                'wid': {'S': workflow_id},
                'clerk_id': {'S': user_id},
                'refined_prompt': {'S': refined_prompt},
                'description': {'S': public_desc},
                'json': {'S': json.dumps(w.workflowjson, indent=2)},
                'uses': {'N': '0'},
                'likes': {'N': '0'},
                'comments': {'L': []},
                'category': {'S': best_category}
            }
        )
        
        print(f"Workflow {workflow_id} added to public workflows")
        
        try:
            response = db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}},
                UpdateExpression="SET workflows.#wid.#pub = :is_public",
                ExpressionAttributeNames={
                    '#wid': w.workflowjson["workflow_id"],
                    '#pub': 'public'
                },
                ExpressionAttributeValues={
                    ':is_public': {'BOOL': True}
                }
            )
        except Exception as e:
            print("Error updating item:", e)
        
        return {"json": w.workflowjson}
    
    except Exception as e:
        print(f"Error in public_workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving public workflow: {str(e)}")


@app.get("/fetch_public")
async def fetch_public(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    try:
        # Scan the public_workflows table to get all items
        response = db_client.scan(TableName='public_workflows')
        
        if 'Items' not in response:
            return {"workflows": []}
        
        # Format the response
        public_workflows = []
        for item in response['Items']:
            workflow = {
                'wid': item.get('wid', {}).get('S', ''),
                'json': json.loads(item.get('json', {}).get('S', '{}')),
                'uses': int(item.get('uses', {}).get('N', '0')),
                'likes': int(item.get('likes', {}).get('N', '0')),
                'comments': [comment.get('S', '') for comment in item.get('comments', {}).get('L', [])],
                'category': item.get('category', {}).get('S', 'uncategorized')
            }
            public_workflows.append(workflow)
        
        return {"workflows":public_workflows}
    
    except Exception as e:
        print(f"Error fetching public workflows: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching public workflows: {str(e)}")

@app.get("/fetch_public_by_category/{category_name}")
async def fetch_public_by_category(category_name: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    try:
        response = db_client.scan(
            TableName='public_workflows',
            FilterExpression="#cat = :category_name",
            ExpressionAttributeNames={'#cat': 'category'},
            ExpressionAttributeValues={':category_name': {'S': category_name}}
        )
        
        if 'Items' not in response:
            return {"workflows": []}
        
        public_workflows = []
        for item in response['Items']:
            workflow = {
                'wid': item.get('wid', {}).get('S', ''),
                'json': json.loads(item.get('json', {}).get('S', '{}')),
                'uses': int(item.get('uses', {}).get('N', '0')),
                'likes': int(item.get('likes', {}).get('N', '0')),
                'comments': [comment.get('S', '') for comment in item.get('comments', {}).get('L', [])],
                'category': item.get('category', {}).get('S', 'uncategorized')
            }
            public_workflows.append(workflow)
        
        return {"workflows": public_workflows}
    
    except Exception as e:
        print(f"Error fetching public workflows by category: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching public workflows by category: {str(e)}")

@app.delete("/delete_workflow/{workflow_id}")
async def delete_workflow(workflow_id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # Fetch user data
        user_data = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )
        
        # Check if workflows exist
        workflows = user_data.get("Item", {}).get("workflows", {}).get("M", {})
        if workflow_id not in workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Delete the workflow
        response = db_client.update_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}},
            UpdateExpression="REMOVE workflows.#wid",
            ExpressionAttributeNames={
                "#wid": workflow_id
            },
            ReturnValues="UPDATED_NEW"
        )
        # print("Workflow deleted successfully:", response)
        return {"status": "success", "message": "Workflow deleted successfully"}
    
    except Exception as e:
        print("Error deleting workflow:", e)
        raise HTTPException(status_code=500, detail=f"Error deleting workflow: {str(e)}")

@app.post("/save_api_keys")
async def save_api_keys(api_keys:ApiKeys,credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    # api_keys.gemini = os.getenv("GOOGLE_API_KEY")
    api_key_dict = api_keys.dict()
    api_key_dict["gemini"] = os.getenv("GOOGLE_API_KEY")
    try:
        # print(api_keys.dict().items())
        response = db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET api_key = :new_api_keys",
            ExpressionAttributeValues={
            ':new_api_keys': {'M': {k: {'S': v} for k, v in api_key_dict.items() if v!=None }}
            },
            ReturnValues="UPDATED_NEW"
        )
        # print("Update succeeded:", response)
    except Exception as e:
        print("Error updating item:", e)
        
    return {"status": "success"}


@app.get("/sidebar_workflows")
async def get_sidebar_workflows(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    user_data = db_client.get_item(
        TableName="users",
        Key={"clerk_id": {"S": user_id}}
    )
    
    workflows = user_data.get("Item", {}).get("workflows", {}).get("M", {})
    
    formatted_workflows = []
    
    for item in workflows.items():
        wid = item[0]
        workflow = item[1]
        
        # Check if we have a third element (public flag)
        public = False
        if len(item) == 3:
            public = item[2]
        elif "public" in workflow.get("M", {}):
            public = workflow.get("M", {}).get("public", {}).get("BOOL", False)
        
        formatted_workflows.append({
            "id": wid,
            "name": json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("workflow_name", ""),
            "json": workflow.get("M", {}).get("json", {}).get("S", "{}"),
            "prompt": workflow.get("M", {}).get("prompt", {}).get("S", ""),
            "active": json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("active", False),
            "public": public
        })
    # print(formatted_workflows)

    return formatted_workflows



@app.post("/run_workflow")
async def run_workflow(w:W,credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    w.workflowjson["active"]=True
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    user_data = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )
        
    plan = user_data.get("Item", {}).get("plan", {}).get("S", "")
    api_key = user_data.get("Item", {}).get("api_key", {}).get("M", {})
    
    if plan == "free" and not api_key:
        return JSONResponse(content={"status": "error", "message": "Please fill in your API keys to proceed."}, status_code=400)
    
    # task = syn.delay(user_id, w.workflowjson)
    # trigger=w.workflowjson["trigger"]
    tools=[i["name"].lower() for i in w.workflowjson["workflow"] if i["type"]=="tool" and i["name"].upper() in composio_tools]
    # print(trigger)
    response = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}}
            )
    api_keys = response.get('Item', {}).get('api_key', {}).get('M', {})
    final_dict = {k: v['S'] for k, v in api_keys.items()}
    
    for i in tools:
        if not check_connection(i, final_dict["composio"]):
            return JSONResponse(content={"status": "error", "message": "Please fill in your API keys to proceed."}, status_code=400)

    try:
        response = db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET workflows.#wid.#json = :new_json",
            ExpressionAttributeNames={
            '#wid': w.workflowjson["workflow_id"],
            '#json': 'json'
            },
            ExpressionAttributeValues={
            ':new_json': {'S': json.dumps(w.workflowjson, indent=2)}
            },
            ReturnValues="UPDATED_NEW"
        )
        # print("Update succeeded:", response)
    except Exception as e:
        print("Error updating item:", e)
    task = syn.delay(w.workflowjson["workflow_id"], w.workflowjson["workflow"], user_id, "")
    
    return {"json": w.workflowjson}
    


    



@app.post("/activate_workflow")
async def activate_workflow(w:W,credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    if not w.workflowjson["active"]:
        
        user_data = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )
        
        plan = user_data.get("Item", {}).get("plan", {}).get("S", "")
        api_key = user_data.get("Item", {}).get("api_key", {}).get("M", {})
        api_key = {k: v['S'] for k, v in api_key.items()}
        
        if not api_key.get("composio"):
            return JSONResponse(content={"status": "error", "message": "Please fill in your API keys to proceed."}, status_code=400)

        # task = syn.delay(user_id, w.workflowjson)
        trigger=w.workflowjson["trigger"]
        tools=[i["name"].lower() for i in w.workflowjson["workflow"] if i["type"]=="tool" and i["name"].upper() in composio_tools]
        # for tool in w.workflowjson["workflow"]:
        #     if tool["type"] == "tool" and tool["name"].upper() in composio_tools:
        #         class_name = globals().get(tool["name"].upper())
        #         if class_name:
        #             method_name = tool["tool_action"]
        #             if hasattr(class_name, method_name):
        #                 method = getattr(class_name, method_name)   
        #                 signature = inspect.signature(method)
        #                 required_params = set(signature.parameters.keys()) - {"self"}
        #                 config_inputs_keys = set(tool.get("config_inputs", {}).keys())
        #                 data_flow_inputs_keys = set(tool.get("data_flow_inputs", []))
        #                 missing_params = required_params - (config_inputs_keys | data_flow_inputs_keys)
        #                 for param in missing_params:
        #                     tool.setdefault("config_inputs", {})[param] = ""
        # print(trigger)
        # print(tools)








        if trigger["name"]=="TRIGGER_NEW_GMAIL_MESSAGE":
            # here try to handle all auths, 

            try:
                user_details = clerk_sdk.users.list(user_id=[user_id])[0]
                mail=user_details.email_addresses[0].email_address
                setup_watch(mail)
                # return {"status":"workflow activated"}
            except Exception as e:
                print("Error setting up Gmail watch:", e)
                return JSONResponse(content={"status": "error", "message": "Please fill in your API keys to proceed."}, status_code=400)
            




        elif trigger["name"]=="TRIGGER_PERIODIC":
            inputs = w.workflowjson["trigger"]["config_inputs"]
            task=f"""
You are a workflow scheduler, who schedules the workflow based on user inputs
Your task is to find out the start_datetime and interval (hrs) from the user inputs and return a dictionary with keys "start_datetime" and "interval"
The response should be a json with keys "start_datetime" and "interval" :-
{{
    "start_datetime": "2023-10-01T10:00:00+05:30",  (default time zone is Indian Standard Time)
    "interval": 24  (in hours)

}}

user inputs: {inputs}
current time in india : {datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()}
current day : {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%A')}

IMPORTANT:-
- The start_datetime must be in future. Example, if the current time is 2025-10-01T09:00:00+05:30, then the start_datetime must be after this time.
- in user inputs, if the user has provided a date in the past, then you need to find the most relevant next start_datetime based on the current time and user inputs.
- Example, if user says "every day at 10 AM" or " start time is 2025-09-30 at 10 AM", and the current time is 2025-10-01T11:00:00+05:30, then the start_datetime should be "2025-10-02T10:00:00+05:30" (next day at 10 AM)
"""
            def call_periodic_activate(): 
                return get_chain(user_id,gemini_prompt,{"prompt": task})
            loop = asyncio.get_event_loop()
            call_periodic_activate_chain= await loop.run_in_executor(None, call_periodic_activate)
            if call_periodic_activate_chain[0]=="`":
                call_periodic_activate_chain=json.loads(call_periodic_activate_chain[7:-4])
            else:
                call_periodic_activate_chain=json.loads(call_periodic_activate_chain)
            start_datetime = call_periodic_activate_chain.get("start_datetime")
            interval = call_periodic_activate_chain.get("interval")
            item = {
                "reminder_id": {"S": w.workflowjson["workflow_id"]},
                "user_id": {"S": user_id},
                "start_time": {"S": start_datetime},
                "status": {"S": "active"},
                "reminder_type": {"S": "trigger"},
                "message": {"S": w.workflowjson["workflow_name"]},
            }
            
            
            item["repeat_interval_hours"] = {"N": str(interval)}  # Store interval as a number

            db_client.put_item(
                TableName="reminders",
                Item=item
            )










        elif trigger["name"]=="TRIGGER_SIGMOYD_WHATSAPP":
            
            # otp verification
            # Check if the user has already verified their WhatsApp number
            response = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}}
            )

            # Check if whatsapp_verified exists and has a value
            if 'whatsapp_verified' not in response.get('Item', {}) or not response['Item']['whatsapp_verified'].get('S'):
                return JSONResponse(
                    content={
                        "status": "error", 
                        "message": "Please verify your WhatsApp number first to use the WhatsApp trigger."
                    },
                    status_code=400
                )

            # If the number is verified, we can proceed with activating the trigger
            # (existing code continues below)



        #tool auths

        response = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}}
            )
        api_keys = response.get('Item', {}).get('api_key', {}).get('M', {})
        final_dict = {k: v['S'] for k, v in api_keys.items()}
        
        for i in tools:
            if i=="whatsapp":
                # check if whatsapp is verified
                if 'whatsapp_verified' not in response.get('Item', {}) or not response['Item']['whatsapp_verified'].get('S'):
                    return JSONResponse(content={"status": "error", "message": "Please verify your WhatsApp number first to use the WhatsApp trigger."}, status_code=400)
            elif not check_connection(i, final_dict["composio"]):
                return JSONResponse(content={"status": "error", "message": "Please fill in your API keys to proceed."}, status_code=400)
        
        w.workflowjson["active"]=True
        try:
            response = db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}},
                UpdateExpression="SET workflows.#wid.#json = :new_json",
                ExpressionAttributeNames={
                '#wid': w.workflowjson["workflow_id"],
                '#json': 'json'
                },
                ExpressionAttributeValues={
                ':new_json': {'S': json.dumps(w.workflowjson, indent=2)}
                },
                ReturnValues="UPDATED_NEW"
            )
            # print("Update succeeded:", response)
        except Exception as e:
            print("Error updating item:", e)









            
    else:
        # deactivate workflow
        w.workflowjson["active"]=False
        # store in database that it is false now
        try:
            response = db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}},
                UpdateExpression="SET workflows.#wid.#json = :new_json",
                ExpressionAttributeNames={
                '#wid': w.workflowjson["workflow_id"],
                '#json': 'json'
                },
                ExpressionAttributeValues={
                ':new_json': {'S': json.dumps(w.workflowjson, indent=2)}
                },
                ReturnValues="UPDATED_NEW"
            )
            # print("Update succeeded:", response)
        except Exception as e:
            print("Error updating item:", e)
        trigger=w.workflowjson["trigger"]
        # print(trigger)

        if trigger["name"]=="TRIGGER_NEW_GMAIL_MESSAGE":
            # delete setup watch
            pass
        elif trigger["name"]=="TRIGGER_PERIODIC":
            # For periodic trigger deactivation, delete any reminders
            try:
                # Delete reminder entries for this workflow ID from the reminders table
                db_client.delete_item(
                    TableName='reminders',
                    Key={'reminder_id': {'S': w.workflowjson["workflow_id"]}}
                )
                print(f"Deleted reminder for workflow ID: {w.workflowjson['workflow_id']}")
            except Exception as e:
                print(f"Error deleting reminder: {e}")
                # Continue with deactivation even if reminder deletion fails


    return {"json":w.workflowjson}
    
        
    
def get_latest_email_id(email_address):
    url = f"https://gmail.googleapis.com/gmail/v1/users/{email_address}/messages"
    
    creds,user_id = get_fresh_google_credentials(email_address)
    headers = {"Authorization": f"Bearer {creds.token}"}

    params = {"maxResults": 1, "labelIds": ["INBOX"]}
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        messages = response.json().get("messages", [])
        if messages:
            return messages[0]["id"],user_id  # ✅ Returns message ID
    return None


def get_email_content(email_address, message_id):
    url = f"https://gmail.googleapis.com/gmail/v1/users/{email_address}/messages/{message_id}"

    creds,user_id = get_fresh_google_credentials(email_address)
    headers = {"Authorization": f"Bearer {creds.token}"}

    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        email_data = response.json()
        return email_data  # Full email content
    
    return None

import base64
import json
import binascii

async def get_email_attachment(message_id: str, attachment_id: str,email_address: str,filename:str) -> str:
    """
    downloads attachment and saves to S3
    """
    try:
        # Get fresh credentials for the user
        creds, user_id = get_fresh_google_credentials(email_address)
        
        # Get attachment via Gmail API
        url = f"https://gmail.googleapis.com/gmail/v1/users/{email_address}/messages/{message_id}/attachments/{attachment_id}"
        headers = {"Authorization": f"Bearer {creds.token}"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to get attachment: {response.status_code} - {response.text}")
            return None
        
        # Extract attachment data
        attachment_data = response.json()
        attachment_content = base64.urlsafe_b64decode(attachment_data.get("data", ""))
        
        # Extract filename without extension if it has one
        filenamewithoutext = os.path.splitext(filename)[0]

        # Generate a unique filename for S3
        filename = f"{user_id}/{filenamewithoutext}_{uuid.uuid4()}"
        
        # Save to S3
        s3_client.put_object(
            Bucket="gmail-attachments-2709",
            Key=filename,
            Body=attachment_content
        )
        
        # Generate S3 URL for the saved attachment
        s3_url = f"s3://gmail-attachments-2709/{filename}"
        return s3_url
        
    except Exception as e:
        logger.error(f"Error downloading attachment: {str(e)}")
        return None
    

# async def process_gmail_message(message_id, user_id, data):
#         try:
#             email_data = get_email_content(data["emailAddress"], message_id)
#             tg_o = extract_email(email_data)
            
#             final_att = []
#             for att in tg_o["attachments"]:
#                 s3_path = await get_email_attachment(message_id, tg_o["attachments"][att], data["emailAddress"], att)
#                 final_att.append(s3_path)
                
#             tg_o["attachments"] = final_att
#             user_data = db_client.get_item(
#                 TableName="users",
#                 Key={"clerk_id": {"S": user_id}}
#             )

#             workflows = user_data.get("Item", {}).get("workflows", {}).get("M", {})

#             active_gmail_workflows = [
#                 {
#                 "id": wid,
#                 "workflow": json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}"))
#                 }
#                 for wid, workflow in workflows.items()
#                 if json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("active",False) and
#                 json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("trigger", {}).get("name") == "TRIGGER_NEW_GMAIL_MESSAGE"
#             ]

#             for workflow in active_gmail_workflows:
#                 syn.delay(workflow["id"], workflow["workflow"]["workflow"], user_id, tg_o)
            
#         except Exception as e:
#             logger.error(f"Error processing Gmail message: {str(e)}")

@app.post("/gmail-webhook",tags=["trigger"])
async def gmail_webhook(request: Request):
    data = await request.json()
    # print(data)
    # Extract message details
    # if "message" in data:
    #     message_id = data["message"]["data"]  # Base64 encoded Pub/Sub message
    #     print(f"New email notification received: {message_id}")
    

    message_id ,user_id= get_latest_email_id(data["emailAddress"])
    # Create EMAILS directory if it doesn't exist
    emails_dir = os.path.join(os.path.dirname(__file__), 'EMAILS')
    os.makedirs(emails_dir, exist_ok=True)
    
    # Define path to user's email history file
    user_email_file = os.path.join(emails_dir, f"{user_id}.json")
    
    # Check if this email was already processed
    recent_emails = []
    if os.path.exists(user_email_file):
        try:
            with open(user_email_file, 'r') as f:
                recent_emails = json.load(f)
        except json.JSONDecodeError:
            recent_emails = []
    
    # Skip if message_id already in recent emails
    if message_id in recent_emails:
        return {"status": "success"}
    
    # Add current message_id and maintain only the 5 most recent
    recent_emails.insert(0, message_id)
    if len(recent_emails) > 5:
        recent_emails = recent_emails[:5]
    
    # Save updated list back to file
    with open(user_email_file, 'w') as f:
        json.dump(recent_emails, f)
    print("user_id",user_id)

    if message_id:
         # Update user's WhatsApp message count
        try:
            # Increment mail_received_count if it exists, otherwise create it with value 1
            response = db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET mail_received_count = if_not_exists(mail_received_count, :zero) + :one",
            ExpressionAttributeValues={
                ':zero': {'N': '0'},
                ':one': {'N': '1'}
            },
            ReturnValues="UPDATED_NEW"
            )
            
            # Extract the updated count value
            updated_count = int(response.get('Attributes', {}).get('mail_received_count', {}).get('N', '0'))
            logger.info(f"Updated mail received count for user {user_id} to {updated_count}")
        except Exception as e:
            logger.error(f"Error updating mail received count: {str(e)}")

        user_data = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )
        
        
        # Check the user's plan
        user_plan = user_data.get('Item', {}).get('plan', {}).get('S', 'free')
        logger.info(f"User {user_id} with plan {user_plan} received an email")
        
        # For free plan users, check if they've exceeded their daily limit
        if user_plan == 'free' or user_plan == 'trial_ended':
            
            # If exceeded limit (10 for free plan), skip processing
            if updated_count > 10:
                logger.info(f"User {user_id} exceeded daily email processing limit (10)")
                creds,user_id= get_fresh_google_credentials(data["emailAddress"])
                service = build('gmail', 'v1', credentials=creds)
                stop_gmail_watch(service)
                return {"status": "success"}
        
        elif user_plan == 'pro' or user_plan=="free_15_pro" or user_plan=="pro++":

            if updated_count > 1000:
                logger.info(f"User {user_id} exceeded daily email processing limit (1000)")
                creds,user_id= get_fresh_google_credentials(data["emailAddress"])
                service = build('gmail', 'v1', credentials=creds)
                stop_gmail_watch(service)
                return {"status": "success"}

        email_data = get_email_content(data["emailAddress"], message_id)
        # print(email_data)
        tg_o=extract_email(email_data)
        # Fetch user data from DynamoDB
        final_att=[]
        for att in tg_o["attachments"]:
            s3_path=get_email_attachment(message_id,tg_o["attachments"][att],data["emailAddress"],att)
            final_att.append(s3_path)
            

        tg_o["attachments"]=final_att
        

        # Extract workflows
        workflows = user_data.get("Item", {}).get("workflows", {}).get("M", {})

        # Filter active workflows with Gmail trigger
        # Debugging: Print the structure of workflows
        # print("Workflows structure:", workflows)

        active_gmail_workflows = [
            {
                "id": wid,
                "workflow": json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}"))
            }
            for wid, workflow in workflows.items()
            if json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("active",False) and
               json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("trigger", {}).get("name") == "TRIGGER_NEW_GMAIL_MESSAGE"
        ]

        # Debugging: Print the filtered workflows
        # print("Active Gmail Workflows:", active_gmail_workflows)

        # # Log or process the filtered workflows
        # print("Active Gmail Workflows:", active_gmail_workflows)
        for workflow in active_gmail_workflows:
            # print(workflow["workflow"]["workflow"])
            # print(type(workflow["workflow"]["workflow"]))
            syn.delay(workflow["id"],workflow["workflow"]["workflow"], user_id, tg_o)
        # syn.delay()
        
      
    return {"status": "success"}



@app.get("/user_auths")
async def user_auths(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    api_keys_response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
    api_keys = api_keys_response.get('Item', {}).get('api_key', {}).get('M', {})
    api_keys = {k: v['S'] for k, v in api_keys.items()}
    if "gemini" in api_keys:
        del api_keys["gemini"]

    # print(api_keys)
    comp={}
    try:
        comp={app.capitalize():check_connection(app,api_keys["composio"]) for app in composio_tools}
    except:
        print("Error checking connections:", "Missing Composio API key")
    
    # Check for WhatsApp verification
    whatsapp_verified = ""
    try:
        user_data = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
        if 'Item' in user_data and 'whatsapp_verified' in user_data['Item']:
            whatsapp_verified = user_data['Item']['whatsapp_verified'].get('S', '')
    except Exception as e:
        logger.error(f"Error fetching WhatsApp verification status: {str(e)}")
    
    comp["WhatsApp"] = True if whatsapp_verified else False
    # extract user auths from database
    return {"user_auths":comp,
            "api_keys":api_keys,
            "WhatsApp": whatsapp_verified 
            }


@app.post("/auth")
async def auth(tool:Tool,credentials: HTTPAuthorizationCredentials = Depends(security)):  # Ensure user is logged in via Clerk
    # take a post req with parameter - tool : gmail/sheets etc

    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    

    # Generate OAuth URL with Clerk user info
    # print(user_id)
    if tool.service=="gmailtrigger":
        state = json.dumps({"user_id": user_id})  # Save Clerk User ID in state
        encoded_state = urllib.parse.quote(state)  # Encode state for URL safety

        auth_url, _ = flow.authorization_url(
            access_type='offline',
            prompt='consent',
            include_granted_scopes='true',
            state=encoded_state  # Pass state in the request
        )
        print(auth_url)

         # save in database  auth_temp = gmail / sheets / etc


    else:
        response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
        api_keys = response.get('Item', {}).get('api_key', {}).get('M', {})
        final_dict = {k: v['S'] for k, v in api_keys.items()}
        auth_url=create_connection_oauth2(tool.service,final_dict["composio"])
        print(auth_url)
        # if check_connection(tool.name,final_dict["composio"]):
            

        

    return {"auth_url": auth_url}

@app.post("/delete_auth")  
async def dele(tool:Tool,credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    # return JSONResponse(content={"status": "error", "message": "cannot delete auth"}, status_code=400)
    
    # if tool.service=="gmailtrigger":

    return {"status": "success"}
    



@app.get("/auth/callback")
async def auth_callback(request: Request):
    # can i send all google wale to here? or only gmail?
    print("url check",request.url)
    flow.fetch_token(authorization_response=str(request.url))
    credentials = flow.credentials

    # Extract state parameter (which contains Clerk user ID)
    state_param = request.query_params.get("state")
    if not state_param:
        raise HTTPException(status_code=400, detail="Missing state parameter")

    state_data = json.loads(unquote(state_param))  # Decode and parse JSON
    user_id = state_data.get("user_id")  # Clerk User ID

    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    user_details = clerk_sdk.users.list(user_id=[user_id])[0]
    mail=user_details.email_addresses[0].email_address
    # Save credentials mapped to Clerk user ID
    # retrieve auth_temp for that user from the database  gmail / sheets / etc
    # for that tool, save the credentials, and make auth_temp=None
    # if auth_temp is already None, then raise an error
    print(credentials.to_json())
    save_google_credentials(user_id,mail, credentials.to_json())
    
    
    success_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Authorization Successful</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            .container { max-width: 500px; margin: auto; padding: 20px; border-radius: 10px; background-color: #f3f3f3; }
            h2 { color: green; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Authorization Successful to SIGMOYD</h2>
            <p>You have successfully authorized your account. You can close this page now.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=success_html)


def delete_gmail_trigger():
    #just remove from user_watch_sessions.
    pass





@app.get("/protected")
async def protected_route( credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
# Now that we have verified the Bearer token and extracted the 
# user ID, we can proceed to access protected resources. Note that # # using the Bearer token is more secure than passing a session ID in # the query parameter.
# We retrieve user details from Clerk directly using the user ID.
    clerk_sdk = Clerk(bearer_auth=clerk_secret_key)
    user_details = clerk_sdk.users.list(user_id=[user_id])[0]
    email = user_details.email_addresses[0].email_address
    ret= {
        "status": "success",
        "data": {
            "first_name": user_details.first_name,
            "last_name": user_details.last_name,
            "email": email,
            "phone": user_details.phone_numbers,
            "session_created_at": user_details.created_at,
            "session_last_active_at": user_details.last_active_at,
        }
    }
    print(ret)
    return ret






    
    
    



async def find_tool_and_execute(new_query: str,chat_history: list,user_id: str):
    print(f"Finding tool for user: {user_id}, query: {new_query}, chat history: {chat_history}")
    # Get user data to fetch API keys
    print("current date and time :", (datetime.now(pytz.timezone('Asia/Kolkata'))).isoformat())

    now_in_india = datetime.now(pytz.timezone('Asia/Kolkata'))
    new_query += f"\nCurrent indian date and time (YYYY-MM-DDTHH:MM:SS) is : {now_in_india.isoformat()}, and the day is: {now_in_india.strftime('%A')}"

    print("New query:", new_query)
    if not new_query:
        new_query=chat_history[-1] if chat_history else ""
        chat_history = chat_history[:-1] if chat_history else []
    def call1():
        
        try:
            to_ret=get_chain(user_id,initial_tools,{"question":"(Chat History from past to present) :- \n"+str(chat_history)+"\n\nNEW QUERY :"+new_query})
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
                time.sleep(retry_seconds + 1)  # Add 1 second buffer
                # Retry the request
                to_ret=get_chain(user_id,initial_tools,{"question":"(Chat History from past to present) :- \n"+str(chat_history)+"\n\nNEW QUERY :"+new_query})
            else:
                raise e
        return to_ret
        
    loop = asyncio.get_event_loop()
    initial_tool_json=await loop.run_in_executor(None, call1)
    if initial_tool_json[0]=="`":
        initial_tool_json=json.loads(initial_tool_json[7:-4])
    else:
        initial_tool_json=json.loads(initial_tool_json)
    print("Initial tool JSON:", initial_tool_json)
    # file_path = os.path.join(os.path.dirname(__file__), 'tools', 'composio.json')
    # with open(file_path) as f:
    #     composio=json.load(f)
    composio = {
    "NOTION": ["NOTION_CREATE_PAGE_IN_PAGE", "NOTION_INSERT_ROW_DATABASE","NOTION_ADD_ONE_CONTENT_BLOCK_IN_PAGE",
    "NOTION_SEARCH_BY_PAGE_OR_ROW_OR_DB_NAME","NOTION_GET_CHILD_BLOCKS_FOR_ROW_OR_PAGE","NOTION_QUERY_DATABASE"],
    "GMAIL": ["GMAIL_SEND_EMAIL", "GMAIL_CREATE_EMAIL_DRAFT", "GMAIL_REPLY_TO_THREAD","GMAIL_FETCH_OLD_EMAILS"],
    "GOOGLESHEETS": ["GOOGLESHEETS_ALL_ACTIONS","CREATE_SHEET_FROM_SCRATCH_AND_ADD_DATA"],
    "GOOGLEMEET": ["GOOGLEMEET_CREATE_MEET"],
    "GOOGLEDOCS": ["GOOGLEDOCS_CREATE_DOCUMENT", "GOOGLEDOCS_UPDATE_EXISTING_DOCUMENT", "GOOGLEDOCS_GET_DOCUMENT_BY_URL","REPORT_MAKER"],
    "GOOGLECALENDAR": ["ALL_ACTIONS"],
    "CSV":["CSV_ANALYSER_AND_REPORT_MAKER"],
    "IMAGE":["IMAGE_ANALYSER"],
    "REPORT_MAKER":["PDF_REPORT_MAKER"],
    "PERSONAL_MEMORY":["SAVE_MEMORY", "FETCH_FROM_MEMORY"],
    "WEB_SEARCH_AGENT": ["WEB_ALL_ACTIONS"],
}


    user_data = db_client.get_item(
                TableName="users",
                Key={"clerk_id": {"S": user_id}}
    )
    api_key = user_data.get("Item", {}).get("api_key", {}).get("M", {})
    api_keys = {k: v['S'] for k, v in api_key.items()}

    comp = api_keys.get("composio", "")
    gemini_key= api_keys.get("gemini", "")

    
    

    print("Initial tool JSON:", initial_tool_json)
    to_embed_composio=[]
    for tool in initial_tool_json:
        if tool.upper() in composio:
            tool_actions=tool.upper()+"(Actions):-\n"
            for use_case in initial_tool_json[tool]:
                cls = globals()[tool.upper()]
                method = getattr(cls, use_case)
                # Extract details from the docstring
                docstring = method.__doc__ if method.__doc__ else "No documentation available."
                
                for action in composio[tool.upper()]:
                    if action.upper() == use_case.upper():
                        # print(docstring)
                        tool_actions += f"{use_case} : \n{docstring.strip()}\n"
                
            to_embed_composio.append(tool_actions)
    to_embed_composio="\n".join(to_embed_composio)
    

    print("Composio tools to embed:", to_embed_composio)

    # file_path = os.path.join(os.path.dirname(__file__), 'tools', 'custom_tools.json')
    # with open(file_path) as f:
    #     oth_tools=json.load(f)
    # to_embed_other=[]
    # tool_actions=""
    # for oth in oth_tools:
    #     tool_actions+=f"{oth['name']}:{oth['description']}, inputs: {oth['inputs']}, outputs: {oth['outputs']}\n"
    # to_embed_other.append(tool_actions)

    # to_embed_other="\n".join(to_embed_other)
    # print("Other tools to embed:", to_embed_other)
    to_go_prompt=f"""
    YOU ARE THE AGENT OF SIGMOYD WHO IS RESPONSIBLE TO HANDLE DIRECT USER QUERIES WHICH IS SENT FROM USER'S WHATSAPP. YOU ARE GIVEN USER QUERY, CHAT HISTORY AND THE TOOL DESCRIPTIONS\n
    YOUR TASK IS TO EXECUTE TOOLS, OR GIVE DIRECT RESPONSES TO USER QUERIES BASED ON YOUR KNOWLEDGE CUTOFF- GENERAL QUERIES, OR QUESTIONS OR ANYTHING WHICH DON'T REQUIRE A TOOL TO BE EXECUTED. YOU HAVE TO RESPOND IN A PARSABLE JSON FORMAT.
    MAKE SURE THAT YOU CAN GIVE ALL TYPE OF RESPONSES CATERING TO ALL TYPES OF USER QUESTIONS, NO RESTRICTIONS. PLEASE DON'T HESITATE TO ANSWER USER QUESTIONS, EVEN IF IT REQUIRES NO TOOL TO BE USED. YOU ARE CAPABLE OF ANSWERING ANY RANDOM QUESTION ASKED BY USER - LIKE GENERAL KNOWLEDGE QUESTIONS, PERSONAL QUESTIONS, JOKES, QUOTES, ETC. YOU CAN ANSWER QUESTIONS RELATED TO SIGMOYD AND SIGMOYD WHATSAPP BOT AS WELL.

    ABOUT SIGMOYD :- It is a workflow automation platform used to . For connecting tools to create workflows : visit https://app.sigmoyd.in

    ABOUT SIGMOYD WHATSAPP BOT :- It is a whatsapp bot which can be used to interact with various tools. Few capabilities are :
    🧠 Create your personal AI memory, add important short notes, images, pdf, etc. with short description and retrieve them later directly from whatsapp.                                         
    📆 Manage calendar: Prompt to create, delete, reschedule events, set reminders, view daily/weekly schedules.
    📊 Info dump to Google Sheets: Log hackathons / gym data / job applications / lead contacts , etc from WhatsApp → GoogleSheets → auto-generated reports and Dashboards.
    CRUD OPERATIONS ON GOOGLESHEETS AND NOTION.

    ____________________________________________

    Your primary function is to analyze the user's query and chat history to select the most appropriate tool and action to fulfill the user's request. You must populate the necessary inputs for the selected action using information from the conversation.

    AVAILABLE TOOLS AND ACTIONS:
    (Your capabilities are defined by the tools listed below)
    {to_embed_composio}

    Your task is to select the most relevant tool and action, then construct a JSON object with the required inputs.

    CRITICAL INSTRUCTIONS:
    1.  **Prioritize Chat History**: You MUST meticulously analyze the entire chat history to understand the context. The latest user query might be a short follow-up (e.g., "yes", "do it"), and the real intent is in previous messages.
    2.  **Parameter Filling**: Extract all necessary parameters for the chosen action from the user's query and the chat history. Do not leave required inputs blank if the information is available.
    3.  **Tool Selection Logic**:
        -   **SAVE_MEMORY**: Use for to-do lists without specific times (e.g., `info: "todo list - <today's date>: all tasks"`), or when the user dumps information to be remembered. If the user says "save this" or "remember this," look for file paths (like S3 links) and descriptions in the chat history to save together.
        -   **FETCH_FROM_MEMORY**: Use for queries involving "show," "retrieve," "fetch," "view," "find," "get," "list," or "search," unless another tool is more specific (e.g., searching emails).
        -   **WEB_SEARCH_AGENT**: Use for requests requiring real-time information, news, or general web searches.
        -   **GOOGLECALENDAR**: Use for scheduling, creating/deleting/rescheduling events, setting reminders, or when a to-do list includes specific times.
        -   **IMAGE_ANALYSER**: Use when the query includes an S3 image path and asks for text extraction or analysis.
        -   **CSV_ANALYSER_AND_REPORT_MAKER**: Use when the query includes an S3 CSV path and asks for analysis or reporting.
        -   **Type 2 (No Tool)**: If no tool matches the user's intent, or if the query is a simple conversational response, select `type: 2` and provide a helpful, direct answer in the `no_tool_found` field.

    4.  **Handling Ambiguity**:
        -   If a new query contains a file path (e.g., S3 link) but no explicit instruction, review the chat history to infer the user's intent.
        -   If the intent is still unclear, DO NOT guess. Instead, use `type: 2` and ask a clarifying question in the `no_tool_found` field, suggesting possible actions (e.g., "I see you've provided an image. Would you like me to extract text, describe it, or save it to your memory?").

    REQUIRED OUTPUT FORMAT (Strictly follow this JSON structure):
    ```json
    {{
      "type": 1,
      "tool_name": "TOOL_NAME",
      "action_name": "ACTION_NAME",
      "input": {{
        "parameter1": "value1",
        "parameter2": "value2"
      }},
      "no_tool_found": null
    }}
    ```
    OR
    ```json
    {{
      "type": 2,
      "tool_name": null,
      "action_name": null,
      "input": {{}},
      "no_tool_found": "A helpful response or a clarifying question to the user."
    }}
    ```
    """
    to_go_prompt += "\n(Chat History from past to present) : "+str(chat_history)+"\nNew Query:"+new_query
    model = genai.GenerativeModel('gemini-2.0-flash')
    got_tool = await generate_con(user_id, model, to_go_prompt)
    print("Got tool response:", got_tool)
    if got_tool[0]=="`":
        got_tool=json.loads(got_tool[7:-4])
    else:
        got_tool=json.loads(got_tool)
    print("Got tool:", got_tool)
    if not got_tool:
        return "No tool found or unable to execute the query."
    

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


    if got_tool.get("type")==1:

        if not comp and (got_tool.get("tool_name","").lower() not in ["web_search_agent","personal_memory","image","csv","report_maker"]):
            return "Composio API key not found. Please set it up in your account settings.\n Visit  : https://app.sigmoyd.in/manage-auths"

        kwargs = {"action": got_tool.get("action_name", "").upper(), **got_tool.get("input", {})}
        print("kwargs", kwargs)
        tool_obj = composio_tools_dict[got_tool.get("tool_name", "").lower()](comp, kwargs,llm,model,user_id)
        print("tool_obj",tool_obj)
        
        response = await tool_obj.execute()


    elif got_tool.get("type")==2:
        response = got_tool.get("no_tool_found", "No tool found for the query. Please try rephrasing your question or using a different tool.")

    if got_tool.get("type")!=2 :
        to_ret={
            "response": response,
            "tool_executed":got_tool
        }
    else:
        to_ret=response

    return to_ret if to_ret else "No response from the tool execution."

async def instant_execution(q: str, user_id: str, session_id: str="",whatsapp=False):
    print(f"Instant execution for user: {user_id}, session: {session_id}, query: {q}")
    if session_id=="":
        session_id=user_id
    #extract memory using session_id
    try:
        # First, get the user data from DynamoDB
        response = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )
        
        # Extract the chat history for this session
        chat_history = []
        session_chats = []
        
        if "Item" in response and "general_chats" in response["Item"]:
            session_chats = response["Item"]["general_chats"].get("M", {}).get(session_id, {}).get("L", [])
            
            for message in session_chats:
                msg_data = message.get("M", {})
                chat_history.append({
                    "msg": msg_data.get("msg", {}).get("S", ""),
                    "content": msg_data.get("content", {}).get("S", ""),
                    # "timestamp": msg_data.get("timestamp", {}).get("S", "time not available")
                })
        print(f"Retrieved chat history : {len(chat_history)} messages ")
        # Get up to 5 most recent messages from the chat history
        # Get up to 5 most recent messages, maintaining original order
        if len(chat_history) > 5:
            chat_history = chat_history[-5:]  # Keep only the 5 most recent messages in the same order they were received

    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        chat_history = []
        session_chats = []

    if q:
        # Store current message
        timestamp = int(datetime.now().timestamp())
        initial_message = {
            "msg": "user",
            "content": q,
            "timestamp": timestamp
        }

        # Add the message to in-memory session chats
        session_chats.append({
            "M": {
                "msg": {"S": initial_message["msg"]},
                "content": {"S": initial_message["content"]},
                "timestamp": {"N": str(initial_message["timestamp"])}
            }
        })

        try:
            # Make sure general_chats attribute exists
            if session_id:
                response = db_client.get_item(
                    TableName="users",
                    Key={"clerk_id": {"S": user_id}}
                )
                
                # If general_chats doesn't exist, create it first
                if "Item" not in response or "general_chats" not in response["Item"]:
                    db_client.update_item(
                        TableName="users",
                        Key={"clerk_id": {"S": user_id}},
                        UpdateExpression="SET general_chats = :empty_map",
                        ExpressionAttributeValues={
                            ":empty_map": {"M": {}}
                        }
                    )
        except Exception as e:
            print(f"Error checking general_chats: {e}")
            logger.error(f"Database error when checking general_chats for user {user_id}: {str(e)}")
       

    active_whatsapp_workflows=[]
    if whatsapp:
        user_data = db_client.get_item(
                TableName="users",
                Key={"clerk_id": {"S": user_id}}
            )

        # Extract workflows
        workflows = user_data.get("Item", {}).get("workflows", {}).get("M", {})

        active_whatsapp_workflows = [
            {
                "id": wid,
                "workflow": json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}"))
            }
            for wid, workflow in workflows.items()
            if json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("active",False) and
                json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("trigger", {}).get("name") == "TRIGGER_SIGMOYD_WHATSAPP" 
        ]


    if active_whatsapp_workflows:
        complete_text=f"""
    Following is the query from user via whatsapp. 

    {chat_history[-1] if chat_history else ""}

    current date and time in india YYYY-MM-DDTHH:MM:SS: {(datetime.now(pytz.timezone('Asia/Kolkata'))).isoformat()}
    """
        print("Executing WhatsApp workflows for user:", complete_text)
        for workflow in active_whatsapp_workflows:
            syn.delay(workflow["id"], workflow["workflow"]["workflow"], user_id, complete_text)

        return "workflows executed successfully, please check logs on sigmoyd dashboard for more details."

    response = await find_tool_and_execute(q, chat_history, user_id)

    timestamp = datetime.now().isoformat()
    ai_message = {
        "msg": "bot",
        "content": response,
        "timestamp": timestamp
    }

    # Add the AI message to the in-memory session chats
    # Convert content to JSON string if it's a dictionary
    if isinstance(ai_message["content"], dict):
        content_str = json.dumps(ai_message["content"])
    else:
        content_str = str(ai_message["content"])
    
    session_chats.append({
        "M": {
            "msg": {"S": ai_message["msg"]},
            "content": {"S": content_str},
            "timestamp": {"S": ai_message["timestamp"]}
        }
    })
    
    # Keep only the last 6 messages
    if len(session_chats) > 6:
        session_chats = session_chats[-6:]
    
    try:
        if session_id:
            # Replace the entire chat history with just the last 6 messages
            db_client.update_item(
                TableName="users",
                Key={"clerk_id": {"S": user_id}},
                UpdateExpression="SET general_chats.#sid = :messages",
                ExpressionAttributeNames={
                    "#sid": session_id
                },
                ExpressionAttributeValues={
                    ":messages": {"L": session_chats}
                }
            )
    except Exception as e:
        print(f"Error updating chat history with AI response: {e}")
        logger.error(f"Database error when updating chat history for user {user_id}: {str(e)}")
    
    return response






@app.post("/general")
async def general_query(q: General, credentials: HTTPAuthorizationCredentials = Depends(security)):

    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    response = await instant_execution(q.query, user_id, q.session_id)
    response_key = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
    api_keys = response_key.get('Item', {}).get('api_key', {}).get('M', {})
    gemini_key = api_keys.get('gemini', {}).get('S') if 'gemini' in api_keys else ""

    # Configure with the appropriate key
    genai.configure(api_key=gemini_key)

    # Format the response to be more readable and concise
    if isinstance(response, dict):
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        formatted_response = await generate_con(user_id,model,f"""
YOU ARE AN AI ASSISTANT "SIGMOYD". YOUR TASK IS GIVING RESPONSES TO USER BASED ON WHAT THE USER ASKED TO PERFORM.
GIVE CONSICE RESPONSE TO THE USER AS IF YOU PERFORMED THE REQUIRED OPERATIONS.

IMPORTANT GUIDELINES TO FOLLOW:-
1. Make the output from the system more understandable according to user's question . Remove unwanted stuff like ids, unnessary logs and give a consice response. please return just what user asked and make it convinient for the user to understand
Strictly delete all ids like calendar id, etc. unless specifically asked by user.

2. Keep all the important information whatever is required by user. Don't remove any important information. But convert it in simple readable language. Example - if the output is in json format, convert it to a simple readable string. Or if the output is in isoformat, convert it to a simple readable format like 5th Jan 2022, 10:30 AM.

3. if the system's output has some error, explain the user the reason for the error and how to mitigate it. If output mentions about some missing variable or api key limits exceeded, then explain the user the reason for the error and how to mitigate it, Example - "Please provide the missing variable"

4. The system's response might have something missing. if it happens during fetch functions, it means that there is no data available for the query. So, please explain the user that there is no data available for the query.

5. Please don't show the tool executed, action , input, until the system outputs some error. return this only to explain the user if system outputs some error.

6. Please talk clearly to the user and don't use any technical jargon. Make it simple and easy to understand. Give reply such that you know everything. don't say that system executed some tool, just give the response to user as if you only executed the tool and got the output.

SYSTEM'S OUTPUT AFTER EXECUTION: {response}

USER'S QUESTION: {q.query}

According to the output and user's query, find out if there is any next operation to be performed, if yes then at the end ask the user 'should i perform that next task?'. Please don't return the full user's question again.
If there is no next operation, then please don't ask/mention about next operation.
Apart from this, no preambles or postambles needed. Give response such that you are an AI Assistant directly talking to the user in second person. Act as you are the system and you only have executed the tools""")
        # Send the formatted response
        send_response = formatted_response
    else:
        # If not a dict, send as is
        send_response = response
    return {"response": send_response}



@app.post("/refine_query")
async def refine_query(q: Question,credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    # user_id= "test_user"  # For testing purposes, replace with actual user_id extraction logic
    # comp="fyvn2yln306o052h5mt007"  # For testing purposes, replace with actual composio key extraction logic

    if q.flag==2:
        # print("Refining query for user:", user_id,type(q.flag))
        def call3():
            return get_chain(user_id,router,{"question":q.query})
        loop = asyncio.get_event_loop()
        try:
            mode=await loop.run_in_executor(None, call3)
        except Exception as e:
            print(f"Error determining mode.maybe gemini key not filled: {e}")
            raise HTTPException(status_code=500, detail="Please check your gemini key in settings or try again later.")
        print("Mode determined:", mode)
        if int(mode)==0:
            def call4():
                return get_chain(user_id,ques_flow,{"question":"\nQUERY:-\n"+q.query},"gemini-2.5-flash")
            loop = asyncio.get_event_loop()
            ques=await loop.run_in_executor(None, call4)
            # print(ques)
            if ques[0]=="`":
                try:
                    ques=json.loads(ques[10:-4])
                except:
                    ques=json.loads(ques[10:-3])
            else:
                ques=json.loads(ques)
            # print(ques)
            return {"response":ques,"mode" : "workflow"}
        elif int(mode)==1:  
            print("Instant execution mode detected")
            
            # Handle instant execution mode
            session_id = str(uuid.uuid4())
            # Store session_id in the database or session store if needed
            response = db_client.get_item(
                TableName="users",
                Key={"clerk_id": {"S": user_id}}
            )

            # Check if general_chats attribute exists, if not create it
            try:
                # Check if general_chats attribute exists, if not create it
                if "general_chats" not in response.get("Item", {}):
                    db_client.update_item(
                        TableName="users",
                        Key={"clerk_id": {"S": user_id}},
                        UpdateExpression="SET general_chats = :empty_map",
                        ExpressionAttributeValues={
                            ":empty_map": {"M": {}}  # Initialize as empty map
                        }
                    )

                
            except Exception as e:
                print(f"Error initializing chat session: {e}")
                logger.error(f"Database error when initializing chat session for user {user_id}: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to initialize chat session")

            ###########################################################
            print(f"Starting instant execution for user: {user_id}, session: {session_id}, query: {q.query}")
            response = await instant_execution(q.query, user_id, session_id)
            response_key = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
            api_keys = response_key.get('Item', {}).get('api_key', {}).get('M', {})
            gemini_key = api_keys.get('gemini', {}).get('S') if 'gemini' in api_keys else ""

            # Configure with the appropriate key
            genai.configure(api_key=gemini_key)

            # Format the response to be more readable and concise
            if isinstance(response, dict):
                model = genai.GenerativeModel('gemini-2.0-flash-lite')
                formatted_response = await generate_con(user_id,model,
                    f"Make this output more understandable according to user's question . If the content is very long, make it concise in under 500 words.  please just give what user asked in a readable string, and remove unwanted information. if there is some error, explain the user the reason for the error and how to mitigate it, example (any missing variable or api key limits exceeded) .\n OUTPUT: {response}\n##########\n USER'S QUESTION: {q.query}\n#########\n According to the output and user's query, find out if there is any next operation to be performed, if yes then at the end ask the user 'should i perform that next task?'")
                # Send the formatted response
                send_response = formatted_response
            else:
                # If not a dict, send as is
                send_response = response

            # print ({
            # "response": response,
            # "mode": "general",
            # "session_id": session_id
            # })

            return {
            "response": send_response,
            "mode": "general",
            "session_id": session_id
            }

    
    elif q.flag==1:
        file_path = os.path.join(os.path.dirname(__file__), 'tools', 'composio.json')
        with open(file_path) as f:
            composio=json.load(f)
        # print(q.question)
        print(composio)
        def call5():
            return get_chain(user_id,workflow_initial_tools,{"question":q.query+"\n\nHere are the user needs (questions answered by user) :- \n"+str(q.question)})
        loop = asyncio.get_event_loop()
        initial_tool_json=await loop.run_in_executor(None, call5)
        if initial_tool_json[0]=="`":
            initial_tool_json=json.loads(initial_tool_json[7:-4])
        else:
            initial_tool_json=json.loads(initial_tool_json)

        print("Initial tool JSON:", initial_tool_json)
        to_embed_composio=[]
        for tool in initial_tool_json:
            if tool.upper() in composio:
                tool_actions=tool.upper()+"(Actions):-\n"
                for use_case in initial_tool_json[tool]:
                    cls = globals()[tool.upper()]
                    method = getattr(cls, use_case)
                    # Extract details from the docstring
                    docstring = method.__doc__ if method.__doc__ else "No documentation available."
                    
                    for action in composio[tool.upper()]:
                        if action.upper() == use_case.upper():
                            # print(docstring)
                            tool_actions += f"{use_case} : \n{docstring.strip()}\n"
                    
                to_embed_composio.append(tool_actions)
        to_embed_composio="\n".join(to_embed_composio)
        

        print("Composio tools to embed:", to_embed_composio)
        def call6():
            return get_chain(user_id,gemini_prompt,{"prompt":f"Act as a query enhancer specialist . Refine the old query and generate a detailed new query based on the following questions and answers (user needs). "+"\nOLD QUERY:-\n"+q.query+f"\n More specific user needs (questions answered by user):- \n{q.question}"+"\n.Add all necessary details from user's answers to the new query ,and return a new query which will be further used to create agentic workflow. \nGive only the refined query, don't skip anything. keep the details in refined query in a proper order so that user can get an optimized and relevant workflow, according to exactly what he wants. (No preambles and postambles)\
        SOME CONTEXT:- You are problem understanding agent, whose task is to transform the user's vague query to a well defined query, so that, it can be used to generate multi agent, multi tool workflows including agents like AI (LARGE LANGUAGE MODEL), VALIDATOR (This decides the path to follow A/B/C/D...(out of multiple path options), validator's output affects the execution of next tools), ITERATOR (used to one by one pass each element of list of inputs to next agents for execution),TRIGGER (which starts the workflow), and many differnt tools (functions which takes some inputs and returns some outputs)\
        you need to understand what user actually wants and which agents to use, in what order, and using this understanding, generate a well defined query having proper explanation in ordered bullet points (not only agent names) (For each agent, keep Short explainations, and don't copy above questions and answers, instead give short explaination of what needs to be done).\
    \
        Some points to follow: - if any tool's input parameters has to be decided by some condition, so rather than using a validator, strictly use a TEXT INPUT TOOL (even if condition is already given) where user will again define all conditions, and then strictly use LLM ( which will decide next tool input based on the condition and previous tool output) Eg - sending mail to different people based on some condition (here tool is send mail, and inputs can vary).\
        - must use validator if user wants to execute a different set of tools (chose a path A/B/C/D) based on different conditions. And give seperate agents for each set. \
        - Validator outputs a path A/B/C/D... instead of boolean. So it's prompt shoud be like - 'if something happens, then go to path A, else go to path B'. Also no need to specify that validator uses LLM. (it's understood) (make sure to use single letter A/B/C/D only for paths. no other names allowed )\
        - ITER_END : inputs : [merged_data] output : [merged_data] (use this after the last agent which will come under iteration. A ITER_END must be used whenever ITERATOR is used. If user asks to collect some information from each iteration and merge them  (optional), the information will get populated and merged into merged_data, which will be passed to the next agent after the last iteration)\
            and for populating the information, we'll need an extra agent under each iteration, which will send exact information to ITER_END (use this extra agent only if asked by user to get collective information/ merged information from each iteration)\
            In iteration , [ITERATOR -> ALL TOOLS UNDER ITERATION -> ITER_END] , make sure to keep ITER_END after all tools under iteration\
        - Just after gmail trigger / whatsapp trigger, STRICTLY use a validator for deciding the further path if only for some of the cases, workflow should be executed (eg : execute the workflow for only some kind of emails after email trigger)\
        - AI : generalized multipurpose llm for general text generation. input : llm_prompt. output : generated content (all required outputs in json format)\
            please keep less usage of AI. Use only when needed. AI cannot analyse csv files directly from the s3 path\
                                            \
            (TOOLS WITH ACTIONS):-\
        " + to_embed_composio + """

        OTHER TOOLS (TOOLS WITHOUT ACTIONS):-
        TEXT_INPUT (Collect a text input from the user at the beginning of the workflow, prior to its execution. This input is intended solely for gathering additional context, conditions or information from the user to help guide the workflow logic. Do not request or collect any user credentials, sensitive data, or tool-specific function parameters),
        FILE_UPLOAD (take file upload from user and returns the S3 File paths. If any tool requires S3 path. MUST USE FILE UPLOAD BEFORE THAT TOOL if that tool needs to operate on any file which user have)
        PDF_TO_TEXT (extract text from pdf, STRICTLY MUST BE USED WITH FILE_UPLOAD tool , if user mentions pdf file upload (or files which might be pdf - like resume, report, etc),so that the text inside the file can be extracted and passed to next tool/llm if required),
        
NOTE :- TEXT_INPUT and FILE_UPLOAD are tools which dont execute during runtime. thy just pass the data to the next tool.
Initially before runtime, TEXT_INPUT tool will collect and save user's input. FILE_UPLOAD tool will upload the file to s3 and save the s3 path.
Now, during runtime, these tools just pass the saved data to the next tool/llm, so that it can be used for further processing.



            TOOL USAGE TIPS:- (IMPORTANT)
            - IMPORTANT : Strictly don't use TEXT_INPUT to ask the s3 path from the user, instead use FILE_UPLOAD tool to upload the file and get the s3 path.
            - Must use FILE_UPLOAD tool if user sounds like he needs to do something with a file he have. 
            - CSV_READER_FOR_ITERATOR, CSV_MODIFY, CSV_ANALYSER_AND_REPORT_MAKER tools need s3 paths, hence use them after FILE_UPLOAD tool or attachment s3 paths from gmail trigger / whatsapp trigger.
            - If user asks to create report from CSV file, then use CSV_ANALYSER_AND_REPORT_MAKER tool, which will return the answer of asked query and the S3 path of the pdf analysis report. 
            - Use report maker tool when user asks to create a report from the text data. if user dont specify the format of input data  (then use both CSV_ANALYSER_AND_REPORT_MAKER and report maker tool)
            - GMAIL TRIGGER returns the sender's email address, body, subject, list of attachments (s3 paths). so please don't use gmail_fetch_old_emails tool again after gmail trigger. Also don't use file upload if gmail trigger output attachments are needed to be processed instead of any file user already have with him, 
            - WHATSAPP TRIGGER returns the message sent by user, list of attachments (s3 paths) of files sent by user via whatsapp. Please don't use file upload if whatsapp trigger output attachments are needed to be processed instead of any file user already have with him,
            - if user wants to perform some operation on each row of the filtered csv file, then STRICTLY use following sequence : CSV_MODIFY (returns filtered/modified csv path) -> CSV_READER_FOR_ITERATOR (returns each row in list format) -> ITERATOR (iterate each row of csv) -> ...(further operations)
            - STRICTLY USE ITERATOR IMMEDIATELY AFTER SOME FUNCTION WHICH RETURNS A LIST IN WHICH EACH ELEMENT HAVE TO BE SENT ONE BY ONE FOR FURTHER PROCESSING. Example : - LINKEDIN_GET_RECRUITER_EMAIL
            - please keep less usage of AI. Use only when needed. Never use AI for CSV file analysis or web searches
            - don't give extra irrelevant tools which are not asked (Eg : email parser, error handling, etc.)


        STRICTLY GIVE FINAL OUTPUT IN SIMPLE TEXT (bullet points) each point have SHORT EXPLAINATIONS (no json please!)
        STRICTLY DON'T KEEP THE ENHANCED QUERY VERY LONG. ALSO KEEP DON'T KEEP UNNECESSARY AGENTS OR TOOLS. TRY TO MAKE THE WORKFLOW PLAN CONCISE, SIMPLE WITH MINIMUM NUMBER OF AGENTS AND TOOLS.
        
            """},"gemini-2.5-flash")
        loop = asyncio.get_event_loop()
        refined_query = await loop.run_in_executor(None, call6)
        query=refined_query
        return {"response":query}


@app.post("/create_agents")
async def create_agents(query : Query, credentials: HTTPAuthorizationCredentials = Depends(security)):   
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")


    user_data = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )
        
        
        # Check the user's plan
    user_plan = user_data.get('Item', {}).get('plan', {}).get('S', 'free')
    logger.info(f"User {user_id} with plan {user_plan} initiated workflow creation.")

    if user_plan == "free" or user_plan == "trial_ended":
        logger.warning(f"User {user_id} with plan {user_plan} attempted to create workflows.")
        raise HTTPException(status_code=403, detail="Your plan does not allow creating workflows. Please upgrade to pro plan")
    
    try:
        # Increment workflow_count if it exists, otherwise create it with value 1
        count = db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET workflow_created_daily_count = if_not_exists(workflow_created_daily_count, :zero) + :one",
            ExpressionAttributeValues={
                ':zero': {'N': '0'},
                ':one': {'N': '1'}
            },
            ReturnValues="UPDATED_NEW"
        )
        logger.info(f"Updated workflow created daily count for user {user_id}")
        count_value = count.get('Attributes', {}).get('workflow_created_daily_count', {}).get('N', '0')
        if (user_plan=="pro" or user_plan=="pro++") and int(count_value) > 10:
            logger.warning(f"User {user_id} has reached the daily workflow creation limit.")
            raise HTTPException(status_code=403, detail="You have reached the daily limit of 10 workflows. Please try again tomorrow.")
        elif user_plan=="free_15_pro" and int(count_value) > 2:
            logger.warning(f"User {user_id} has reached the daily workflow creation limit.")
            raise HTTPException(status_code=403, detail="You have reached the daily limit of 10 workflows. Please try again tomorrow.")
        
    except Exception as e:
        logger.error(f"Error updating workflow created daily count: {str(e)}")


    file_path = os.path.join(os.path.dirname(__file__), 'tools', 'custom_tools.json')
    with open(file_path) as f:
        custom=json.load(f)
    
    to_embed=[f"{i['name']} : {i['description']}" for i in custom]
    to_embed="\n".join(to_embed)

    # dynamically add all custom tools to major_tool_chain from custom_tools.json
    def call7():
        return get_chain(user_id,major_tools,{"question":query,"customs":to_embed},"gemini-2.5-flash")
    loop = asyncio.get_event_loop()
    major_tool_list = await loop.run_in_executor(None, call7)
    # print(query.query)
    
    if major_tool_list[0]=="`":
        major_tool_list=json.loads(major_tool_list[7:-4])
    else:
        major_tool_list=json.loads(major_tool_list)
    print("MAJOR TOOL LIST :-\n",major_tool_list)
    ret_un=None
    if major_tool_list.get("UNAVAILABLE"):
        ret_un=major_tool_list["UNAVAILABLE"]
    file_path = os.path.join(os.path.dirname(__file__), 'tools', 'composio.json')
    with open(file_path) as f:
        composio=json.load(f)
    file_path = os.path.join(os.path.dirname(__file__), 'tools', 'custom_tools.json')
    with open(file_path) as f:
        oth_tools=json.load(f)
    file_path = os.path.join(os.path.dirname(__file__), 'tools', 'user_made_custom.json')
    with open(file_path) as f:
        user=json.load(f)
    oth_tools+=user
    to_embed_composio=[]
    to_embed_other=[]
    for tool in major_tool_list["WITH_ACTION"]:
        if tool.upper() in composio:
            tool_actions=tool.upper()+"(Actions):-\n"
            for use_case in major_tool_list["WITH_ACTION"][tool]:
                cls = globals()[tool.upper()]
                method = getattr(cls, use_case)
                # Extract details from the docstring
                docstring = method.__doc__ if method.__doc__ else "No documentation available."
                
                for action in composio[tool.upper()]:
                    if action.upper() == use_case.upper():
                        # print(docstring)
                        tool_actions += f"{use_case} : \n{docstring.strip()}\n"
                
            to_embed_composio.append(tool_actions)
        
        # elif tool.upper()=="OTHERS":
    for oth_maj in major_tool_list["OTHERS"]+list(major_tool_list["WITH_ACTION"].keys()):
        tool_actions=""
        for oth in oth_tools:
            if oth["name"].upper()==oth_maj.upper():
                tool_actions+=f"{oth['name']}:{oth['description']}, inputs: {oth['inputs']}, outputs: {oth['outputs']}\n"
        to_embed_other.append(tool_actions)

            
    to_embed_composio="\n".join(to_embed_composio)
    to_embed_other="\n".join(to_embed_other)

    # chat_completion = groq.chat.completions.create(                          
        
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": trigger_finder                           

    #             },
    #             {
    #                 "role": "user",
    #                 "content": query.query,
    #             },
    #         ],
    #         model="llama-3.3-70b-versatile",                       
    #         temperature=0,    
    #         # Streaming is not supported in JSON mode
    #         stream=False,
    #         # Enable JSON mode by setting the response format
    #     )
    def call8():

        return get_chain(user_id,trigger_flow,{"question":query.query})
    loop = asyncio.get_event_loop()
    trigger = await loop.run_in_executor(None, call8)
    # print(query.query)

    if trigger[0]=="`":
        trigger=json.loads(trigger[7:-4])
    else:
        trigger=json.loads(trigger)
    # print(trigger)
    # print("Triggers:-",trigger["name"], trigger["description"], "outputs:", trigger["output"], "\n")  # Updated to print trigger details
    # if chat_completion.choices[0].message.content[0]=="`":
    #     try:
    #         trigger=json.loads(chat_completion.choices[0].message.content[7:-4])
    #     except:
    #         trigger={"name":"unavailable","description":"unavailable","output":"unavailable"}
    # else:
    #     try:
    #         trigger=json.loads(chat_completion.choices[0].message.content)
    #     except:
    #         trigger={"name":"unavailable","description":"unavailable","output":"unavailable"}
    

    trigger["id"]=0
    # Return the response as JSON

    # print("to_embed_composio",to_embed_composio)
    # print("to_embed_other",to_embed_other)


#     deligates="""
#  "deligation_prompt" (only for deligator) : "detailed deligation criteria", (just keep  deligation criteria, not the output format)
#     "available_deligation" (only for deligator):["list of agent ids to which connector can deligate tasks"]

# Deligation Rules:
# Can delegate tasks to previous relevant agents only
# use delegation only if required, example :- if there are 2 agents - codeer_llm, log_summarizer, then a deligator after that. deligator check summarized_logs and delegate back to coder_llm agent if summarized_logs has some errors. 
# try keep no. of delegations to minimum, and only if required.
# dont keep deligator if not asked by user.


# 3. DELIGATOR. inputs :[deligation_prompt , output of previous agents ]. outputs : {{to_deligate_from_p (bool) : True/False , deligation_p : {{agent_id : id of agent to which task is to be deligated, changes_required : "detailed explaination of what changes are required"}}}} 
# (use deligator to execute a set of agents repeatedly until a condition is met.)
    
# *note -> don't mix delegation and validation. both serve different purposes. use delegation only if required , and use validation for checking conditions.
#     """


    tool_finder=f"""
You are an expert workflow architect specializing in AI agent orchestration. When given a user query, design an ordered list of specialized agents with rectifier checkpoints.  using this structure:


{{
"workflow_name": (suitable name with spaces between words),
"workflow" (list of agents): [
{{  "id": int (1/2/3...),
    "type" : str (tool/llm/connector),
    "name" : str (name of tool/llm/connector),  
    "tool_action" (only for tool agents. keep empty string if no action available for that tool): str (action for that tool) ,
    "description" : str (how the tool/llm/connector will be used to execute the subtask. don't keep it very specific, because specific information might get changed by user later. just keep the main idea of how this tool will be used to solve the problem. example - "analyse/modify the csv file" is main idea, but "remove the first column from csv file" is specific information which might change later.) ,
    "to_execute" (List with 2 strings)/None: ['validator_p','A'] if this agent needs to be executed if validator_p is 'A', ['validator_p','B'] if agent needs to be executed if validator_p is 'B'.  (here p is the last just passed validator id. A/B/C... are the different paths decided by validator), None if any validator don't affect this agent.
    "config_inputs" : (dict of inputs required by the agent from the user before workflow execution. example :- {{"link_of_notion_page":"put here if available in user query, else keep empty"}}  (these are pre decided inputs, and don't change for different workflow executions)(use config inputs only if any information needed by this tool cannot be found from previous tool outputs)
    "data_flow_inputs" : ["list of data_flow_notebook_keys this agent reads from"],
    "llm_prompt" (only for llm agent): "descriptive prompt containing Core domain expertise , Problem-solving approach , Response logic", (just keep prompt, output format is simple string, so no need to keep output format in prompt)
    "data_flow_outputs": ["list of data_flow_notebook_keys this agent writes to and reads from"],
    "validation_prompt" (only for validator) : "detailed Validation criteria with further path information. eg: if something happens, chose A, if other thing happens, chose B ", (Dont keep detailed output format, just keep the validation criteria and path information)
}},
(next agents)
],

"data_flow_notebook_keys" : [keep all relevant keys for proper data flow. example:- trigger_output, validator_3, iterator_4, meeting_link_3, etc. ] (keep unique keys only. if id of agent is 3, then all data_flow_output names of this agent will end with _3. example:-  meeting_link_3, etc. )

}}

Notebook structure:
Shared dictionary with entries fom multiple agents
Each agent must explicitly declare read/write fields
Preserve chain-of-thought outputs


Trigger to starrt the workflow :-
{trigger["name"]} : {trigger["description"]} . outputs : {trigger["output"]}
every trigger output will be saved to the data_flow_notebook["trigger_output"] if available. if output from trigger not available, then first agent will directly start the workflow

AVAILABLE TOOL NAMES AND THEIR ACTIONS:- 
{to_embed_composio}

TOOLS WITHOUT ACTIONS:-
{to_embed_other}

IMPORTANT : STRICTLY USE ABOVE TOOLS AND ACTIONS ONLY, NO MATTER THE USER'S PROBLEM IS SOLVED OR NOT! DON'T CREATE ANY NON EXISTING TOOL OR ACTION! KEEP ONLY THOSE TOOLS/ACTIONS WHICH WILL EXACTLY SOLVE USER'S PROBLEM.
IMPORTANT : PLEASE KEEP ONLY THOSE INPUT PARAMETERS OF ABOVE FUNCTIONS IN CONFIG INPUTS , WHOSE VALUES CANNOT BE FOUND FROM PREVIOUS AGENT OUTPUTS (data_flow_notebook). 


available connectors:-
1. VALIDATOR (conditional connector). inputs : [validation_prompt , output from previous agent]. outputs : {{validator_p (p is the id, value will be A/B/C... according to required path)}}
2. ITERATOR : inputs: [list_of_something]. output : [list_element (one at a time)] (use this when need to pass something to next tools one by one from a list. use iterator just after the tool which returns a list of elements)
3. ITER_END : inputs : [merged_data] output : [merged_data] (use this after the last agent which will come under iteration. A ITER_END must be used whenever ITERATOR is used. If user asks to collect some information from each iteration, the information will get populated into merged_data, which will be passed to the next agent after the last iteration)

*note -> never use if-else to check if any element available in iterator. iterator handles full process of sending each element to next agents.
*note -> there must be some agent under iteration which should send required information from each iteration to the ITER_END agent

available llms:-
1. CODER : code generation. input : llm_prompt. output : {{"code":generated_code, "language":language of code}}.
2. AI : generalized multipurpose llm for general text generation. input : llm_prompt. output : generated content (all required outputs in json format)


SOME RULES TO BE FOLLOWED STRICTLY :-
0. MUST SPECIFY THE PATH TO TAKE IN VALIDATION PROMPT, LIKE - "if something happens, then go to path A, else go to path B". MUST KEEP VALIDATOR PROMPT IN THE OUTPUT, DON'T MISS IT. Keep path names as single letter A/B/C/D... only (no other names allowed like PATH_A1, PATH_A, etc)
1. I already know the trigger, so don't give trigger, trigger is not a tool, so find the tools which will be used strictly after the trigger. example if using gmail trigger ,then no need to use gmail_search_mail.
2. IF TRIGGER RETURNS ANY OUTPUT, THEN THE FIRST AGENT MUST BE VALIDATOR, IT SHOULD TAKE INPUT FROM THE TRIGGER'S OUTPUT (data_flow_notebook["trigger_output"]) and decide if the workflow should be executed or not
3. REMEMBER, MANUAL_TRIGGER HAS NO OUTPUTS. IT IS JUST USED TO START THE WORKFLOW WITH EMPTY data_flow_notebook.
4. Strictly make sure that config inputs will be included if that input cannot be found out from previous tool outputs !! 
5. Must use iterator if you need to pass something to next tools one by one from previous output.
6. return a output in json format which will be used to execute the workflow, given the user query. No preambles or postambles are required. Keep all strings in double quotes.
"""
# embed different things in tool_finder prompt
    tool_finder+="\n"+f"WORKFLOW TO CREATE :- {query.query}"
    print("prompt:-\n",tool_finder,"\n")
    # response = groq.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": str(tool_finder)                    #  USE DEEPSEEK HERE

    #         },
    #         {
    #             "role": "user",
    #             "content": query.query,
    #         },
    #     ],
    #     model="llama-3.3-70b-versatile",
    #     temperature=1,    
    #     # Streaming is not supported in JSON mode
    #     stream=False,
    #     # Enable JSON mode by setting the response format
    # )

    # print(chat_completion)
    
    def call_openai_sync():

        return client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": tool_finder},
                {"role": "user", "content": f"WORKFLOW TO CREATE :- {query.query}"},
            ],
            stream=False,
            temperature=0
        )
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, call_openai_sync)

    except Exception as e:
        print(f"Error calling Deepseek API: {e}")
        raise HTTPException(status_code=500, detail="Failed to call Deepseek API")

    
    

    


    # tools=response

    tools=json.loads(response.choices[0].message.content[7:-3])
    print("Tools found:", tools)


    for tool in tools["workflow"]:
        if tool["type"] == "tool":
            if tool["name"].upper() in composio_tools:
                class_name = globals().get(tool["name"].upper())
                if class_name:
                    method_name = tool["tool_action"]
                    if hasattr(class_name, method_name):
                        method = getattr(class_name, method_name)
                        signature = inspect.signature(method)
                        required_params = set(signature.parameters.keys()) - {"self"}
                        config_inputs_keys = set(tool.get("config_inputs", {}).keys())
                        data_flow_inputs_keys = set(tool.get("data_flow_inputs", []))
                        missing_params = required_params - (config_inputs_keys | data_flow_inputs_keys)

                        # # Prepare the sets for LLM
                        # llm_prompt = f"""
                        # You are an expert in understanding input parameters of a function : {method_name} Given the following sets:
                        # 1. Required Parameters: {list(required_params)}
                        # 2. Config Inputs Keys: {list(config_inputs_keys)}
                        # 3. Data Flow Inputs Keys: {list(data_flow_inputs_keys)}

                        # Identify the missing parameters from the Required Parameters set that are not present in the other two sets. 
                        # If a parameter is missing but might be available within the parameters of other sets (means any of the parameters in other sets may have a similar name, or similar context), then do not include thet pseudo missing parameter in the missing list. 
                        # Return only the list of truly missing parameter names in python list format . keep all elements in double quotes:
                        # ["p1","p2"]   
                        # no preambles or postambles.
                        # """

                        # # Query the Gemini LLM
                        # missing_params =  get_chain(user_id,gemini_prompt,"gemini-1.5-flash").invoke({
                        #     "prompt": llm_prompt
                        # })
                        print("Missing parameters identified by LLM:", missing_params)

                        # # Parse the response
                        # if missing_params[0] == "`":
                        #     try:
                        #         missing_params = json.loads(missing_params[10:-4])
                        #     except:
                        #         missing_params = json.loads(missing_params[10:-3])
                        # else:

                        #     missing_params = json.loads(missing_params)

                        # Update the tool's config_inputs dictionary
                        for param in list(missing_params):
                            tool.setdefault("config_inputs", {})[param] = ""

            else:
                for oth_tool in oth_tools:
                    if oth_tool["name"].upper() == tool["name"].upper():
                        required_params = set(oth_tool.get("inputs", []))
                        config_inputs_keys = set(tool.get("config_inputs", {}).keys())
                        data_flow_inputs_keys = set(tool.get("data_flow_inputs", []))
                        missing_params = required_params - (config_inputs_keys | data_flow_inputs_keys)
                        # # Prepare the sets for LLM
                        # llm_prompt = f"""
                        # You are an expert in understanding input parameters of a function : {tool['name'].upper()} Given the following sets:
                        # 1. Required Parameters: {list(required_params)}
                        # 2. Config Inputs Keys: {list(config_inputs_keys)}
                        # 3. Data Flow Inputs Keys: {list(data_flow_inputs_keys)}

                        # Identify the missing parameters from the Required Parameters set that are not present in the other two sets. 
                        # If a parameter is missing but might be available within the parameters of other sets (means any of the parameters in other sets may have a similar name, or similar context), then do not include thet pseudo missing parameter in the missing list. 
                        # Return only the list of truly missing parameter names in python list format . keep all elements in double quotes:
                        # ["p1","p2"]   
                        # no preambles or postambles.
                        # """

                        # # Query the Gemini LLM
                        # missing_params = get_chain(user_id,gemini_prompt,"gemini-1.5-flash").invoke({
                        #     "prompt": llm_prompt
                        # })
                        print("Missing params for tool", tool["name"], ":", missing_params)

                        # # Parse the response
                        # if missing_params[0] == "`":
                        #     try:
                        #         missing_params = json.loads(missing_params[10:-4])
                        #     except:
                        #         missing_params =json.loads(missing_params[10:-3])
                        # else:

                        #     missing_params = json.loads(missing_params)

                        # Update the tool's config_inputs dictionary
                        for param in list(missing_params):
                            tool.setdefault("config_inputs", {})[param] = ""
        if tool["type"].lower()=="connector" and tool["name"].upper()=="VALIDATOR":
            
            # For validator, we need to ensure that the validation_prompt is set
            if "validation_prompt" not in tool:
                print("Validator found without validation_prompt, setting default prompt.")
                tool["validation_prompt"] = "Please specify the validation criteria, conditions and paths to take"
    
    # Add the new entry to prism.json
    # prism_entry = {"query": query.query, "prompt": tool_finder+"\nWORKFLOW TO CREATE:"+query.query, "response": tools}
    # prism_file_path = "prism.json"
    # # if os.path.exists(prism_file_path):
    # file_path = os.path.join(os.path.dirname(__file__), prism_file_path)
    # with open(file_path, "r") as prism_file:
    #     prism_data = json.load(prism_file)
    
    # prism_data.append(prism_entry)
    # file_path = os.path.join(os.path.dirname(__file__), prism_file_path)
    # with open(file_path, "w") as prism_file:
    #     json.dump(prism_data, prism_file, indent=2)





    tools["trigger"]=trigger
    tools["active"]=False

    if query.flag:
        tools["workflow_id"] = query.wid
        # Set the workflow creation timestamp
        tools["created_at"] = datetime.now().isoformat()
        tools["unavailable"]=ret_un
        try:
            # First, ensure workflows exists
            db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}},
                UpdateExpression="SET workflows = if_not_exists(workflows, :empty_map)",
                ExpressionAttributeValues={
                    ':empty_map': {'M': {}}  # Initialize as an empty map if missing
                }
            )

            # Now, update workflows by adding a new key-value pair
            response =  db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}},
                UpdateExpression="SET workflows.#wid = :new_workflow",
                ExpressionAttributeNames={
                    '#wid': tools["workflow_id"]  # New key inside workflows
                },
                ExpressionAttributeValues={
                    ':new_workflow': {
                        'M': {
                            'json': {'S': json.dumps(tools, indent=2)},
                            'prompt': {'S': query.query}
                        }
                    }
                },
                ReturnValues="UPDATED_NEW"
            )

            # print("Update succeeded:", response)
        except Exception as e:
            print("Error updating item:", e)
    else:
        tools["workflow_id"] = str(uuid.uuid4())
        # Set the workflow creation timestamp
        tools["created_at"] = datetime.now().isoformat()
        tools["unavailable"]=ret_un
        try:
            # First, ensure workflows exists
            db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}},
                UpdateExpression="SET workflows = if_not_exists(workflows, :empty_map)",
                ExpressionAttributeValues={
                    ':empty_map': {'M': {}}  # Initialize as an empty map if missing
                }
            )

            # Now, update workflows by adding a new key-value pair
            response =  db_client.update_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}},
                UpdateExpression="SET workflows.#wid = :new_workflow",
                ExpressionAttributeNames={
                    '#wid': tools["workflow_id"]  # New key inside workflows
                },
                ExpressionAttributeValues={
                    ':new_workflow': {
                        'M': {
                            'json': {'S': json.dumps(tools, indent=2)},
                            'prompt': {'S': query.query}
                        }
                    }
                },
                ReturnValues="UPDATED_NEW"
            )

            # print("Update succeeded:", response)
        except Exception as e:
            print("Error updating item:", e)

    return {"response":tools}
    



@app.get("/download_file/{file_path:path}")
async def download_file(file_path: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Downloads a file from S3 and serves it or returns a pre-signed URL for direct download
    """
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # URL decode the path if needed
        file_path = unquote(file_path)
        print(f"Decoded file path: {file_path}")
        parts=file_path.split('/')
        if len(parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid file path format")
        bucket = parts[0]  # Assuming the first part is the bucket name
        file_path = '/'.join(parts[1:])  # The rest is the file path within the bucket
        # Generate a pre-signed URL with 10-minute expiration
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': file_path},
            ExpiresIn=600  # URL valid for 10 minutes
        )
        
        return {"download_url": url}
    
    except Exception as e:
        logger.error(f"Error generating download URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accessing file: {str(e)}")
    
# Models for request validation
class OnboardingData(BaseModel):
    role: str
    customRole: Optional[str] = None
    discoverySource: str
    customDiscovery: Optional[str] = None
    experienceLevel: str
    useCase: str
    customUseCase: Optional[str] = None
    organizationSize: str
    feedbackPreference: bool
    timestamp: Optional[str] = None  # Optional timestamp field
    termsAccepted: bool
    timezone: str
    whatsappVerified: bool
    whatsappNumber: Optional[str] = None


@app.post("/save_onboarding")
async def save_onboarding(
    onboarding_data: OnboardingData,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        # Add timestamp to the data
        onboarding_data_dict = onboarding_data.model_dump()
        onboarding_data_dict["timestamp"] = datetime.now().isoformat()


        # Get user ID from token
        token = credentials.credentials
        payload = decode_token(token)
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        # Update DynamoDB record
        db_client.update_item(
            TableName='users',
            Key={
                'clerk_id': {'S': user_id}
            },
            UpdateExpression="SET onboarding = :onboarding_data",
            ExpressionAttributeValues={
                ':onboarding_data': {'M': {k: {'S': str(v)} for k, v in onboarding_data_dict.items()}}
            }
        )
        
        # Ensure api_key attribute exists before trying to update gemini key
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET api_key = if_not_exists(api_key, :empty_map)",
            ExpressionAttributeValues={
                ':empty_map': {'M': {}}
            }
        )
        
        # Then update the gemini key
        gemini_key = os.getenv("GOOGLE_API_KEY")
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET api_key.#gemini = :val",
            ExpressionAttributeNames={
                '#gemini': 'gemini'
            },
            ExpressionAttributeValues={
                ':val': {'S': gemini_key}
            }
        )
        
        
        return {"status": "success", "message": "Onboarding data saved"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving onboarding data: {str(e)}"
        )



@app.get("/get_public")
async def get_public(wid: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    try:
        # Get the specific public workflow by ID
        response = db_client.get_item(
            TableName='public_workflows',
            Key={'wid': {'S': wid}}
        )
        
        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Format the response
        item = response['Item']
        workflow = {
            'wid': item.get('wid', {}).get('S', ''),
            'json': json.loads(item.get('json', {}).get('S', '{}')),
            'uses': int(item.get('uses', {}).get('N', '0')),
            'likes': int(item.get('likes', {}).get('N', '0')),
            'comments': [comment.get('S', '') for comment in item.get('comments', {}).get('L', [])],
            'description': item.get('description', {}).get('S', ''),
        }
        workflow["name"] = workflow["json"].get("workflow_name", "Untitled Workflow")
        # workflow["description"] = workflow["json"].get("description", "enter description here")
        
        
        
        return workflow
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching public workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching public workflow: {str(e)}")
    






@app.post("/use_public_workflow")
async def use_public_workflow(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Parse request body
    body = await request.json()
    wid = body.get("wid")
    if not wid:
        raise HTTPException(status_code=400, detail="Missing workflow ID")
    
    # Get user ID from token
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # First, get the public workflow
        response = db_client.get_item(
            TableName='public_workflows',
            Key={'wid': {'S': wid}}
        )
        
        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="Public workflow not found")
        
        # Extract workflow data
        workflow_json = json.loads(response['Item']['json']['S'])
        
        # Generate a new ID for this workflow copy
        new_workflow_id = str(uuid.uuid4())
        workflow_json["workflow_id"] = new_workflow_id
        
        # Mark as copied from public template
        workflow_json["copied_from"] = wid
        workflow_json["active"] = False
        
        # Save to the user's workflows
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET workflows = if_not_exists(workflows, :empty_map)",
            ExpressionAttributeValues={
                ':empty_map': {'M': {}}
            }
        )
        
        db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET workflows.#wid = :new_workflow",
            ExpressionAttributeNames={
                '#wid': new_workflow_id
            },
            ExpressionAttributeValues={
                ':new_workflow': {
                    'M': {
                        'json': {'S': json.dumps(workflow_json, indent=2)},
                        'prompt': {'S': response['Item'].get('refined_prompt', {}).get('S', '')}
                    }
                }
            }
        )
        
        # Increment the uses count for the public workflow
        db_client.update_item(
            TableName='public_workflows',
            Key={'wid': {'S': wid}},
            UpdateExpression="SET uses = if_not_exists(uses, :zero) + :inc",
            ExpressionAttributeValues={
                ':zero': {'N': '0'},
                ':inc': {'N': '1'}
            }
        )
        
        return {
            "status": "success",
            "message": "Workflow added to your collection",
            "workflow_id": new_workflow_id,
            "workflow": workflow_json
        }
        
    except Exception as e:
        print(f"Error using public workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error using public workflow: {str(e)}")


VERIFY_TOKEN=""
from fastapi.responses import PlainTextResponse
from tools.whatsapp import download_whatsapp_media, send_whatsapp_message,send_whatsapp_options, whatsapp_verify,send_whatsapp_typing_on
from tools.dashboard import sheet_data_fetcher,Dashboard_Logic_Maker, Dashboard_Maker
import speech_recognition as sr
from pydub import AudioSegment
import subprocess
import sys
import os
import ast
class DashboardRequest(BaseModel):
    sheet_link: str
    query: str
    sheet_name: str = None  # Making sheet_name optional with default None



def get_id_from_url(url):
    # Matches both with or without a trailing slash
    match = re.search(r'/d/([a-zA-Z0-9_-]+)(?:/|$)', url)
    return match.group(1) if match else None



@app.get("/fetch_dashboards")
async def fetch_dashboards(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
        
        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="User not found")
        
        item = response['Item']
        dashboards = item.get('dashboards', {}).get('S', '[]')
        
        # Convert DynamoDB list to Python list
        dashboard_list = json.loads(dashboards)

        # Fetch all dashboards specified in the list
        dashboards = []
        for dashboard_id in dashboard_list:
            try:
                dashboard_response = db_client.get_item(
                    TableName='Dashboard_data',
                    Key={'dashboard_id': {'S': dashboard_id}}
                )
                
                if 'Item' in dashboard_response:
                    item = dashboard_response['Item']
                    dashboard_data = {
                    'dashboard_id': dashboard_id,
                    'created_at': item.get('created_at', {}).get('S', ''),
                    'sheet_id': item.get('sheet_id', {}).get('S', ''),
                    'sheet_name': item.get('sheet_name', {}).get('S', 'Sheet1'),
                    }
                    dashboards.append(dashboard_data)
            except Exception as e:
                logger.error(f"Error fetching dashboard {dashboard_id}: {str(e)}")

        return {"dashboards": dashboards}

    except Exception as e:
        logger.error(f"Error fetching dashboards: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching dashboards: {str(e)}")


@app.post("/make_dashboard")
async def make_dashboard(request: DashboardRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    
    

    response_key = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
    
    user_plan = response_key.get('Item', {}).get('plan', {}).get('S', 'free')
    if user_plan == "free" or user_plan == "trial_ended":
        logger.warning(f"User {user_id} with plan {user_plan} attempted to create workflows.")
        raise HTTPException(status_code=403, detail="Your plan does not allow creating workflows. Please upgrade to pro plan")
    
   
    count = db_client.update_item(
        TableName='users',
        Key={'clerk_id': {'S': user_id}},
        UpdateExpression="SET dashboard_created_daily_count = if_not_exists(dashboard_created_daily_count, :zero) + :one",
        ExpressionAttributeValues={
            ':zero': {'N': '0'},
            ':one': {'N': '1'}
        },
        ReturnValues="UPDATED_NEW"
    )
    logger.info(f"Updated dashboard created daily count for user {user_id}")
    count_value = count.get('Attributes', {}).get('dashboard_created_daily_count', {}).get('N', '0')
    if user_plan=="pro"  and int(count_value) > 5:
        logger.warning(f"User {user_id} has reached the daily dashboard creation limit.")
        raise HTTPException(status_code=403, detail="You have reached the daily limit of 5 dashboards. Please try again tomorrow.")
    elif user_plan=="free_15_pro" and int(count_value) > 2:
        logger.warning(f"User {user_id} has reached the daily dashboard creation limit.")
        raise HTTPException(status_code=403, detail="You have reached the daily limit of 2 dashboards. Please try again tomorrow.")
    elif user_plan=="pro++" and int(count_value) > 15:
        logger.warning(f"User {user_id} has reached the daily dashboard creation limit.")
        raise HTTPException(status_code=403, detail="You have reached the daily limit of 15 dashboards. Please try again tomorrow.")

    
    api_keys = response_key.get('Item', {}).get('api_key', {}).get('M', {})
    api_keys = {k: v['S'] for k, v in api_keys.items()}
    comp = api_keys.get("composio", "")
    gemini_key = api_keys.get("gemini", "")
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=gemini_key
    # other params...
)
    sheet_id=get_id_from_url(request.sheet_link)
    try:
        des=Dashboard_Logic_Maker(api_key=comp,llm=llm,user_id=user_id)
        representations=des.Best_representation_finder(sheetID=sheet_id, query=request.query, sheet_name=request.sheet_name)
        # time.sleep(61)
        dashboard_maker=Dashboard_Maker(api_key=comp, llm=llm,user_id=user_id)
        url = dashboard_maker.final_end_process(representations=representations,sheetID=sheet_id,sheet_name=request.sheet_name)["dashboard_url"]

        # Return the dashboard URL or data
        return {
            "status": "success",
            "dashboard_url": url,
        }
    
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating dashboard: {str(e)}")




def final_json_maker(rules:list, sheet_id:str,sheet_name:str,api_key:str):
    data_fetcher = sheet_data_fetcher(api_key=api_key)
    sheet_data = data_fetcher.GOOGLESHEETS_BATCH_GET(spreadsheetID=sheet_id, sheet_name=sheet_name)
    #print(f"Fetched sheet data: {sheet_data['response']}")

    header=f"import pandas as pd\ndf=pd.DataFrame({sheet_data['response']})\n"
    jsondata={}
    a=0
    for i in rules:
        a=a+1
        print("i:",i['representation'])
        print("starting rule processing")
        
        footer="print("+i['function_name']+"(df))"
        code=header+i['code']+"\n"+footer
        
        try:
            env = os.environ.copy()
            result=subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
            print("Subprocess output:", result.stdout)

            try:
                json_output = ast.literal_eval(result.stdout.strip())
                json_output['caption']=i['description']
            except Exception:
                json_output = result.stdout 
                 # fallback if not valid JSON
            jsondata[i['representation']+"_"+str(a)] = json_output
            
            if result.stderr:
                print("Subprocess error:", result.stderr)

                
        except subprocess.CalledProcessError as e:
            print("Subprocess failed:", e.stderr)

        
        time.sleep(1)
    print(jsondata)
    return jsondata  # Simulate some processing time
    



@app.get("/get_json")
def get_rules_and_sheet_id(dashboard_id: str,credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    try:
        response = db_client.get_item(
            TableName='Dashboard_data',
            Key={'dashboard_id': {'S': dashboard_id}}
        )
        
        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="Dashboard not found")
            
        item = response['Item']
        rules = json.loads(item.get('rules', {}).get('S', '[]'))
        sheet_name = item.get('sheet_name', {}).get('S', None)
        if not sheet_name:
            sheet_name = "Sheet1"  # Default sheet name if not provided
        sheet_id = item.get('sheet_id', {}).get('S', '')
        api_key = item.get('api_key', {}).get('S', '')
        jsondata= final_json_maker(rules=rules, sheet_id=sheet_id, sheet_name=sheet_name, api_key=api_key)
        print(jsondata)
        return jsondata

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






@app.get("/whatsapp_webhook")
async def whatsapp_webhook(request: Request):
    params = request.query_params

    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        challenge = params.get("hub.challenge")
        return PlainTextResponse(content=challenge, status_code=200)
    else:
        return PlainTextResponse(content="Verification failed", status_code=403)




async def whatsapp_process_and_reply(phone_number: str, messages: list,whatsapp=False):
    """
    This function is called to process incoming WhatsApp messages and send replies.
    It should be run in a separate thread or as a background task.
    """


    # Fetch user_id from whatsapp table using phone_number as partition key
    try:
        response = db_client.get_item(
            TableName='whatsapp',
            Key={'phone_number': {'S': phone_number}}
        )
        
        if 'Item' not in response:
            print(f"No user found for phone number {phone_number}")
            return
        
        user_id = response['Item'].get('user_id', {}).get('S')
        if not user_id:
            print(f"User ID not found for phone number {phone_number}")
            return
            
    except Exception as e:
        print(f"Error fetching user ID for phone number {phone_number}: {str(e)}")
        return
    
    if whatsapp:
        response = await instant_execution("", user_id, user_id, True)
        send_whatsapp_message( str(response),phone_number)

    else:

        final_msg=""
        for message in messages:
            
            # Check if the message looks like a local file path
            if os.path.exists(message) and os.path.isfile(message):
                # If it's an audio file, convert to text
                if message.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg')):
                    try:
                        
                        # Create temporary files for audio processing
                        temp_audio = f"/tmp/{uuid.uuid4()}.wav"
                        
                        # Convert audio file to wav format (if needed)
                        if not message.lower().endswith('.wav'):
                            audio = AudioSegment.from_file(message)
                            audio.export(temp_audio, format="wav")
                        else:
                            temp_audio = message
                        
                        # Use speech recognition to convert audio to text
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(temp_audio) as source:
                            audio_data = recognizer.record(source)
                            text = recognizer.recognize_google(audio_data)
                            
                            # Add the transcribed text to message
                            final_msg += f"\nAUDIO TRANSCRIPTION: {text}\n"
                            
                        # Clean up temporary file if it was created
                        if temp_audio != message and os.path.exists(temp_audio):
                            os.remove(temp_audio)
                    except Exception as e:
                        logger.error(f"Error transcribing audio {message}: {str(e)}")
                        final_msg += f"Error transcribing audio: {str(e)}\n"
                else:
                    try:
                        # Extract message from path
                        basename = os.path.basename(message)
                        
                        # Read file content
                        with open(message, 'rb') as file:
                            file_content = file.read()
                        
                        # Upload to S3
                        bucket_name = "personal-memory-2709"
                        s3_key = f"{user_id}/{uuid.uuid4()}_{basename}"
                        
                        s3_client.put_object(
                            Bucket=bucket_name,
                            Key=s3_key,
                            Body=file_content
                        )
                        
                        # Add S3 path to final message
                        s3_path = f"s3://{bucket_name}/{s3_key}"
                        final_msg += f"\nS3 PATH OF FILE : {s3_path}\n"
                        
                        # Clean up local file after upload
                        os.remove(message)
                        
                    except Exception as e:
                        logger.error(f"Error processing file {message}: {str(e)}")
                        final_msg += f"Error processing file {message}: {str(e)}\n"
            else:
                # Just append the text message
                final_msg += f"\nUSER MESSAGE : {message}\n"

        # Process the combined messages
        if final_msg.strip():
            # Process all the messages at once
            # session_id = str(uuid.uuid4())
            
            user_data = db_client.get_item(
                    TableName="users",
                    Key={"clerk_id": {"S": user_id}}
                )

            # Extract workflows
            workflows = user_data.get("Item", {}).get("workflows", {}).get("M", {})

            active_whatsapp_workflows = [
                {
                    "id": wid,
                    "workflow": json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}"))
                }
                for wid, workflow in workflows.items()
                if json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("active",False) and
                    json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("trigger", {}).get("name") == "TRIGGER_SIGMOYD_WHATSAPP" 
            ]
            if active_whatsapp_workflows:
                # Store current message
                timestamp = int(datetime.now().timestamp())
                initial_message = {
                    "msg": "user",
                    "content": final_msg,
                    "timestamp": timestamp
                }


                try:
                    # Add the message to the session chat history
                    session_id=user_id
                    if session_id:
                        # First check if general_chats attribute exists
                        response = db_client.get_item(
                            TableName="users",
                            Key={"clerk_id": {"S": user_id}}
                        )
                        
                        # If general_chats doesn't exist, create it first
                        if "Item" not in response or "general_chats" not in response["Item"]:
                            db_client.update_item(
                                TableName="users",
                                Key={"clerk_id": {"S": user_id}},
                                UpdateExpression="SET general_chats = :empty_map",
                                ExpressionAttributeValues={
                                    ":empty_map": {"M": {}}
                                }
                            )
                        
                        # Now add the message to the session
                        db_client.update_item(
                            TableName="users",
                            Key={"clerk_id": {"S": user_id}},
                            UpdateExpression="SET general_chats.#sid = list_append(if_not_exists(general_chats.#sid, :empty_list), :message)",
                            ExpressionAttributeNames={
                                "#sid": session_id
                            },
                            ExpressionAttributeValues={
                                ":empty_list": {"L": []},
                                ":message": {"L": [{"M": {
                                    "msg": {"S": initial_message["msg"]},
                                    "content": {"S": initial_message["content"]},
                                    "timestamp": {"N": str(initial_message["timestamp"])}
                                }}]}
                            }
                        )
                except Exception as e:
                    print(f"Error updating chat history: {e}")
                    logger.error(f"Database error when updating chat history for user {user_id}: {str(e)}")
                send_whatsapp_options(phone_number)

            else:
                response = await instant_execution(final_msg, user_id, user_id)

                # Send the response back to WhatsApp
                # Get user data to fetch API keys
                response_key = db_client.get_item(
                    TableName='users',
                    Key={'clerk_id': {'S': user_id}}
                )
                api_keys = response_key.get('Item', {}).get('api_key', {}).get('M', {})
                gemini_key = api_keys.get('gemini', {}).get('S') if 'gemini' in api_keys else ""

                # Configure with the appropriate key
                genai.configure(api_key=gemini_key)

                # Format the response to be more readable and concise
                if isinstance(response, dict):
                    model = genai.GenerativeModel('gemini-2.0-flash-lite')
                    formatted_response = await generate_con(user_id,model,
                        f"""You are "Sigmoyd," a helpful AI assistant. Your primary goal is to provide clear, concise, and user-friendly responses based on the actions you've performed. Communicate directly to the user in the second person, as if you executed the task yourself.

**Core Instructions:**

1.  **Simplify and Clarify:**
    *   Translate technical system output into simple, understandable language.
    *   **Strictly remove** all internal IDs (e.g., calendar IDs), unnecessary logs, and the word "SiGmOyD".
    *   Focus only on what the user asked for.

2.  **Format for Readability:**
    *   Present important information clearly, using bullet points for lists or complex data.
    *   Convert JSON to a readable string or list.
    *   Format dates and times into a user-friendly format (e.g., "5th Jan 2022, 10:30 AM" instead of ISO format).
    *   For calendar events, group them by day (e.g., "**Events for 5th Jan 2022:**") and list them chronologically (e.g., "• 10:30 AM - Meeting with John").

3.  **Handle Errors Gracefully:**
    *   If an error occurred, explain the reason in simple terms and suggest a solution (e.g., "It seems a required input is missing. Please provide the...").

4.  **Manage "No Data" Scenarios:**
    *   If a search or fetch operation returns no results, inform the user clearly (e.g., "I couldn't find any data matching your request.").

5.  **Be Transparent Only When Necessary:**
    *   Do not mention the tools, actions, or inputs used unless it's to explain an error.

6.  **Provide Actionable Links:**
    *   If the output includes a downloadable link (like an S3 URL), present the full, clickable link without modification.

**System & User Context:**

*   **System's Raw Output:** {response}
*   **User's Original Request:** {final_msg}

**Your Final Task:**

Based on the context above, generate your response. If the user's request implies a logical next step, ask a clarifying question at the end (e.g., "Should I add these details to the Google Sheet now?"). If no next step is apparent, do not ask.

Provide only the final, polished response. No preambles or postambles.
"""
                        )
                    # Send the formatted response
                    send_response = formatted_response
                else:
                    # If not a dict, send as is
                    send_response = response


                send_whatsapp_message( str(send_response),phone_number)
        else:
            response = await instant_execution(final_msg, user_id, user_id)

            # Send the response back to WhatsApp
            # Get user data to fetch API keys
            response_key = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}}
            )
            api_keys = response_key.get('Item', {}).get('api_key', {}).get('M', {})
            gemini_key = api_keys.get('gemini', {}).get('S') if 'gemini' in api_keys else ""

            # Configure with the appropriate key
            genai.configure(api_key=gemini_key)

            # Format the response to be more readable and concise
            if isinstance(response, dict):
                model = genai.GenerativeModel('gemini-2.0-flash-lite')
                formatted_response = await generate_con(user_id, model,
                    f"""You are "Sigmoyd," a helpful AI assistant. Your primary goal is to provide clear, concise, and user-friendly responses based on the actions you've performed. Communicate directly to the user in the second person, as if you executed the task yourself.

**Core Instructions:**

1.  **Simplify and Clarify:**
    *   Translate technical system output into simple, understandable language.
    *   **Strictly remove** all internal IDs (e.g., calendar IDs), unnecessary logs, and the word "SiGmOyD".
    *   Focus only on what the user asked for.

2.  **Format for Readability:**
    *   Present important information clearly, using bullet points for lists or complex data.
    *   Convert JSON to a readable string or list.
    *   Format dates and times into a user-friendly format (e.g., "5th Jan 2022, 10:30 AM" instead of ISO format).
    *   For calendar events, group them by day (e.g., "**Events for 5th Jan 2022:**") and list them chronologically (e.g., "• 10:30 AM - Meeting with John").

3.  **Handle Errors Gracefully:**
    *   If an error occurred, explain the reason in simple terms and suggest a solution (e.g., "It seems a required input is missing. Please provide the...").

4.  **Manage "No Data" Scenarios:**
    *   If a search or fetch operation returns no results, inform the user clearly (e.g., "I couldn't find any data matching your request.").

5.  **Be Transparent Only When Necessary:**
    *   Do not mention the tools, actions, or inputs used unless it's to explain an error.

6.  **Provide Actionable Links:**
    *   If the output includes a downloadable link (like an S3 URL), present the full, clickable link without modification.

**System & User Context:**

*   **System's Raw Output:** {response}
*   **User's Original Request:** {final_msg}

**Your Final Task:**

Based on the context above, generate your response. If the user's request implies a logical next step, ask a clarifying question at the end (e.g., "Should I add these details to the Google Sheet now?"). If no next step is apparent, do not ask.

Provide only the final, polished response. No preambles or postambles.""")
                # Send the formatted response
                send_response = formatted_response
            else:
                # If not a dict, send as is
                send_response = response


            send_whatsapp_message( str(send_response),phone_number)




class Reminder(BaseModel):
    user_id: str
    start_time: str  # ISO format string
    message: str
    repeat_interval_hours: Optional[int] = None


def start_demo(phone_number: str):
    pass
    

@app.post("/whatsapp_webhook")
async def whatsapp_webhook(request: Request):
    body = await request.json()

    j=body["entry"][0]["changes"][0]["value"]
    print("Received WhatsApp webhook:", j)
    if "contacts" in j and "messages" in j:
        print("Received a WhatsApp message")
        username= j["contacts"][0]["profile"]["name"]
        phone_number = j["contacts"][0]["wa_id"]
        print(f"Received message from {username} ({phone_number})")


        try:
            response = db_client.get_item(
                TableName='whatsapp',
                Key={'phone_number': {'S': phone_number}}
            )
            
            if 'Item' not in response:
                print(f"No user found for phone number {phone_number}")
                if phone_number in ["917000978867"]:
                    start_demo(phone_number)

                else:


                    send_whatsapp_message("Please register your WhatsApp number with Sigmoyd to use this service. Visit https://app.sigmoyd.in", phone_number)
                    return PlainTextResponse(content="Webhook received and processing started", status_code=200)
            
            user_id = response['Item'].get('user_id', {}).get('S')
            if not user_id:
                print(f"User ID not found for phone number {phone_number}")
                return PlainTextResponse(content="Webhook received and processing started", status_code=200)
            verified = response['Item'].get('verified', {}).get('BOOL', False)
            otp = response['Item'].get('otp', {}).get('S')
            for message in j["messages"]:
                if "text" in message:
                    print(f"Received message: {message['text']['body']}")
                    

                    if not verified:
                        

                        print(f"User {username} ({phone_number}) is not verified for WhatsApp")
                        
                        try:
                
                            # Check if OTP has expired (10 minutes validity)
                            created_at = datetime.fromisoformat(response['Item']['created_at']['S'])
                            if (datetime.now() - created_at).total_seconds() > 300 or message['text']['body'] != otp:
                                # Delete expired OTP
                                db_client.delete_item(
                                    TableName='whatsapp',
                                    Key={'phone_number': {'S': phone_number}}
                                )
                                send_whatsapp_message("Your OTP has been expired or wrong. Please re-register to verify your WhatsApp account.", phone_number)
                                return PlainTextResponse(content="Webhook received and processing started", status_code=200)
                            
                            
                            # Mark as verified in the whatsapp table
                            # Check if users table already has some other phone number with this user_id
                            existing_user = db_client.get_item(
                                TableName='users',
                                Key={'clerk_id': {'S': user_id}}
                            )
                            if 'Item' in existing_user:
                                existing_phone = existing_user['Item'].get('whatsapp_verified', {}).get('S')
                                if existing_phone and existing_phone != phone_number:
                                    # Remove old phone number from whatsapp table if exists
                                    # db_client.update_item(
                                    #     TableName='whatsapp',
                                    #     Key={'phone_number': {'S': existing_phone}},
                                    #     UpdateExpression="REMOVE user_id, verified, created_at"
                                    # )
                                    send_whatsapp_message(f"Your Sigmoyd account is already linked with another phone number: {existing_phone}. Please unlink it first from that number.", phone_number)
                                    return PlainTextResponse(content="Webhook received and processing started", status_code=200)

                            db_client.update_item(
                                TableName='whatsapp',
                                Key={'phone_number': {'S': phone_number}},
                                UpdateExpression="SET verified = :verified",
                                ExpressionAttributeValues={
                                    ':verified': {'BOOL': True}
                                }
                            )
                            
                            # Store the verified phone number in the user's record
                            db_client.update_item(
                                TableName='users',
                                Key={'clerk_id': {'S': user_id}},
                                UpdateExpression="SET whatsapp_verified = :phone, whatsapp_verified_at = :timestamp",
                                ExpressionAttributeValues={
                                    ':phone': {'S': phone_number},
                                    ':timestamp': {'S': datetime.now().isoformat()}
                                }
                            )



                            logger.info(f"Verified whtsapp for mobile: {phone_number}, user_id: {user_id}")


                            #TODO :
                            # Instead of asking to send msg, ask a personal question which user will like to answer



                            # Create a session reminder for WhatsApp
                            current_time = (datetime.now(pytz.timezone('Asia/Kolkata')) + timedelta(hours=23)).isoformat()
                            rem_id=create_reminder(Reminder(
                                user_id=user_id,
                                start_time=current_time,
                                message="Please send a message within one hour to keep receiving reminders and updates on WhatsApp ",
                            ))["reminder_id"]
                            try:
                                # Update the whatsapp table with the reminder_id
                                db_client.update_item(
                                    TableName='whatsapp',
                                    Key={'phone_number': {'S': phone_number}},
                                    UpdateExpression="SET reminder_id = :reminder_id",
                                    ExpressionAttributeValues={
                                        ':reminder_id': {'S': rem_id}
                                    }
                                )
                                logger.info(f"Created reminder for WhatsApp user: {phone_number}, reminder_id: {rem_id}")
                            except Exception as e:
                                logger.error(f"Failed to update whatsapp table with reminder_id: {str(e)}")

                            send_whatsapp_message("""WhatsApp verification successful! Your 15-day pro trial is active.

🧠 Create AI memory: Store and retrieve notes, images, PDFs (No auth needed)
📆 Calendar: Create events, set reminders, view schedules
✅ Notion: Track tasks, assign members, generate notification emails
📰 Gmail: Fetch and summarize newsletters
📊 Sheets: Log data from WhatsApp → auto-generate reports/dashboards
""", phone_number)
                            
                            # First, check current plan
                            user_check = db_client.get_item(
                                TableName='users',
                                Key={'clerk_id': {'S': user_id}}
                            )
                            
                            current_plan = user_check.get('Item', {}).get('plan', {}).get('S', '')
                            
                            # Only update if current plan is free
                            if current_plan == 'free' or current_plan == '':
                                response = db_client.update_item(
                                    TableName='users',
                                    Key={'clerk_id': {'S': user_id}},
                                    UpdateExpression="SET #plan_attr = :plan, trial_start_date = :start_date",
                                    ExpressionAttributeNames={
                                        '#plan_attr': 'plan'
                                    },
                                    ExpressionAttributeValues={
                                        ':plan': {'S': 'free_15_pro'},
                                        ':start_date': {'S': datetime.now().isoformat()}
                                    },
                                    ReturnValues="UPDATED_NEW"
                                )
                                logger.info(f"Updated plan for user {user_id} from free to free_15_pro trial")
                                
                            else:
                                logger.info(f"User {user_id} already has plan {current_plan}, not updating to trial")

                            # Update all existing reminders for this user with their WhatsApp number
                            try:
                                reminders_response = db_client.scan(
                                    TableName='reminders',
                                    FilterExpression='user_id = :user_id',
                                    ExpressionAttributeValues={
                                        ':user_id': {'S': user_id}
                                    }
                                )
                                
                                for reminder in reminders_response.get('Items', []):
                                    reminder_id = reminder['reminder_id']['S']
                                    db_client.update_item(
                                        TableName='reminders',
                                        Key={'reminder_id': {'S': reminder_id}},
                                        UpdateExpression="SET #number = :phone_number, #status = :active",
                                        ExpressionAttributeNames={
                                            '#number': 'number',
                                            '#status': 'status'
                                        },
                                        ExpressionAttributeValues={
                                            ':phone_number': {'S': phone_number},
                                            ':active': {'S': 'active'}
                                        }
                                    )
                                
                                logger.info(f"Updated {len(reminders_response.get('Items', []))} reminders for user {user_id} with phone number {phone_number}")
                            except Exception as e:
                                logger.error(f"Error updating reminders with phone number: {str(e)}")


                            
                            return PlainTextResponse(content="Webhook received and processing started", status_code=200)
                        
                        except Exception as e:
                            logger.error(f"Error verifying : {str(e)}")
                            return PlainTextResponse(content="Webhook received and processing started", status_code=200)
                        
                    if message["text"]["body"].strip().lower() == "/unlink":
                        # User wants to unlink WhatsApp
                        if verified:
                            db_client.delete_item(
                                TableName='whatsapp',
                                Key={'phone_number': {'S': phone_number}}
                            )
                            db_client.update_item(
                                TableName='users',
                                Key={'clerk_id': {'S': user_id}},
                                UpdateExpression="REMOVE whatsapp_verified, whatsapp_verified_at"
                            )
                            # Handle unlink whatsapp - deactivate all reminders for this user
                            try:
                                # Find all reminders for this user that aren't trigger type
                                reminders_response = db_client.scan(
                                    TableName='reminders',
                                    FilterExpression='user_id = :user_id AND reminder_type <> :trigger_type',
                                    ExpressionAttributeValues={
                                        ':user_id': {'S': user_id},
                                        ':trigger_type': {'S': 'trigger'}
                                    }
                                )
                                
                                # Update each reminder to inactive status
                                for reminder in reminders_response.get('Items', []):
                                    reminder_id = reminder['reminder_id']['S']
                                    db_client.update_item(
                                        TableName='reminders',
                                        Key={'reminder_id': {'S': reminder_id}},
                                        UpdateExpression="SET #status = :inactive",
                                        ExpressionAttributeNames={'#status': 'status'},
                                        ExpressionAttributeValues={':inactive': {'S': 'inactive'}}
                                    )
                                
                                logger.info(f"Deactivated all reminders for user {user_id} after unlinking WhatsApp")
                            except Exception as e:
                                logger.error(f"Error deactivating reminders after WhatsApp unlink: {str(e)}")
                            # Optionally, send a confirmation message
                            send_whatsapp_message("Your WhatsApp account has been unlinked with the current sigmoyd account", phone_number)
                            print(f"Unlinked WhatsApp account for {phone_number}")
                        
                        return PlainTextResponse(content="Webhook received and processing started", status_code=200)
            
                
        except Exception as e:
            print(f"Error fetching user ID for phone number {phone_number}: {str(e)}")
            return PlainTextResponse(content="Webhook received and processing started", status_code=200)
        # Create WHATSAPP directory if it doesn't exist
        whatsapp_dir = os.path.join(os.path.dirname(__file__), 'WHATSAPP')
        os.makedirs(whatsapp_dir, exist_ok=True)

        # Check for duplicate messages
        phone_history_file = os.path.join(whatsapp_dir, f"{phone_number}.json")
        recent_messages = []

        # Load existing message history if available
        if os.path.exists(phone_history_file):
            try:
                with open(phone_history_file, 'r') as f:
                    recent_messages = json.load(f)
            except json.JSONDecodeError:
                recent_messages = []

        # Get the message ID to check for duplicates
        message_id = j["messages"][0]["id"]

        # Skip if it's a duplicate message
        if message_id in recent_messages:
            print(f"Skipping duplicate message with ID: {message_id}")
            return PlainTextResponse(content="Duplicate message", status_code=200)

        send_whatsapp_typing_on(message_id)
        # Add the current message ID to the history and keep only the 2 most recent
        recent_messages.insert(0, message_id)
        if len(recent_messages) > 2:
            recent_messages = recent_messages[:2]

        # Save the updated message history
        with open(phone_history_file, 'w') as f:
            json.dump(recent_messages, f)

        # Check if reminder_id exists in whatsapp table
        reminder_response = db_client.get_item(
            TableName='whatsapp',
            Key={'phone_number': {'S': phone_number}}
        )
        
        # If reminder_id exists, update the start_time in reminders table
        if 'Item' in reminder_response and 'reminder_id' in reminder_response['Item']:
            reminder_id = reminder_response['Item']['reminder_id'].get('S')
            if reminder_id:
                try:
                    # Check if this reminder exists in reminders table
                    reminder_item = db_client.get_item(
                    TableName='reminders',
                    Key={'reminder_id': {'S': reminder_id}}
                    )
                    
                    if 'Item' in reminder_item:
                    # Calculate new time (current Indian time + 23 hours)
                        new_time = (datetime.now(pytz.timezone('Asia/Kolkata')) + timedelta(hours=23)).isoformat()
                        
                        # Update the reminder's start_time
                        db_client.update_item(
                            TableName='reminders',
                            Key={'reminder_id': {'S': reminder_id}},
                            UpdateExpression="SET start_time = :new_time",
                            ExpressionAttributeValues={
                            ':new_time': {'S': new_time}
                            }
                        )
                        logger.info(f"Updated reminder {reminder_id} with new start_time: {new_time}")
                    else:
                        current_time = (datetime.now(pytz.timezone('Asia/Kolkata')) + timedelta(hours=23)).isoformat()

                        #TODO :
                        # Instead of asking to send msg, ask a personal question which user will like to answer



                        rem_id=create_reminder(Reminder(
                            user_id=user_id,
                            start_time=current_time,
                            message="Please send a message within one hour to keep receiving reminders and updates on WhatsApp ",
                        ))["reminder_id"]
                        try:
                            # Update the whatsapp table with the reminder_id
                            db_client.update_item(
                                TableName='whatsapp',
                                Key={'phone_number': {'S': phone_number}},
                                UpdateExpression="SET reminder_id = :reminder_id",
                                ExpressionAttributeValues={
                                    ':reminder_id': {'S': rem_id}
                                }
                            )
                            logger.info(f"Created reminder for WhatsApp user: {phone_number}, reminder_id: {rem_id}")
                        except Exception as e:
                            logger.error(f"Failed to update whatsapp table with reminder_id: {str(e)}")
                except Exception as e:
                    logger.error(f"Error updating reminder: {str(e)}")

        
        all=[]

        # Update user's WhatsApp message count
        if j["messages"][0].get("interactive", {}).get("type", "") != "button_reply":
            try:
                # Increment whatsapp_count if it exists, otherwise create it with value 1
                response = db_client.update_item(
                    TableName='users',
                    Key={'clerk_id': {'S': user_id}},
                    UpdateExpression="SET whatsapp_count = if_not_exists(whatsapp_count, :zero) + :one",
                    ExpressionAttributeValues={
                        ':zero': {'N': '0'},
                        ':one': {'N': '1'}
                    },
                    ReturnValues="UPDATED_NEW"
                )
                logger.info(f"Updated WhatsApp count for user {user_id}")
                updated_count = response.get("Attributes", {}).get("whatsapp_count", {}).get("N", 0)
            except Exception as e:
                logger.error(f"Error updating WhatsApp message count: {str(e)}")

            
            user_data = db_client.get_item(
                TableName="users",
                Key={"clerk_id": {"S": user_id}}
            )
            
            
            # Check the user's plan
            user_plan = user_data.get('Item', {}).get('plan', {}).get('S', 'free')
            logger.info(f"User {user_id} with plan {user_plan} sent a whatsapp message")

            if user_plan == 'free' or user_plan == 'trial_ended':
                send_whatsapp_message("Please upgrade your plan : https://app.sigmoyd.in/pricing", phone_number)
                return PlainTextResponse(content="Daily limit exceeded", status_code=200)
                
                
            elif user_plan == 'free_15_pro':
                # For free plan users, check if they've exceeded their daily limit
                if int(updated_count) > 25:
                    send_whatsapp_message("You have exceeded your daily limit of 25 messages + workflow runs for the free plan. Please upgrade to continue using WhatsApp features from the link : https://app.sigmoyd.in/pricing", phone_number)
                    return PlainTextResponse(content="Daily limit exceeded", status_code=200)
                
            elif user_plan == 'pro':
                # For pro plan users, check if they've exceeded their daily limit
                if int(updated_count) > 50:
                    send_whatsapp_message("You have exceeded your daily limit of 50 messages + workflow runs for the pro plan. Please upgrade to continue using WhatsApp features from the link : https://app.sigmoyd.in/pricing", phone_number)
                    return PlainTextResponse(content="Daily limit exceeded", status_code=200)

            elif user_plan == 'pro++':
                # For pro++ plan users, check if they've exceeded their daily limit
                if int(updated_count) > 150:
                    send_whatsapp_message("You have exceeded your daily limit of 150 messages + workflow runs for the pro++ plan.", phone_number)
                    return PlainTextResponse(content="Daily limit exceeded", status_code=200)

        for message in j["messages"]:
            print("Processing message:", message)
            if "text" in message:
                print(f"Received message: {message['text']['body']}")
                all.append(message["text"]["body"])
                # do something with the text message
            elif "audio" in message:
                print(f"Received audio message with ID: {message['audio']['id']}")
                path= download_whatsapp_media(message["audio"]["id"],message["audio"]["mime_type"])
                print("Received audio message, saved to path:", path)
                all.append(path)
                if "caption" in message["audio"]:
                    all.append(message["audio"]["caption"])
            elif "image" in message:
                print(f"Received image message with ID: {message['image']['id']}")
                path= download_whatsapp_media(message["image"]["id"],message["image"]["mime_type"])
                print("Received image message, saved to path:", path)
                all.append(path)
                if "caption" in message["image"]:
                    all.append(message["image"]["caption"])
            elif "video" in message:
                print(f"Received video message with ID: {message['video']['id']}")
                path= download_whatsapp_media(message["video"]["id"],message["video"]["mime_type"])
                print("Received video message, saved to path:", path)
                all.append(path)
                if "caption" in message["video"]:
                    all.append(message["video"]["caption"])
            elif "document" in message:
                print(f"Received document message with ID: {message['document']['id']}")
                filename= message["document"].get("filename", "unknown_document")
                path= download_whatsapp_media(message["document"]["id"],message["document"]["mime_type"])
                print("Received document message, saved to path:", path)
                all.append(path)
                if "caption" in message["document"]:
                    all.append(message["document"]["caption"])
            else:
                print("Received unknown message type:", message)
                if message["interactive"]["type"] == "button_reply":
                    button_id = message["interactive"]["button_reply"]["id"]
                    print(f"Button with ID {button_id} was pressed.")
                    if button_id == "1":
                        await whatsapp_process_and_reply(phone_number, all,whatsapp=True)
                        return PlainTextResponse(content="Webhook received and processing started", status_code=200)
        # Process all messages as needed
        print("All messages received:", all)
        await whatsapp_process_and_reply(phone_number, all)
        # Clean up any local file paths after processing
        try:
            for message_path in all:
                if os.path.exists(message_path) and os.path.isfile(message_path):
                    print(f"Cleaning up local file: {message_path}")
                    os.remove(message_path)
        except Exception as e:
            print(f"Error while cleaning up local files: {e}")
        return PlainTextResponse(content="Webhook received and processing started", status_code=200)
    





# Models for WhatsApp OTP verification
class GenerateOTPRequest(BaseModel):
    mobile_number: str

class VerifyOTPRequest(BaseModel):
    mobile_number: str
    otp: str

@app.post("/generate_otp")
async def generate_otp(request: GenerateOTPRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # Generate a random 6-digit OTP
        otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # Store the OTP with the mobile number and user_id in DynamoDB
        # Check if the phone number already exists in the whatsapp table
        response = db_client.get_item(
            TableName='whatsapp',
            Key={'phone_number': {'S': request.mobile_number}}
        )
        if 'Item' not in response or not response['Item'].get('verified', {}).get('BOOL', False):
            db_client.put_item(
            TableName='whatsapp',
            Item={
                'phone_number': {'S': request.mobile_number},
                'otp': {'S': otp},
                'user_id': {'S': user_id},
                'created_at': {'S': datetime.now().isoformat()},
                'verified': {'BOOL': False}
            }
            )
        

            clerk_sdk = Clerk(bearer_auth=clerk_secret_key)
            user_details = clerk_sdk.users.list(user_id=[user_id])[0]
            whatsapp_verify(user_details.first_name, request.mobile_number,user_details.email_addresses[0].email_address)

            logger.info(f"Generated verification request for mobile: {request.mobile_number}, user_id: {user_id}")

            return {"status": "success", "message": "Verification process initiated. Please check your WhatsApp", "otp": otp}
        else:
            raise HTTPException(status_code=400, detail="WhatsApp number already verified")
    
    except Exception as e:
        logger.error(f"Error generating OTP: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate OTP: {str(e)}")




class PeriodicTrigger(BaseModel):
    wid: str
    user_id: str
    key: str




@app.post("/start_workflow")
async def start_workflow(request: PeriodicTrigger):
    workflow_id = request.wid
    user_id = request.user_id
    key = request.key
    if not key == "":
        raise HTTPException(status_code=403, detail="Invalid key")

    try:
        # Get the workflow from the database
        user_data = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )
        
        workflows = user_data.get("Item", {}).get("workflows", {}).get("M", {})
        
        active_periodic_workflows = [
            {
                "id": wid,
                "workflow": json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}"))
            }
            for wid, workflow in workflows.items()
            if json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("active",False) and
                json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("trigger", {}).get("name") == "TRIGGER_PERIODIC" 
        ]

        for workflow in active_periodic_workflows:
            if workflow["id"] == workflow_id:
                syn.delay(workflow["id"], workflow["workflow"]["workflow"], user_id,"")
        
        return {"status": "success", "message": "Workflow started", "workflow_id": workflow_id}
        
    except Exception as e:
        logger.error(f"Error starting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting workflow: {str(e)}")


class CustomTriggerRequest(BaseModel):
    wid: str
    user_id: str
    key: str
    trigger_details: dict


@app.post("/start_workflow_custom")
async def start_workflow(request: CustomTriggerRequest):
    workflow_id = request.wid
    user_id = request.user_id
    key = request.key
    trigger_details = request.trigger_details
    if not key == "":
        raise HTTPException(status_code=403, detail="Invalid key")

    try:
        # Get the workflow from the database
        user_data = db_client.get_item(
                TableName="users",
                Key={"clerk_id": {"S": user_id}}
        )

        # Extract workflows
        workflows = user_data.get("Item", {}).get("workflows", {}).get("M", {})

        active_whatsapp_workflows = [
            {
                "id": wid,
                "workflow": json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}"))
            }
            for wid, workflow in workflows.items()
            if json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("active",False) and
                json.loads(workflow.get("M", {}).get("json", {}).get("S", "{}")).get("trigger", {}).get("name") == "TRIGGER_SIGMOYD_WHATSAPP" 
        ]

        
        
        print("Executing WhatsApp workflow")
        for workflow in active_whatsapp_workflows:
            if workflow["id"] == workflow_id:
                syn.delay(workflow["id"], workflow["workflow"]["workflow"], user_id, trigger_details)
        return {"status": "success", "message": "Workflow started", "workflow_id": workflow_id}
        
    except Exception as e:
        logger.error(f"Error starting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting workflow: {str(e)}")







# ✅ Define your Pydantic models
class QuestionItem(BaseModel):
    question: str
    type: str = "text"  # text, radio, select, checkbox, number, date, file
    required: bool = False
    options: Optional[List[str]] = None
    # File upload specific fields
    file_types: Optional[List[str]] = None  # e.g., ["pdf", "jpg", "png"]
    max_file_size_mb: Optional[int] = 10
    drive_folder_id: Optional[str] = None

class FormInput(BaseModel):
    title: str
    description: str
    spreadsheet_url: Optional[str] = None
    dashboard: Optional[str] = None
    question: List[QuestionItem]
    drive_folder_id: Optional[str] = None  # Single folder for entire form

class FormDraft(BaseModel):
    form_id: Optional[str] = None  # None for new forms
    title: str
    description: str
    questions: List[QuestionItem]
    drive_folder_id: Optional[str] = None  # Single folder for entire form

class FormUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    questions: Optional[List[QuestionItem]] = None
    dashboard: Optional[str] = None

class FormSubmission(BaseModel):
    form_id: str
    responses: List[dict]

class FormSubmissionData(BaseModel):
    form_id: str
    responses: List[dict]

def get_id_from_url(url):
    """Extract spreadsheet ID from Google Sheets URL"""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def check_connection(app_name: str, api_key: str) -> bool:
    """Check if user has connected their account to Composio"""
    try:
        composio = Composio(api_key=api_key)
        connected_accounts = composio.connected_accounts.get()

        for acc in connected_accounts:
            if app_name.lower() in acc.appName.lower() and acc.status == 'ACTIVE':
                return True
        return False
    except Exception as e:
        print(f"Error checking connection: {e}")
        return False

def GOOGLESHEETS_BATCH_GET(spreadsheetID, sheet_name=None, api_key=None):
    """Get data from Google Sheets"""
    try:
        composio = Composio(api_key=api_key)

        # Get connected account
        connected_accounts = composio.connected_accounts.get()
        google_sheets_accounts = [acc for acc in connected_accounts if acc.appName == 'googlesheets' and acc.status == 'ACTIVE']

        if not google_sheets_accounts:
            raise Exception("No active Google Sheets account found")

        connected_account_id = google_sheets_accounts[0].id

        # Use the core Composio client to execute actions
        response = composio.actions.execute(
            action=Action.GOOGLESHEETS_BATCH_GET,
            params={
                'spreadsheet_id': spreadsheetID,
                "ranges": [sheet_name] if sheet_name else ["Sheet1"]
            },
            entity_id='default',
            connected_account=connected_account_id,
            session_id=str(uuid.uuid4())
        )

        values = response['data']['valueRanges'][0].get('values', [])
        header = values[0] if values else []
        rows = values[1:] if len(values) > 1 else []

        mapped_data = [dict(zip(header, row)) for row in rows]
        print("mapped data:", mapped_data)

        return {'response': mapped_data, "headers": header}
    except Exception as e:
        print(f"Error in GOOGLESHEETS_BATCH_GET: {e}")
        raise e

def create_real_spreadsheet(title, questions, composio_api_key):
    """Create a Google Spreadsheet with columns"""
    try:
        if not check_connection("googlesheets", composio_api_key):
            raise Exception("No active Google Sheets account found. Please connect Google Sheets in Composio.")

        composio = Composio(api_key=composio_api_key)

        # Get connected account
        connected_accounts = composio.connected_accounts.get()
        google_sheets_accounts = [acc for acc in connected_accounts if acc.appName == 'googlesheets' and acc.status == 'ACTIVE']

        if not google_sheets_accounts:
            raise Exception("No active Google Sheets account found")

        connected_account_id = google_sheets_accounts[0].id

        # Create spreadsheet
        response3 = composio.actions.execute(
            action=Action.GOOGLESHEETS_CREATE_GOOGLE_SHEET1,
            params={'title': title},
            entity_id='default',
            connected_account=connected_account_id,
            session_id=str(uuid.uuid4())
        )

        if not response3.get('successfull', False):
            raise Exception(f"Failed to create spreadsheet: {response3}")

        spreadsheet_id = response3['data']['response_data']['spreadsheet_id']
        spreadsheet_url = "https://docs.google.com/spreadsheets/d/" + str(spreadsheet_id)

        # Second API call to fill sheet with questions as headers
        if questions:
            composio.actions.execute(
                action=Action.GOOGLESHEETS_BATCH_UPDATE,
                params={
                    'spreadsheet_id': spreadsheet_id,
                    'sheet_name': 'Sheet1',
                    'values': [questions],
                    'first_cell_location': 'A1'
                },
                entity_id='default',
                connected_account=connected_account_id,
                session_id=str(uuid.uuid4())
            )
        else:
            raise Exception("No questions provided to create spreadsheet")

        return {
            'spreadsheet_id': spreadsheet_id,
            'spreadsheet_url': spreadsheet_url,
            'title': title,
            'questions': questions
        }
    except Exception as e:
        print(f"Error creating real spreadsheet: {e}")
        raise e

def create_drive_folder(folder_name: str, composio_api_key: str, parent_folder_id: Optional[str] = None):
    """Create a folder in Google Drive using Composio"""
    try:
        if not check_connection("googledrive", composio_api_key):
            raise Exception("No active Google Drive account found. Please connect Google Drive in Composio.")

        composio = Composio(api_key=composio_api_key)

        # Get connected account
        connected_accounts = composio.connected_accounts.get()
        google_drive_accounts = [acc for acc in connected_accounts if acc.appName == 'googledrive' and acc.status == 'ACTIVE']

        if not google_drive_accounts:
            raise Exception("No active Google Drive account found")

        connected_account_id = google_drive_accounts[0].id

        params = {
            'folder_name': folder_name
        }

        if parent_folder_id:
            params['parent_id'] = parent_folder_id

        response = composio.actions.execute(
            action=Action.GOOGLEDRIVE_CREATE_FOLDER,
            params=params,
            entity_id='default',
            connected_account=connected_account_id,
            session_id=str(uuid.uuid4())
        )

        print(f"DEBUG: Composio create folder response: {response}")

        if response.get('successfull'):
            # Folder data is directly in response['data'], not response['data']['response_data']
            folder_data = response.get('data', {})
            print(f"DEBUG: Folder data from response: {folder_data}")
            folder_id = folder_data.get('id')
            print(f"DEBUG: Extracted folder_id: {folder_id}")
            folder_url = folder_data.get('webViewLink', f'https://drive.google.com/drive/folders/{folder_id}')

            return {
                'folder_id': folder_id,
                'folder_url': folder_url,
                'folder_name': folder_name
            }
        else:
            raise Exception(f"Failed to create folder: {response}")

    except Exception as e:
        print(f"Error creating Drive folder: {e}")
        raise e

def upload_file_to_drive(file_content: bytes, file_name: str, mime_type: str,
                         folder_id: str, composio_api_key: str):
    """Upload a file to Google Drive folder using direct API (supports files up to 10MB)"""
    try:
        if not check_connection("googledrive", composio_api_key):
            raise Exception("No active Google Drive account found.")

        composio = Composio(api_key=composio_api_key)

        # Get connected account
        connected_accounts = composio.connected_accounts.get()
        google_drive_accounts = [acc for acc in connected_accounts if acc.appName == 'googledrive' and acc.status == 'ACTIVE']

        if not google_drive_accounts:
            raise Exception("No active Google Drive account found")

        connected_account = google_drive_accounts[0]

        # Get the access token from the connected account
        # Composio stores OAuth tokens in the connected account
        access_token = None
        try:
            # Try to get access token from connection params
            if hasattr(connected_account, 'connectionParams'):
                access_token = connected_account.connectionParams.get('access_token')

            # If not found, use Composio's API to get it
            if not access_token:
                # Use Composio's connection to get credentials
                import requests as req
                headers = {'X-API-Key': composio_api_key}
                conn_resp = req.get(
                    f'https://backend.composio.dev/api/v1/connectedAccounts/{connected_account.id}',
                    headers=headers
                )
                if conn_resp.status_code == 200:
                    conn_data = conn_resp.json()
                    access_token = conn_data.get('connectionParams', {}).get('access_token')
        except Exception as token_error:
            print(f"Warning: Could not get access token directly: {token_error}")

        # Always use direct Google Drive API to avoid Composio's 1MB limit
        print(f"Using direct Google Drive API for file upload (size: {len(file_content)} bytes)")

        # Get access token if not already available
        if not access_token:
            try:
                import requests as req
                headers = {'X-API-Key': composio_api_key}
                conn_resp = req.get(
                    f'https://backend.composio.dev/api/v1/connectedAccounts/{connected_account.id}',
                    headers=headers
                )
                if conn_resp.status_code == 200:
                    conn_data = conn_resp.json()
                    access_token = conn_data.get('connectionParams', {}).get('access_token')

                if not access_token:
                    raise Exception("Could not get access token from Composio")
            except Exception as token_error:
                raise Exception(f"Failed to get access token: {token_error}")

        # Step 1: Create file metadata
        metadata = {
            'name': file_name,
            'parents': [folder_id]
        }

        print(f"DEBUG: Uploading file '{file_name}' to folder ID: {folder_id}")
        print(f"DEBUG: Metadata: {metadata}")

        # Step 2: Upload using multipart upload
        import requests as req
        from io import BytesIO

        headers = {
            'Authorization': f'Bearer {access_token}',
        }

        # Create multipart upload
        files = {
            'metadata': ('metadata', json.dumps(metadata), 'application/json'),
            'file': (file_name, BytesIO(file_content), mime_type)
        }

        # Add fields parameter to get full response including parents
        upload_response = req.post(
            'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,mimeType,parents,webViewLink',
            headers=headers,
            files=files
        )

        if upload_response.status_code not in [200, 201]:
            raise Exception(f"Failed to upload file: {upload_response.text}")

        file_data = upload_response.json()
        file_id = file_data.get('id')
        file_url = f"https://drive.google.com/file/d/{file_id}/view"

        print(f"DEBUG: File uploaded successfully. File ID: {file_id}")
        print(f"DEBUG: Full file response data: {file_data}")
        print(f"DEBUG: File parents from response: {file_data.get('parents', 'No parents in response')}")

        # Make file publicly accessible using Composio
        try:
            share_params = {
                'file_id': file_id,
                'type': 'anyone',
                'role': 'reader'
            }
            composio.actions.execute(
                action=Action.GOOGLEDRIVE_ADD_FILE_SHARING_PREFERENCE,
                params=share_params,
                entity_id='default',
                connected_account=connected_account.id,
                session_id=str(uuid.uuid4())
            )
        except Exception as share_error:
            print(f"Warning: Could not share file publicly: {share_error}")

        return {
            'file_id': file_id,
            'file_url': file_url,
            'file_name': file_name,
            'mime_type': mime_type
        }

    except Exception as e:
        print(f"Error uploading file to Drive: {e}")
        raise e


def create_form_folder(form_title: str, composio_api_key: str) -> Optional[str]:
    """Create a single Google Drive folder for the entire form"""
    try:
        if not check_connection("googledrive", composio_api_key):
            print("Warning: Google Drive not connected, skipping folder creation")
            return None

        folder_name = f"{form_title} - Form Uploads"
        folder_result = create_drive_folder(folder_name, composio_api_key)
        folder_id = folder_result['folder_id']
        print(f"Created form folder: {folder_name} (ID: {folder_id})")
        return folder_id

    except Exception as e:
        print(f"Error creating form folder: {e}")
        return None


@app.get("/")
async def root():
    return {
        "message": "Forms API with File Upload Support",
        "composio_available": COMPOSIO_AVAILABLE,
        "dynamodb_available": DYNAMODB_AVAILABLE,
        "features": ["Google Sheets", "Google Drive", "File Uploads"]
    }

@app.head("/")
async def root_head():
    return


@app.post("/get_upload_url")
async def get_upload_url(
    form_id: str = Form(...),
    question_index: int = Form(...),
    file_name: str = Form(...),
    mime_type: str = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate a resumable upload URL for direct browser upload to Google Drive"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    try:
        # Get form data to find the owner and folder
        table = dynamodb.Table('forms')
        response = table.get_item(Key={'form_id': form_id})
        form_item = response.get('Item')

        if not form_item:
            return {'error': 'Form not found'}

        form_owner_id = form_item.get('user_id')

        # Get form owner's Composio API key
        api_keys_response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': form_owner_id}}
        )
        api_keys = api_keys_response.get('Item', {}).get('api_key', {}).get('M', {})
        api_keys = {k: v['S'] for k, v in api_keys.items()}
        composio_api = api_keys.get('composio')

        if not composio_api:
            return {'error': 'Form owner has not connected Google Drive'}

        # Get folder ID from form (not per question)
        folder_id = form_item.get('drive_folder_id')

        if not folder_id:
            # Create folder if doesn't exist
            folder_name = f"{form_item.get('title', 'Form')} - Form Uploads"
            folder_result = create_drive_folder(folder_name, composio_api)
            folder_id = folder_result['folder_id']

            # Update form with folder_id
            table.update_item(
                Key={'form_id': form_id},
                UpdateExpression='SET drive_folder_id = :folder_id',
                ExpressionAttributeValues={':folder_id': folder_id}
            )

        # Get access token from Composio
        composio = Composio(api_key=composio_api)
        connected_accounts = composio.connected_accounts.get()
        google_drive_accounts = [acc for acc in connected_accounts if acc.appName == 'googledrive' and acc.status == 'ACTIVE']

        if not google_drive_accounts:
            return {'error': 'No active Google Drive account found'}

        connected_account = google_drive_accounts[0]

        # Get access token
        import requests as req
        headers = {'X-API-Key': composio_api}
        conn_resp = req.get(
            f'https://backend.composio.dev/api/v1/connectedAccounts/{connected_account.id}',
            headers=headers
        )

        if conn_resp.status_code != 200:
            return {'error': 'Failed to get access token'}

        conn_data = conn_resp.json()
        access_token = conn_data.get('connectionParams', {}).get('access_token')

        if not access_token:
            return {'error': 'Could not get access token'}

        # Create resumable upload session
        metadata = {
            'name': file_name,
            'parents': [folder_id]
        }

        session_response = req.post(
            'https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable',
            headers={
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json; charset=UTF-8',
                'X-Upload-Content-Type': mime_type
            },
            json=metadata
        )

        if session_response.status_code not in [200, 201]:
            return {'error': f'Failed to create upload session: {session_response.text}'}

        upload_url = session_response.headers.get('Location')

        return {
            'success': True,
            'upload_url': upload_url,
            'access_token': access_token,
            'folder_id': folder_id,
            'message': 'Upload URL generated successfully'
        }

    except Exception as e:
        print(f"ERROR: Error generating upload URL: {e}")
        return {'error': f'Failed to generate upload URL: {str(e)}'}



@app.post("/share_file")
async def share_file(
    file_id: str = Form(...),
    form_id: str = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Make uploaded file publicly accessible (background task)"""
    try:
        token = credentials.credentials
        payload = decode_token(token)
        user_id = payload.get('sub')

        # Get form owner
        table = dynamodb.Table('forms')
        response = table.get_item(Key={'form_id': form_id})
        form_item = response.get('Item')

        if not form_item:
            return {'success': False, 'error': 'Form not found'}

        form_owner_id = form_item.get('user_id')

        # Get Composio API key
        api_keys_response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': form_owner_id}}
        )
        api_keys = api_keys_response.get('Item', {}).get('api_key', {}).get('M', {})
        api_keys = {k: v['S'] for k, v in api_keys.items()}
        composio_api = api_keys.get('composio')

        # Share file using Composio
        composio = Composio(api_key=composio_api)
        connected_accounts = composio.connected_accounts.get()
        google_drive_accounts = [acc for acc in connected_accounts if acc.appName == 'googledrive' and acc.status == 'ACTIVE']

        if google_drive_accounts:
            share_params = {
                'file_id': file_id,
                'type': 'anyone',
                'role': 'reader'
            }
            composio.actions.execute(
                action=Action.GOOGLEDRIVE_ADD_FILE_SHARING_PREFERENCE,
                params=share_params,
                entity_id='default',
                connected_account=google_drive_accounts[0].id,
                session_id=str(uuid.uuid4())
            )

        return {'success': True, 'message': 'File shared successfully'}

    except Exception as e:
        print(f"Warning: Could not share file: {e}")
        return {'success': False, 'error': str(e)}


@app.post("/create_sheet")
async def get_dashboard(form_json: FormInput, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    try:
        api_keys_response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
        api_keys = api_keys_response.get('Item', {}).get('api_key', {}).get('M', {})
        api_keys = {k: v['S'] for k, v in api_keys.items()}
        composio_api = api_keys.get('composio')

        # Extract questions and add Email column
        questions = [q.question for q in form_json.question]
        questions.append('Email')  # Add Email column

        # Create spreadsheet using modern API
        sheet_result = create_real_spreadsheet(form_json.title, questions, composio_api)

        return {
            'spreadsheet_url': sheet_result['spreadsheet_url'],
            'spreadsheet_id': sheet_result['spreadsheet_id'],
            'message': 'Real Google Spreadsheet created successfully with modern Composio API'
        }

    except Exception as e:
        print(f"ERROR: Error creating real spreadsheet: {e}")
        return {'error': f'Failed to create spreadsheet: {str(e)}'}


@app.post("/upload_form_file")
async def upload_form_file(
    form_id: str = Form(...),
    question_index: int = Form(...),
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Upload a file for a form submission"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    try:
        if not DYNAMODB_AVAILABLE:
            return {'error': 'Database not available'}

        table = dynamodb.Table('forms')
        response = table.get_item(Key={'form_id': form_id})
        form_item = response.get('Item')

        if not form_item:
            return {'error': 'Form not found'}

        # Get form owner's Composio API key
        form_owner_id = form_item.get('user_id')
        api_keys_response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': form_owner_id}}
        )
        api_keys = api_keys_response.get('Item', {}).get('api_key', {}).get('M', {})
        api_keys = {k: v['S'] for k, v in api_keys.items()}
        composio_api = api_keys.get('composio')

        if not composio_api:
            return {'error': 'Form owner has not connected Google Drive'}

        # Get question data for validation
        questions = form_item.get('question', [])
        if question_index >= len(questions):
            return {'error': 'Invalid question index'}

        question = questions[question_index]

        # Get or create folder for the form (not per question)
        folder_id = form_item.get('drive_folder_id')
        print(f"DEBUG: Form folder_id from database: {folder_id}")

        if not folder_id:
            # Create folder for entire form
            folder_name = f"{form_item.get('title', 'Form')} - Form Uploads"
            print(f"DEBUG: Creating new folder: {folder_name}")
            folder_result = create_drive_folder(folder_name, composio_api)
            folder_id = folder_result['folder_id']
            print(f"DEBUG: New folder created with ID: {folder_id}")

            # Update form with folder_id
            table.update_item(
                Key={'form_id': form_id},
                UpdateExpression='SET drive_folder_id = :folder_id',
                ExpressionAttributeValues={':folder_id': folder_id}
            )
        else:
            print(f"DEBUG: Using existing folder ID: {folder_id}")

        # Read file content
        file_content = await file.read()

        # Check file size
        max_size_mb = question.get('max_file_size_mb', 10)
        if len(file_content) > max_size_mb * 1024 * 1024:
            return {'error': f'File size exceeds maximum allowed size of {max_size_mb}MB'}

        # Get mime type
        mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'

        # Upload to Drive
        upload_result = upload_file_to_drive(
            file_content,
            file.filename,
            mime_type,
            folder_id,
            composio_api
        )

        return {
            'success': True,
            'file_id': upload_result['file_id'],
            'file_url': upload_result['file_url'],
            'file_name': upload_result['file_name'],
            'message': 'File uploaded successfully'
        }

    except Exception as e:
        print(f"ERROR: Error uploading file: {e}")
        return {'error': f'Failed to upload file: {str(e)}'}


@app.post("/publish")
async def publish(form_json: FormInput, form_id: str = None, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Publish form - create new or update existing form status"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    try:
        if not DYNAMODB_AVAILABLE:
            return {'error': 'Database not available'}

        table = dynamodb.Table('forms')

        # Get user's Composio API key for creating folders
        try:
            api_keys_response = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}}
            )
            api_keys = api_keys_response.get('Item', {}).get('api_key', {}).get('M', {})
            api_keys = {k: v['S'] for k, v in api_keys.items()}
            composio_api = api_keys.get('composio')
        except Exception as e:
            print(f"Warning: Could not get Composio API key: {e}")
            composio_api = None

        # Create single folder for form if it has file upload questions
        questions = [q.dict() for q in form_json.question]
        form_folder_id = None

        # Check if form has any file upload questions
        has_file_questions = any(q.get('type') == 'file' for q in questions)

        if has_file_questions and composio_api:
            # Create one folder for the entire form
            form_folder_id = create_form_folder(form_json.title, composio_api)

        # Generate public form URL
        if not form_id:
            form_id = str(uuid.uuid4())

        public_form_url = f'https://app.sigmoyd.in/form?form_id={form_id}'  # Use localhost for testing

        final_json = {
            'form_id': form_id,
            'title': form_json.title,
            'description': form_json.description,
            'spreadsheet_url': form_json.spreadsheet_url,
            'dashboard': form_json.dashboard,
            'public_url': public_form_url,
            'question': questions,
            'drive_folder_id': form_folder_id,  # Store folder at form level
            'user_id': user_id,
            'status': 'published',
            'submissions': 0,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'updated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'workflows': [],
            'active': True
        }

        # Insert/update the item
        table.put_item(Item=final_json)

        return {
            'success': True,
            'form_url': public_form_url,
            'form_id': form_id,
            'spreadsheet_url': form_json.spreadsheet_url,
            'status': 'published'
        }

    except Exception as e:
        print(f"Error publishing form: {e}")
        return {'error': f'Failed to publish form: {str(e)}'}


@app.post("/save_draft")
async def save_draft(form_draft: FormDraft, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Save form as draft - create new or update existing"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    try:
        if not DYNAMODB_AVAILABLE:
            return {'error': 'Database not available'}

        table = dynamodb.Table('forms')

        # Get user's Composio API key for creating folders
        try:
            api_keys_response = db_client.get_item(
                TableName='users',
                Key={'clerk_id': {'S': user_id}}
            )
            api_keys = api_keys_response.get('Item', {}).get('api_key', {}).get('M', {})
            api_keys = {k: v['S'] for k, v in api_keys.items()}
            composio_api = api_keys.get('composio')
        except Exception as e:
            print(f"Warning: Could not get Composio API key: {e}")
            composio_api = None

        # Create single folder for form if it has file upload questions
        questions = [q.dict() for q in form_draft.questions]
        form_folder_id = None

        # Check if form has any file upload questions
        has_file_questions = any(q.get('type') == 'file' for q in questions)

        if has_file_questions and composio_api:
            # Create one folder for the entire form
            form_folder_id = create_form_folder(form_draft.title, composio_api)

        # Create form data structure
        form_data = {
            'title': form_draft.title,
            'description': form_draft.description,
            'question': questions,
            'drive_folder_id': form_folder_id,  # Store folder at form level
            'user_id': user_id,
            'status': 'draft',
            'submissions': 0,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'updated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'active': True,
            'workflows': [],
            'spreadsheet_url': None,
            'public_url': None,
            'dashboard': None
        }

        # If form_id is provided, update existing form
        if form_draft.form_id:
            form_data['form_id'] = form_draft.form_id
            # Only update the updated_at timestamp for existing forms
            form_data.pop('created_at')
        else:
            # Create new form with new ID
            form_data['form_id'] = str(uuid.uuid4())

        table.put_item(Item=form_data)

        return {
            'success': True,
            'form_id': form_data['form_id'],
            'message': 'Draft saved successfully',
            'status': 'draft'
        }

    except Exception as e:
        print(f"Error saving draft: {e}")
        return {'error': f'Failed to save draft: {str(e)}'}


@app.get("/get_form/{form_id}")
async def get_form_by_id(form_id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get complete form data by ID for editing"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    try:
        if not DYNAMODB_AVAILABLE:
            return {'error': 'Database not available'}

        table = dynamodb.Table('forms')
        response = table.get_item(Key={'form_id': form_id})
        form_item = response.get('Item')

        if not form_item:
            return {'error': 'Form not found'}

        # Format response to match frontend expectations
        return {
            'form_id': form_item.get('form_id'),
            'title': form_item.get('title'),
            'description': form_item.get('description'),
            'dashboard': form_item.get('dashboard'),
            'public_url': form_item.get('public_url'),
            'spreadsheet_url': form_item.get('spreadsheet_url'),
            'status': form_item.get('status', 'draft'),
            'questions': form_item.get('question', []),
            'created_at': form_item.get('created_at'),
            'updated_at': form_item.get('updated_at'),
            'submissions': form_item.get('submissions', 0)
        }

    except Exception as e:
        print(f"Error getting form: {e}")
        return {'error': f'Failed to get form: {str(e)}'}


@app.get('/get_user_forms')
async def get_user_forms(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    try:
        if not DYNAMODB_AVAILABLE:
            return {'error': 'Database not available'}

        table = dynamodb.Table('forms')

        response = table.scan(
            FilterExpression='user_id = :user_id',
            ExpressionAttributeValues={':user_id': user_id}
        )

        forms = response.get('Items', [])

        # Format forms for frontend
        formatted_forms = []
        for form in forms:
            formatted_forms.append({
                'id': form.get('form_id'),
                'title': form.get('title'),
                'description': form.get('description'),
                'createdAt': form.get('created_at'),
                'updatedAt': form.get('updated_at'),
                'status': form.get('status', 'draft'),
                'isActive': form.get('active', True),
                'responses': form.get('submissions', 0),
                'isConnectedToSheet': bool(form.get('spreadsheet_url')),
                'spreadsheetUrl': form.get('spreadsheet_url'),
                'publicUrl': form.get('public_url'),
                'dashboard': form.get('dashboard'),
                'connectedWorkflows': form.get('workflows', [])
            })

        return {'forms': formatted_forms}

    except Exception as e:
        print(f"Error getting user forms: {e}")
        return {'error': f'Failed to get forms: {str(e)}'}


def get_user_email_from_clerk(user_id: str) -> str:
    """Get user email from Clerk using direct API call"""
    try:
        headers = {
            'Authorization': f'Bearer {clerk_secret_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            f'https://api.clerk.com/v1/users/{user_id}',
            headers=headers
        )
        
        if response.status_code == 200:
            user_data = response.json()
            email_addresses = user_data.get('email_addresses', [])
            if email_addresses:
                return email_addresses[0].get('email_address', 'unknown@example.com')
        
        print(f"Failed to get user email: {response.status_code} - {response.text}")
        return 'unknown@example.com'
        
    except Exception as e:
        print(f"Error getting user email from Clerk: {e}")
        return 'unknown@example.com'


@app.post("/submit")
async def submit_form(submission: FormSubmissionData, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Submit form responses to Google Sheets"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    try:
        if not DYNAMODB_AVAILABLE:
            return {'error': 'Database not available'}

        # Get user email using direct API call
        email = get_user_email_from_clerk(user_id)

        # Get form data
        table = dynamodb.Table('forms')
        response = table.get_item(Key={'form_id': submission.form_id})
        form_item = response.get('Item')

        if not form_item:
            return {'error': 'Form not found'}

        made_user_id = form_item.get('user_id')

        # Check if form has a spreadsheet
        spreadsheet_url = form_item.get('spreadsheet_url')
        if not spreadsheet_url:
            # For forms without spreadsheets, just acknowledge the submission
            table.update_item(
                Key={'form_id': submission.form_id},
                UpdateExpression='SET submissions = if_not_exists(submissions, :zero) + :one',
                ExpressionAttributeValues={':zero': 0, ':one': 1}
            )
            return {'response': True, 'success': True, 'message': 'Response submitted successfully (no spreadsheet connected)'}

        # Extract spreadsheet ID
        spreadsheet_id = get_id_from_url(spreadsheet_url)
        if not spreadsheet_id:
            return {'error': 'Invalid spreadsheet URL. Please check the spreadsheet connection in form settings.'}

        api_keys_response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': made_user_id}}
        )
        api_keys = api_keys_response.get('Item', {}).get('api_key', {}).get('M', {})
        api_keys = {k: v['S'] for k, v in api_keys.items()}

        # Initialize Composio
        composio_api = api_keys.get('composio')

        if not check_connection("googlesheets", composio_api):
            return {'error': 'No active Google Sheets account found'}

        composio = Composio(api_key=composio_api)

        # Get connected account
        connected_accounts = composio.connected_accounts.get()
        google_sheets_accounts = [acc for acc in connected_accounts if acc.appName == 'googlesheets' and acc.status == 'ACTIVE']

        if not google_sheets_accounts:
            raise Exception("No active Google Sheets account found")

        connected_account_id = google_sheets_accounts[0].id

        # Get current data to find next row
        try:
            rows = GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheet_id, sheet_name='Sheet1', api_key=composio_api)
            new_data_location = 'A' + str(len(rows['response']) + 2)
        except Exception as e:
            print(f"Warning: Could not get existing data, using row 2: {e}")
            new_data_location = 'A2'

        # Prepare submission data - extract just the answers
        response_values = []
        for i, response_item in enumerate(submission.responses):
            question = form_item.get('question', [])[i] if i < len(form_item.get('question', [])) else {}

            if question.get('type') == 'file':
                # For file uploads, create a clickable hyperlink in Google Sheets
                file_url = response_item.get('answer', '')
                if file_url:
                    # Use HYPERLINK formula for clickable links in Google Sheets
                    response_values.append(f'=HYPERLINK("{file_url}", "View File")')
                else:
                    response_values.append('')
            elif isinstance(response_item.get('answer'), list):
                # For checkbox responses, join multiple values
                response_values.append(', '.join(response_item.get('answer', [])))
            else:
                response_values.append(str(response_item.get('answer', '')))

        response_values.append(email)  # Add email at the end

        # Submit to Google Sheets
        params = {
            'spreadsheet_id': spreadsheet_id,
            'sheet_name': 'Sheet1',
            'first_cell_location': new_data_location,
            'values': [response_values],
            'includeValuesInResponse': False,
            'valueInputOption': 'USER_ENTERED'
        }

        submit_response = composio.actions.execute(
            action=Action.GOOGLESHEETS_BATCH_UPDATE,
            params=params,
            entity_id='default',
            connected_account=connected_account_id,
            session_id=str(uuid.uuid4())
        )

        if submit_response.get('successfull'):
            # Update form submission count
            table.update_item(
                Key={'form_id': submission.form_id},
                UpdateExpression='SET submissions = if_not_exists(submissions, :zero) + :one',
                ExpressionAttributeValues={':zero': 0, ':one': 1}
            )

            return {'response': True, 'success': True, 'message': 'Response submitted successfully'}
        else:
            return {'error': f'Failed to submit to spreadsheet: {submit_response}'}

    except Exception as e:
        print(f"ERROR: Error submitting form: {e}")
        return {'error': f'Failed to submit form: {str(e)}'}


@app.delete("/delete_form/{form_id}")
async def delete_form(form_id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Delete a form (only by the owner)"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    try:
        if not DYNAMODB_AVAILABLE:
            return {'error': 'Database not available'}

        table = dynamodb.Table('forms')

        # First check if form exists and user owns it
        response = table.get_item(Key={'form_id': form_id})
        form_item = response.get('Item')

        if not form_item:
            return {'error': 'Form not found'}

        if form_item.get('user_id') != user_id:
            return {'error': 'Unauthorized: You can only delete your own forms'}

        # Delete the form
        table.delete_item(Key={'form_id': form_id})

        return {
            'success': True,
            'message': 'Form deleted successfully',
            'form_id': form_id
        }

    except Exception as e:
        print(f"ERROR: Error deleting form: {e}")
        return {'error': f'Failed to delete form: {str(e)}'}


@app.post("/update_form_spreadsheet")
async def update_form_spreadsheet(form_id: str, spreadsheet_url: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Update form with spreadsheet URL"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    try:
        if not DYNAMODB_AVAILABLE:
            return {'error': 'Database not available'}

        table = dynamodb.Table('forms')

        # First check if form exists and user owns it
        response = table.get_item(Key={'form_id': form_id})
        form_item = response.get('Item')

        if not form_item:
            return {'error': 'Form not found'}

        if form_item.get('user_id') != user_id:
            return {'error': 'Unauthorized: You can only update your own forms'}

        # Update the form with spreadsheet URL
        table.update_item(
            Key={'form_id': form_id},
            UpdateExpression='SET spreadsheet_url = :url, updated_at = :timestamp',
            ExpressionAttributeValues={
                ':url': spreadsheet_url,
                ':timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

        return {
            'success': True,
            'message': 'Form updated with spreadsheet URL',
            'spreadsheet_url': spreadsheet_url
        }

    except Exception as e:
        print(f"ERROR: Error updating form spreadsheet: {e}")
        return {'error': f'Failed to update form: {str(e)}'}


@app.get("/get_form_responses/{form_id}")
async def get_form_responses(form_id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get form responses from Google Sheets"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    try:
        if not DYNAMODB_AVAILABLE:
            return {'error': 'Database not available'}

        # Get form data
        table = dynamodb.Table('forms')
        response = table.get_item(Key={'form_id': form_id})
        form_item = response.get('Item')

        if not form_item:
            return {'error': 'Form not found'}

        # Check if user owns the form
        if form_item.get('user_id') != user_id:
            return {'error': 'Unauthorized: You can only view responses for your own forms'}

        # Check if form has a spreadsheet
        spreadsheet_url = form_item.get('spreadsheet_url')
        if not spreadsheet_url:
            return {'responses': [], 'message': 'No spreadsheet connected to this form'}

        # Extract spreadsheet ID
        spreadsheet_id = get_id_from_url(spreadsheet_url)
        if not spreadsheet_id:
            return {'error': 'Invalid spreadsheet URL'}

        api_keys_response = db_client.get_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}}
        )
        api_keys = api_keys_response.get('Item', {}).get('api_key', {}).get('M', {})
        api_keys = {k: v['S'] for k, v in api_keys.items()}
        composio_api = api_keys.get('composio')

        if not check_connection("googlesheets", composio_api):
            return {'error': 'No active Google Sheets account found'}

        composio = Composio(api_key=composio_api)

        # Get connected account
        connected_accounts = composio.connected_accounts.get()
        google_sheets_accounts = [acc for acc in connected_accounts if acc.appName == 'googlesheets' and acc.status == 'ACTIVE']

        if not google_sheets_accounts:
            raise Exception("No active Google Sheets account found")

        connected_account_id = google_sheets_accounts[0].id

        batch_get_response = composio.actions.execute(
            action=Action.GOOGLESHEETS_BATCH_GET,
            params={'spreadsheet_id': spreadsheet_id, 'ranges': ['Sheet1']},
            entity_id='default',
            connected_account=connected_account_id,
            session_id=str(uuid.uuid4())
        )

        if not batch_get_response.get('successfull'):
            return {'error': 'Failed to fetch responses from spreadsheet'}

        spreadsheet_data = batch_get_response.get('data', {}).get('valueRanges', {}).get('values', [])

        if not spreadsheet_data:
            return {'responses': [], 'headers': [], 'message': 'No responses found'}

        # First row is headers, rest are responses
        headers = spreadsheet_data[0] if len(spreadsheet_data) > 0 else []
        responses = spreadsheet_data[1:] if len(spreadsheet_data) > 1 else []

        # Format responses
        formatted_responses = []
        for i, response_row in enumerate(responses):
            response_dict = {}
            for j, value in enumerate(response_row):
                header = headers[j] if j < len(headers) else f'Column {j+1}'
                response_dict[header] = value
            formatted_responses.append({
                'id': i + 1,
                'responses': response_dict,
                'submitted_at': 'Unknown'
            })

        return {
            'responses': formatted_responses,
            'headers': headers,
            'total_responses': len(formatted_responses),
            'form_title': form_item.get('title', 'Untitled Form')
        }

    except Exception as e:
        print(f"ERROR: Error getting form responses: {e}")
        return {'error': f'Failed to get form responses: {str(e)}'}




@app.get("/fetch_pricing")
async def fetch_pricing(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # Fetch the user's plan from DynamoDB
        response = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )
        
        if "Item" not in response:
            return {"user_plan": "free"}  # Default to free if user not found
        
        # Extract the plan from the user's data
        user_plan = response["Item"].get("plan", {}).get("S", "free")
        
        return {"user_plan": user_plan}
    
    except Exception as e:
        logger.error(f"Error fetching user plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching pricing information: {str(e)}")





@app.post("/activate_free_trial")
async def activate_free_trial(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # Update user's plan to free_15_pro
        response = db_client.update_item(
            TableName='users',
            Key={'clerk_id': {'S': user_id}},
            UpdateExpression="SET #plan_attr = :plan, trial_start_date = :start_date",
            ExpressionAttributeNames={
                '#plan_attr': 'plan'
            },
            ExpressionAttributeValues={
                ':plan': {'S': 'free_15_pro'},
                ':start_date': {'S': datetime.now().isoformat()}
            },
            ReturnValues="UPDATED_NEW"
        )
        
        logger.info(f"Updated plan for user {user_id} to free_15_pro trial")
        return {"status": "success", "message": "Plan updated to free_15_pro trial"}
    
    except Exception as e:
        logger.error(f"Error updating plan for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating plan: {str(e)}")


@app.post("/enterprise_interested")
async def get_user_email(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Fetch the authenticated user's email address using Clerk"""
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get('sub')
    
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    
    try:
        # Use Clerk SDK to get user details
        clerk_sdk = Clerk(bearer_auth=clerk_secret_key)
        user_details = clerk_sdk.users.list(user_id=[user_id])[0]
        
        # Extract email from user details
        if user_details.email_addresses and len(user_details.email_addresses) > 0:
            email = user_details.email_addresses[0].email_address
            ComposioToolSet(api_key="").execute_action(
                action="GMAIL_SEND_EMAIL",
                params={"recipient_email": "", "subject":"SIGMOYD New enterprise contact request", "body": f"Hey Aryan, Please reach out to {email}. He wants whatsapp enterprise integration"}
            )

            ComposioToolSet(api_key="").execute_action(
                action="GMAIL_SEND_EMAIL",
                params={"recipient_email": "", "subject":"SIGMOYD New enterprise contact request", "body": f"Hey Aayush, Please reach out to {email}. He wants whatsapp enterprise integration"}
            )

            return {"status": "success"}
        else:
            return {"status": "error", "message": "No email address found for this user"}
            
    except Exception as e:
        logger.error(f"Error fetching user email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching user email: {str(e)}")



