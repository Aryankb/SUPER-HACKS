from google_auth_oauthlib.flow import Flow
from dotenv import load_dotenv
import os
import requests
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from .dynamo import db_client
from boto3.dynamodb.types import TypeSerializer
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
load_dotenv()

serializer = TypeSerializer()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
flow = Flow.from_client_config(
    {
        "web": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uris": [REDIRECT_URI],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    },
    scopes=["https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/userinfo.email",
            "openid"]
)
flow.redirect_uri = REDIRECT_URI

def get_user_email(credentials):
    """Fetch the user's email address using their OAuth2 credentials."""
    url = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    headers = {
        "Authorization": f"Bearer {credentials.token}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        user_info = response.json()
        return user_info.get("email")  # ✅ Returns the email address
    else:
        print("Error fetching email:", response.json())
        return None


def save_google_credentials(user_id,mail, credentials):
    """Store credentials in the database"""
   

    # try:
    #     with open("/home/aryan/BABLU/Agentic/tools/credentials.json", "r") as file:
    #         data = json.load(file)
    # except (FileNotFoundError, json.JSONDecodeError):
    #     data = {}
    
    user_data = db_client.get_item(
        TableName="custom_credentials",
        Key={"mail": {"S": mail}}
    )
    
    gt_cred = user_data.get("Item", {}).get("gmailtrigger", {}).get("S", {}) 
    if not gt_cred:
        print("No credentials found for this user.")
        # print(type(credentials))
    gt_cred=credentials
    
    gt_cred_dynamo = serializer.serialize(gt_cred)
    try:
        response = db_client.update_item(
            TableName="custom_credentials",
            Key={'mail': {'S': mail}},
            UpdateExpression="SET gmailtrigger = :gt_cred, user_id = :user_id",
            ExpressionAttributeValues={
            ':gt_cred':  gt_cred_dynamo,
            ':user_id': {'S': user_id}
            },
            ReturnValues="UPDATED_NEW"
        )
        print("Update succeeded:", response)
    except Exception as e:
        print("Error updating item:", e)

    

    


def get_fresh_google_credentials(mail):
    
    user_data = db_client.get_item(
        TableName="custom_credentials",
        Key={"mail": {"S": mail}}
    )
    
    gt_cred = user_data.get("Item", {}).get("gmailtrigger", {}).get("S", {})
    user_id = user_data.get("Item", {}).get("user_id", {}).get("S", "")
    # with open("/home/aryan/BABLU/Agentic/tools/credentials.json", "r") as file:
    #     credentials = json.load(file)[user_id]
    if not gt_cred:
        raise HTTPException(status_code=401, detail="Gmail Trigger authentication required")
    else:
        credentials = gt_cred
    if type(credentials)==str:
        credentials=json.loads(credentials)
    
    # print(type(credentials))
    creds=Credentials(**credentials)
    # print(creds)
    # print(creds.valid)
    # if creds.expired and creds.refresh_token:
    creds.refresh(Request())  # Refresh the token
    save_google_credentials(user_id,mail, creds.to_json())
    return creds,user_id
    


def stop_gmail_watch(service, user_id="me"):
    try:
        response = service.users().stop(userId=user_id).execute()
        print("🛑 Gmail watch stopped successfully.")
    except Exception as e:
        print("❌ Failed to stop Gmail watch:", str(e))



def setup_watch(mail: str):
    creds ,user_id= get_fresh_google_credentials(mail)
    headers = {"Authorization": f"Bearer {creds.token}"}
    # service = build('gmail', 'v1', credentials=creds)   


    watch_request = {
        "labelIds": ["INBOX"],
        "topicName": "projects/sigmoyd-171/topics/sigmoyd-mail"
    }
    # response = service.users().watch(userId="me", body=watch_request).execute()
    

    response = requests.post(
        "https://www.googleapis.com/gmail/v1/users/me/watch",
        headers=headers,
        json=watch_request
    )

    if response.status_code != 200:
        print("❌ Watch setup failed:", response.status_code, response.text)
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    data= response.json()
    print("✅ Watch set successfully:", data)
    return data



import base64
import json
import binascii


def extract_email(email_data):
    email_data_json = json.dumps(email_data, indent=4)
    # print(email_data_json)
    # Extract sender email
    sender_email = next((h["value"] for h in email_data["payload"]["headers"] if h["name"] == "From"), "Unknown Sender")
    # Extract subject
    subject = next((h["value"] for h in email_data["payload"]["headers"] if h["name"] == "Subject"), "No Subject")

    body = ""  # Will store extracted email body
    attachments = {}  # Dictionary to store attachments

    def decode_base64(data):
        """ Decodes base64 safely, handling errors """
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        except (binascii.Error, UnicodeDecodeError):
            print("⚠️ Warning: Invalid Base64 data")
            return ""

    def process_parts(parts):
        """ Recursively processes parts of the email """
        nonlocal body

        for part in parts:
            mime_type = part.get("mimeType", "")
            
            # Handle email body
            if mime_type in ["text/plain", "text/html"]:
                if "data" in part.get("body", {}):
                    decoded_text = decode_base64(part["body"]["data"])
                    if mime_type == "text/plain":  # Prefer plain text over HTML
                        body += decoded_text + "\n"
                    elif mime_type == "text/html" and not body:  # Use HTML only if no plain text is available
                        body = decoded_text

            # Handle multipart (nested structure)
            elif mime_type.startswith("multipart/") and "parts" in part:
                process_parts(part["parts"])

            # Handle attachments
            if "filename" in part and part["filename"]:  # Attachment found

                # if "data" in part.get("body", {}):
                #     file_data = base64.urlsafe_b64decode(part["body"]["data"])
                #     attachments[part["filename"]] = file_data  # Store raw bytes

                if "attachmentId" in part.get("body", {}):
                    attachment_id = part["body"]["attachmentId"]
                    filename = part["filename"]
                    attachments[filename]=attachment_id  # Store attachment ID for later retrieval
                else:
                    print(f"⚠️ Warning: Attachment {part['filename']} has no attachmentId")
                    # Fetch attachment data using Gmail API (not implemented here)
                    # attachments[part["filename"]] = fetch_attachment_data(attachment_id)

    # Process the email payload
    if "data" in email_data["payload"]["body"]:  # Simple email (no attachments)
        body = decode_base64(email_data["payload"]["body"]["data"])
    else:
        process_parts(email_data["payload"].get("parts", []))

    # Print extracted email content
    print("📧 Sender Email:", sender_email)
    print("📌 Subject:", subject)
    print("📜 Body:", body[:500])  # Print only the first 500 chars for preview
    print("📎 Attachments:", list(attachments.keys()))

    # Save attachments to files
    # for filename, content in attachments.items():
    #     file_path = os.path.join(os.path.dirname(__file__), filename)
    #     with open(file_path, "wb") as f:
    #         f.write(content)
    #     print(f"✅ Saved attachment: {filename}")
    if "<html" in body or any(tag in body for tag in ["<div", "<p", "<body", "<table", "<span", "<a", "<ul", "<ol", "<li", "<h1", "<h2", "<h3", "<img", "<br", "<hr"]):
        soup = BeautifulSoup(body, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Get text
        body = soup.get_text(separator=' ', strip=True)
    return {
        "sender_email": sender_email,
        "subject": subject,
        "body": body,
        "attachments": attachments
    }



if __name__=="__main__":

    # save_google_credentials("me", {"token":"token", "refresh_token":"refresh_token
    # email_data={}
    # print(type(email_data))
    # # Convert email data to JSON with indent 4 and print
    # email_data_json = json.dumps(email_data, indent=4)
    # print(email_data_json)
    # extract_email(email_data)
    # Usage (assuming you already have authorized `service`)


   
    # service = build('gmail', 'v1', credentials=creds)
    # stop_gmail_watch(service)
