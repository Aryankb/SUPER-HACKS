import requests
import os
from dotenv import load_dotenv
from tools.dynamo import s3_client
import tempfile
import time
load_dotenv()
access_token = os.getenv("WHATSAPP_TOKEN")  # Ensure you have set this in your .env file
def download_whatsapp_media(media_id: str,mime_type:str, save_dir: str = "./media",) -> str:
    """
    Downloads WhatsApp media using the Graph API and returns the local file path.

    Args:
        media_id (str): The ID of the media file from WhatsApp.
        save_dir (str): Directory to save the file.

    Returns:
        str: Path to the downloaded media file.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Get the media URL
    url = f"https://graph.facebook.com/v22.0/{media_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    media_url = response.json().get("url")

    if not media_url:
        raise ValueError("Media URL not found in response")

    # Step 2: Download the media file
    media_response = requests.get(media_url, headers=headers)
    media_response.raise_for_status()
    if mime_type == "audio/ogg; codecs=opus":
        file_path = os.path.join(save_dir, f"{media_id}.ogg")
    elif mime_type == "audio/mpeg":
        file_path = os.path.join(save_dir, f"{media_id}.mp3")
    elif mime_type == "application/pdf":
        file_path = os.path.join(save_dir, f"{media_id}.pdf")
    elif mime_type == "image/jpeg":
        file_path = os.path.join(save_dir, f"{media_id}.jpg")
    elif mime_type == "image/png":
        file_path = os.path.join(save_dir, f"{media_id}.png")
    elif mime_type == "video/mp4":
        file_path = os.path.join(save_dir, f"{media_id}.mp4")
    elif mime_type == "text/csv" or mime_type == "text/comma-separated-values":
        file_path = os.path.join(save_dir, f"{media_id}.csv")
    elif mime_type == "text/plain":
        file_path = os.path.join(save_dir, f"{media_id}.txt")
    else:
        raise ValueError(f"Unsupported MIME type: {mime_type}")
    with open(file_path, "wb") as f:
        f.write(media_response.content)

    return file_path

def whatsapp_verify(username: str = "user", number: str = "917000978867", email: str = "Not Provided"):
    """
    Sends a verification message to a WhatsApp number.
    """
    url = "https://graph.facebook.com/v22.0/759/messages"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    print("username:", username, "number:", number, "email:", email)
    # for official, change en to en_US
    data = {
        "messaging_product": "whatsapp",
        "to": number,
        "template": {
            "name": "sigmoyd_verification",
            "language": {
                "code": "en"
            },
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": username or "user"},
                        {"type": "text", "text": email or "Not Provided"}
                    ]
                }
            ]
        },
        "type": "template"
    }

    response = requests.post(url, headers=headers, json=data)

    print("Status Code:", response.status_code)
    print("Response:", response.json())
    return response.json()


def send_whatsapp_typing_on(msg_id: str ):
    """
    Sends a 'typing on' indicator to a WhatsApp number using the WhatsApp Cloud API.
    """
    url = "https://graph.facebook.com/v22.0/759/messages"

    headers = {
        "Authorization": f"Bearer {access_token}",  
        "Content-Type": "application/json"
    }

    data = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": msg_id,
        "typing_indicator": {
            "type": "text"
        }
    }

    response = requests.post(url, headers=headers, json=data)

    print("Status Code:", response.status_code)
    print("Response:", response.json())
    return response.json()

def send_whatsapp_message(text:str,number:str="917000978867"):
    """
    Sends a WhatsApp message using the WhatsApp Cloud API.
    """
    # Replace with your WhatsApp Cloud API URL and access token

    url = "https://graph.facebook.com/v22.0/759/messages"


    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # text += "\n\nTo manage your google calendar, reminders, notion, gmail, etc. from whatsapp, Please visit: \nhttps://app.sigmoyd.in/manage-auths"

    data = {
        "messaging_product": "whatsapp",
        "to": number,
        "type": "text",
        # "template": {
        #     "name": "hello_world",
        #     "language": {
        #         "code": "en_US"
        #     }
        # }
        "text": {
            "body": text
        }
    }

    response = requests.post(url, headers=headers, json=data)

    print("Status Code:", response.status_code)
    print("Response:", response.json())
    return response.json()



def send_whatsapp_options( number: str = "917000978867"):
    url = "https://graph.facebook.com/v17.0/759/messages"


    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    data = {
        "messaging_product": "whatsapp",
        "to": number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": "Detected activated workflows."
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "1",
                            "title": "Execute Workflows"
                        }
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "2",
                            "title": "No, Execute Tools"
                        }
                    }
                ]
            }
        }
    }

    response = requests.post(url, headers=headers, json=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
    return response.json()

def send_whatsapp_media_path(number: str, file_path: str, media_type: str = "document", content_type: str = "application/pdf"):
    """
    Send media (image, pdf, video, etc.) via WhatsApp Cloud API using a direct URL.
    file_path must be publicly accessible (S3 presigned URL or public object).
    """
    url = f"https://graph.facebook.com/v22.0/759/messages"

    headers = {
        "Authorization": f"Bearer {access_token}",
        # "Content-Type": "application/pdf"
    }

    # First upload the file to WhatsApp
    upload_url = f"https://graph.facebook.com/v22.0/753957661138769/media"
    files = {"file": (file_path, open(file_path, "rb"), content_type)}
    data = {"messaging_product": "whatsapp"}
    upload_resp = requests.post(upload_url, headers=headers, files=files, data=data).json()
    print("Upload Response:", upload_resp)

    media_id = upload_resp.get("id")
    if not media_id:
        print("Upload failed:", upload_resp)
        return upload_resp

    # Send the uploaded media
    payload = {
        "messaging_product": "whatsapp",
        "to": number,
        "type": media_type,
        media_type: {
            "id": media_id
        }
    }

    resp = requests.post(url, headers=headers, json=payload)
    print("Status:", resp.status_code)
    print("Response:", resp.json())
    return resp.json()







# Create a mapping of file extensions to content types

CONTENT_TYPE_MAP = {
    # Images
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
    
    # Documents
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.txt': 'text/plain',
    '.csv': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    
    # Audio
    '.aac': 'audio/aac',
    '.mp3': 'audio/mpeg',
    '.m4a': 'audio/mp4',
    '.amr': 'audio/amr',
    '.ogg': 'audio/ogg',
    '.opus': 'audio/opus',
    
    # Video
    '.mp4': 'video/mp4',
    '.3gp': 'video/3gpp'
}

def get_content_type(file_path):
    """Determine content type from file extension"""
    ext = os.path.splitext(file_path.lower())[1]
    return CONTENT_TYPE_MAP.get(ext, 'application/octet-stream')  # Default to binary if unknown

# Define mapping of file extensions to WhatsApp media types
MEDIA_TYPE_MAP = {
    # Images
    '.jpg': 'image',
    '.jpeg': 'image',
    '.png': 'image',
    '.webp': 'image',
    
    # Documents
    '.pdf': 'document',
    '.doc': 'document',
    '.docx': 'document',
    '.ppt': 'document',
    '.pptx': 'document',
    '.xls': 'document',
    '.xlsx': 'document',
    '.txt': 'document',
    
    # Audio
    '.aac': 'document',
    '.mp3': 'document',
    '.m4a': 'document',
    '.amr': 'document',
    '.ogg': 'document',
    '.opus': 'document',
    
    # Video
    '.mp4': 'video',
    '.3gp': 'video'
}

def get_media_type(file_path):
    """Determine WhatsApp media type from file extension"""
    ext = os.path.splitext(file_path.lower())[1]
    print("EXT:", ext)
    return MEDIA_TYPE_MAP.get(ext, 'document')  # Default to document if unknown
    
def s3_to_whatsapp(path :str, phone_number: str = "917000978867"):
    # path = "s3://gmail-attachments-2709/user_2rm0Z6dEHfkkTZbBGeiV7x5PCxW/1201863891713792.jpg_d03f0b0d-c063-455b-a30c-e39eba27d16c"  # Example S3 path
    if path.startswith("s3://"):
        s3_parts = path.replace("s3://", "").split("/", 1)
        if len(s3_parts) == 2:
            bucket_name, key = s3_parts
            try:
                # Create a temporary file path
                
                # Create temp file with appropriate extension
                file_extension = os.path.splitext(key)[1]
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
                temp_file_path = temp_file.name
                temp_file.close()
                
                # Download the file from S3 to the temp file
                s3_client.download_file(
                    Bucket=bucket_name,
                    Key=key,
                    Filename=temp_file_path
                )
                
                # Get content type based on the file extension
                content_type = get_content_type(key)
                
                # Get media type based on the file extension
                media_type = get_media_type(key)
                
                # Use the local file path instead of a URL
                file_path = temp_file_path
                send_whatsapp_media_path(number=phone_number, file_path=file_path, media_type=media_type, content_type=content_type)
            except Exception as e:
                file_path = None
                print(f"Error generating presigned URL: {str(e)}")
            if file_path:
                print(f"File Path: {file_path}")
                print(f"Content Type: {content_type}")
                print(f"Media Type: {media_type}")
            else:
                print("Failed to generate presigned URL")
        else:
            print("Invalid S3 path format")
    else:
        print("Path does not start with s3://")




if __name__ == "__main__":

    # whatsapp_verify("user", "917000978867", "user@example.com")
    # s3_to_whatsapp("s3://personal-memory-2709/user_2rm0Z6dEHfkkTZbBGeiV7x5PCxW/0d139565-0a62-4dad-b5a9-152bacd3507b_1991713194911573.jpg")
    send_whatsapp_typing_on("917000978867")
    time.sleep(10)
    send_whatsapp_message("Hello from Sigmoyd! To manage your google calendar, reminders, notion, gmail, etc. from whatsapp, Please visit: \nhttps://app.sigmoyd.in/manage-auths", "917000978867")
