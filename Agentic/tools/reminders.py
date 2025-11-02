from pydantic import BaseModel
from uuid import uuid4
from typing import Optional
from tools.dynamo import db_client



class Reminder(BaseModel):
    user_id: str
    start_time: str  # ISO format string
    message: str
    repeat_interval_hours: Optional[int] = None



def get_number_from_user_id(user_id: str) -> str:   
    """
    Placeholder function to get the user's WhatsApp number from their user ID.
    In a real application, this would query a database or service to retrieve the number.
    """
    try:
        response = db_client.get_item(
            TableName="users",
            Key={"clerk_id": {"S": user_id}}
        )
        if "Item" in response and "whatsapp_verified" in response["Item"]:
            print(f"Retrieved WhatsApp number for user {user_id}: {response['Item']['whatsapp_verified']['S']}")
            return str(response["Item"]["whatsapp_verified"]["S"])
        else:
            return None
    except Exception as e:
        print(f"Error retrieving WhatsApp number for user {user_id}: {e}")
        return None

def create_reminder(reminder: Reminder):
    reminder_id = str(uuid4())[:8]
    number = get_number_from_user_id(reminder.user_id)
    if not number:
        number = "917000978867"  # Default number if not found
    item = {
        "reminder_id": {"S": reminder_id},
        "user_id": {"S": reminder.user_id},
        "start_time": {"S": reminder.start_time},
        "message": {"S": reminder.message},
        "number": {"S": number},
        "status": {"S": "active"}
    }
    
    if reminder.repeat_interval_hours is not None:
        item["repeat_interval_hours"] = {"N": str(reminder.repeat_interval_hours)}

    db_client.put_item(
        TableName="reminders",
        Item=item
    )
    return {"reminder_id": reminder_id, "status": "created"}


def delete_reminder(reminder_id: str):
    try:
        db_client.delete_item(
            TableName="reminders",
            Key={"reminder_id": {"S": reminder_id}}
        )
    except Exception as e:
        print(f"Error deleting reminder {reminder_id}: {e}")
        
    return {"reminder_id": reminder_id, "status": "deleted"}
