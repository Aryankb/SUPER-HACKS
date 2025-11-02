import boto3
from datetime import datetime, timedelta
from dateutil import parser  # safer than fromisoformat for +05:30
import pytz
import os
import requests
import asyncio
import aiohttp

os.environ['AWS_ACCESS_KEY_ID'] = ""
os.environ['AWS_SECRET_ACCESS_KEY'] = ""
access_token = ""
# Initialize DynamoDB client
db = boto3.client("dynamodb", region_name="us-east-1")  # adjust region



def run_reminder_scheduler():
    # now = datetime.now(pytz.utc)

    # Fetch all active reminders
    response = db.scan(
        TableName="reminders",
        FilterExpression="#st = :active",
        ExpressionAttributeNames={"#st": "status"},
        ExpressionAttributeValues={":active": {"S": "active"}}
    )

    async def send_whatsapp_message_async(text: str, number: str = "917000978867"):
        """Async version of send_whatsapp_message"""
        url = "https://graph.facebook.com/v22.0/79/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": number,
            "type": "text",
            "text": {"body": text}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                result = await response.json()
                print("Status Code:", response.status)
                print("Response:", result)
                return result

    async def process_reminder(item):
        try:
            # Parse the datetime with timezone
            start_time_raw = item["start_time"]["S"]
            start_time = parser.isoparse(start_time_raw)
            
            # Make sure we're comparing with the same timezone
            local_now = datetime.now(pytz.utc).astimezone(start_time.tzinfo)
            if start_time <= local_now:
                user_id = item["user_id"]["S"]
                reminder_id = item["reminder_id"]["S"]
                rem_type = item.get("reminder_type", {}).get("S", "default")
                if rem_type == "trigger":
                    # For trigger reminders, we might want to handle them differently
                    # For now, just print the reminder details
                    print(f"Trigger Reminder for User {user_id}: {item}")
                    # For trigger reminders, send a request to the start_workflow endpoint
                    workflow_id = reminder_id  # Assuming reminder_id is the workflow ID

                    # Prepare the request data
                    payload = {
                        "wid": workflow_id,
                        "user_id": user_id,
                        "key": "HAPPY"
                    }

                    # Send the request asynchronously
                    async with aiohttp.ClientSession() as session:
                        try:
                            url = "https://backend.sigmoyd.in/start_workflow"  # Adjust the URL as needed
                            async with session.post(url, json=payload) as response:
                                result = await response.json()
                                print(f"Workflow trigger response: {result}")
                                
                                
                        except Exception as e:
                            print(f"Error triggering workflow: {e}")
                    if "repeat_interval_hours" in item:
                        interval = int(item["repeat_interval_hours"]["N"])
                        new_time = start_time + timedelta(hours=interval)
                        await asyncio.to_thread(
                            db.update_item,
                            TableName="reminders",
                            Key={"reminder_id": {"S": reminder_id}},
                            UpdateExpression="SET start_time = :new",
                            ExpressionAttributeValues={":new": {"S": new_time.isoformat()}}
                        )
                    else:
                        await asyncio.to_thread(
                            db.delete_item,
                            TableName="reminders",
                            Key={"reminder_id": {"S": reminder_id}}
                        )


                else:
                    number = item.get("number", {}).get("S", "917000978867")
                    message = item["message"]["S"]

                    start_time = start_time + timedelta(minutes=5)
                    formatted_message = f"{message}\n\nThis is your scheduled reminder for {start_time.strftime('%B %d at %I:%M %p')}.\n\nHave a great day!"

                    await send_whatsapp_message_async(formatted_message, number)
                    if "repeat_interval_hours" in item:
                        interval = int(item["repeat_interval_hours"]["N"])
                        new_time = start_time + timedelta(hours=interval) - timedelta(minutes=5)
                        await asyncio.to_thread(
                            db.update_item,
                            TableName="reminders",
                            Key={"reminder_id": {"S": reminder_id}},
                            UpdateExpression="SET start_time = :new",
                            ExpressionAttributeValues={":new": {"S": new_time.isoformat()}}
                        )
                    else:
                        await asyncio.to_thread(
                            db.delete_item,
                            TableName="reminders",
                            Key={"reminder_id": {"S": reminder_id}}
                        )
        except Exception as e:
            print("Error processing reminder:", e)

    async def run_reminders_async():
        tasks = []
        for item in response["Items"]:
            tasks.append(process_reminder(item))
        await asyncio.gather(*tasks)

    # Replace the loop with this:
    asyncio.run(run_reminders_async())

if __name__ == "__main__":
    run_reminder_scheduler()
