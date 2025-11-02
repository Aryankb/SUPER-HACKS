import boto3
from dotenv import load_dotenv  
import os
load_dotenv()
import tempfile


os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")

try:
    db_client = boto3.client("dynamodb",region_name='us-east-1')
    print("success dynamo")
except ConnectionError as e:
    raise ConnectionError(f"Failed to connect to DynamoDB: {str(e)}")

try:
    s3_client = boto3.client("s3", region_name='us-east-1')
    print("success s3")
except ConnectionError as e:
    raise ConnectionError(f"Failed to connect to S3: {str(e)}")

try:
    lambda_client = boto3.client("lambda", region_name='us-east-1')
    print("success lambda")
except ConnectionError as e:
    raise ConnectionError(f"Failed to connect to Lambda: {str(e)}")


# DynamoDB setup with error handling
try:
    dynamodb = boto3.resource(
        'dynamodb',
        region_name='us-east-1',
    )
    DYNAMODB_AVAILABLE = True
    print("SUCCESS: DynamoDB connected")
except Exception as e:
    print(f"Warning: DynamoDB not available: {e}")
    DYNAMODB_AVAILABLE = False
    dynamodb = None

if __name__ == "__main__":
    pass
   















    # # Test the connection to DynamoDB
    # try:
    #     response = db_client.list_tables()
    #     print("DynamoDB tables:", response['TableNames'])
    # except Exception as e:
    #     print(f"Error listing DynamoDB tables: {str(e)}")

    # # Test the connection to S3
    # try:
    #     response = s3_client.list_buckets()
    #     print("S3 buckets:", [bucket['Name'] for bucket in response['Buckets']])
    # except Exception as e:
    #     print(f"Error listing S3 buckets: {str(e)}")









    # Create an S3 bucket called 'personal-memory-2709'
    # try:
    #     bucket_name = 'personal-memory-2709'  # Add a unique suffix to avoid naming conflicts
    #     # us-east-1 is the default region and doesn't need LocationConstraint
    #     s3_client.create_bucket(
    #         Bucket=bucket_name,
    #     )
    #     print(f"Bucket '{bucket_name}' created successfully")
    # except Exception as e:
    #     print(f"Error creating bucket: {str(e)}")
        # If the bucket already exists, it will show in the error message
    









    # try:
    #     # Get all users from the 'users' table
    #     response = db_client.scan(TableName='users')
    #     users = response.get('Items', [])
        
    #     # Google API key from environment
    #     google_api_key = os.getenv("GOOGLE_API_KEY")
        
    #     if not google_api_key:
    #         print("Error: GOOGLE_API_KEY environment variable not set")
    #     else:
    #         # Iterate through each user
    #         for user in users:
    #             clerk_id = user.get('clerk_id', {}).get('S')
                
    #             if clerk_id:
    #                 try:
    #                     # Get the current api_key map
    #                     user_response = db_client.get_item(
    #                         TableName='users',
    #                         Key={'clerk_id': {'S': clerk_id}}
    #                     )
                        
    #                     # Extract the current api_keys
    #                     current_api_keys = user_response.get('Item', {}).get('api_key', {}).get('M', {})
                        
    #                     # Update the "gemini" key in the api_keys map
    #                     # Check if gemini already exists in api_keys
    #                     if 'gemini' not in current_api_keys:
    #                         update_response = db_client.update_item(
    #                             TableName='users',
    #                             Key={'clerk_id': {'S': clerk_id}},
    #                             UpdateExpression="SET api_key.#gemini = :val",
    #                             ExpressionAttributeNames={
    #                                 '#gemini': 'gemini'
    #                             },
    #                             ExpressionAttributeValues={
    #                                 ':val': {'S': google_api_key}
    #                             }
    #                         )
    #                         print(f"Added gemini API key for user {clerk_id}")
    #                     else:
    #                         print(f"User {clerk_id} already has a gemini API key - skipping")
    #                     print(f"Updated gemini API key for user {clerk_id}")
    #                 except Exception as e:
    #                     print(f"Error updating user {clerk_id}: {str(e)}")
            
    #         print("Finished updating all users with new gemini API key")
    # except Exception as e:
    #     print(f"Error scanning users table: {str(e)}")





