from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import boto3
import tempfile
load_dotenv()


os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")


def TEXT_INPUT(text:str="Any input text user gives before runtime. it maybe a condition or any extra information user can give."):
    return text


def FILE_UPLOAD(local_file_paths="list of file s3 paths ( ends with uuid . extension) (it is actually list of aws s3 paths. ex: [s3://workflow-files-2709/your_file.pdf,...])"):
    # Check if input is a string starting with '[' and convert to list if needed
    if isinstance(local_file_paths, str) and local_file_paths.startswith('['):


        try:
            # Strip brackets and split by comma
            cleaned = local_file_paths.strip('[]').split(',')
            # Clean up each path
            local_file_paths = [path.strip().strip('"\'') for path in cleaned]
        except Exception as e:
            print(f"Error converting string to list: {e}")
    return local_file_paths


def PDF_TO_TEXT(pdf_paths= "list of pdf file s3 paths ( ends with uuid . extension) (it is actually list of aws s3 paths.  ex: [s3://workflow-files-2709/your_file.pdf,...]"):
    
    s3_client = boto3.client('s3',region_name='us-east-1')
    bucket_name = "workflow-files-2709"
    final = ""
    temp_files = []
    
    try:
        # Convert to list if string
        if isinstance(pdf_paths, str):
            if pdf_paths.startswith('['):
                pdf_paths = [p.strip().strip('"\'') for p in pdf_paths.strip('[]').split(',')]
            else:
                pdf_paths = [pdf_paths]
        
        for path in pdf_paths:
            # Extract the key (filename) from the path
            # key = path

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_files.append(temp_file.name)
            
            try:
                # Download from S3
                s3_parts = path.replace("s3://", "").split("/", 1)
                if len(s3_parts) == 2:
                    bucket_name, key = s3_parts
                    s3_client.head_object(Bucket=bucket_name, Key=key)
                    s3_client.download_file(bucket_name, key, temp_file.name)

                else:
                    raise ValueError("Invalid S3 path format for csv_path")
                
                # Process the PDF
                reader = PdfReader(temp_file.name)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                final += text + "\n"
            except:
                print(f"Error downloading or processing file from S3: {key}")
            finally:
                # Close the temp file
                temp_file.close()
                

        return final

    except Exception as e:
        print(f"Error processing PDF file: {e}")
        return None
    finally:
        # Clean up - delete all temporary files
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


