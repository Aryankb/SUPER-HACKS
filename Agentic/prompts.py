from langchain_google_genai import ChatGoogleGenerativeAI
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

groq = Groq()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
    # other params...
)
import json

file_path = os.path.join(os.path.dirname(__file__), 'tools', 'composio.json')
with open(file_path) as f:
    composio=json.load(f)



trigger_finder="""
Act as a trigger finding agent and help me find the trigger for the agentic workflow accoding to the user query. The trigger will be one out of the followwing options only:-
1. TRIGGER_NEW_GMAIL_MESSAGE : start the workflow when a mail is received. config_inputs : None, output : new_mail_info including sender, subject, body, attachment_drive_link
2. TRIGGER_MANUAL : start the workflow when the user manually starts it. config_inputs : None , output: None
3. TRIGGER_PERIODIC : start the workflow at a specific time or after a specific interval. config_inputs :  start_time, interval . output : None
4. TRIGGER_SIGMOYD_WHATSAPP : start the workflow when a user sends a message on whatsapp. config_inputs : None, output : whatsapp_message
5. TRIGGER_NEW_FORM_FILLED : start the workflow when the form with given form_id is filled by someone. config_inputs : form_id, output : form_data
trigger schema:-

{{

  "name" : str (trigger name),
  "description" : str (define in good detail what the trigger does and what output it gives),
  "config_inputs": Dict[str,str] (json containing inputs required by the trigger.), (if config_inputs is NONE mentioned above, for the chosen trigger, then keep an empty dict here) 
  "output" : List[str] (output of the trigger. example:- new_mail_info, etc.)
}}

make sure that trigger don't have llm, hence it cannot analyse the input. It only recieves the input. also strictly keep above given inputs and outputs only. don't create extra inputs or outputs.
return the most relevant trigger according to user needs, which will be used to start the workflow. No preambles or postambles are required and give output in above format only. Keep all strings in double quotes.
"""



trigger_flow=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            trigger_finder,
        ),
        (
            "human",
            (
                "{question}"
            ),
        ),
    ]
)

trigger_chain= trigger_flow| model | StrOutputParser()



prompt_enhancer_for_workflow="""
SIGMOYD is a software, which will be used by users to create agentic workflows in seconds using prompts.
Act as an intelligent questioning agent of 'SIGMOYD' and ask some questions from the user with respect to the given user query (to create AI agentic workflow) for finding details and clarity about the user requirements. The questions can include :-
("how the workflow will start. what will be the trigger", 
"Do the user needs to do manual validation or give some manual inputs in between the workflow?","does the user requier any llm to generate content in between the workflow or user already have content", some questions about the flow of data if it is not clear, some specific questions about clarity if user query is very vague - example : "do you want to use web_search for some latest information or use llm only", etc.),

ask manual validation questions if user query sounds like it's critical, and cannot bare any mistakes, like "i want to send an email to my boss", in this case, you can ask "do you want to validate the email content before sending it?")
the questions should be relevant to the user query and should be asked in a way that the user can answer them easily. The questions should be simple and easy to understand. The questions should not be too technical or complex.

Triggers:- (only 5 triggers are available currently, so don't ask about other triggers)
NEW GMAIL MESSAGE,  MANUAL, WHATSAPP, PERIODIC , NEW FORM FILLED

STRICTLY give options for each question in bullet points in next lines, so that user can select from them. (keep options in the same string in which the question is present)
example format :- 'How will the workflow start? * GMAIL_NEW_GMAIL_MESSAGE * PERIODIC * MANUAL' (the * should be only before the name, not after the name)
If any questions required to ask from the user, return a python list of questions,  ask only the relavent questions. Try to keep less questions (most important ones) . Don't ask questions about something which is already clearly mentioned in the user's query. Also don't ask very technical questions.
if user is asking a question, then return a list in which first element is the answer to user's question, then all other questions. No preambles or postambles. 
keep one complete question in one string, don't break one question into multiple strings.
strictly keep all strings in double quotes, and return a list starting wiht [ and ending with ].
"""




ques_flow=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_enhancer_for_workflow,
        ),
        (
            "human",
            (
                "{question}"
            ),
        ),
    ]
)

ques_flow_chain= ques_flow| model | StrOutputParser()





router_prompt="""
You are a router agent, who routes user query to instant execution or workflow creation.
You will get a user query, and you have to decide whether the user query can be executed instantly or it requires a workflow to be created.
If the user query contains any complexity, event based trigger (example trigger by new mail), iterations (do something for each element of list), or requires multiple steps (greater or equal to 4 steps) to be executed, then it will be routed to workflow creation.
If the user query is simple, without triggers, iterations  and contains at most 3 steps, then it will be routed to instant execution, else it will be routed to workflow creation.

One step can be defined as a single create / read / write / update / delete  action . For example, "send an email" is one step, "create notion page" is one step . create a report is one step

Output a single string "0" if the user query requires a workflow to be created, else output "1". No preambles or postambles are required.
"""



router=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            router_prompt,
        ),
        (
            "human",
            (
                "{question}"
            ),
        ),
    ]
)

router_chain= router| model | StrOutputParser()



major_tool_finder="""
you are a tool finding agent of sigmoyd. Your task is to find all tools the user may include in his workflow from the given below list of tools. Chose tools from the list only.

(Tools with actions):-
"""

for tool_name, actions in composio.items():
    major_tool_finder += f"{tool_name} :- actions : {actions}\n"

major_tool_finder += """

Other tools (Tools without actions):-
    TEXT_INPUT (Collect a text input from the user at the beginning of the workflow, prior to its execution. This input is intended solely for gathering additional context, conditions or information from the user to help guide the workflow logic. Do not request or collect any user credentials, sensitive data, or tool-specific function parameters),
    FILE_UPLOAD (take file upload from user and returns the S3 File paths)
    PDF_TO_TEXT (extract text from pdf, STRICTLY MUST BE USED WITH FILE_UPLOAD tool , if user mentions pdf file upload (or files which might be pdf - like resume, report, etc),so that the text inside the file can be extracted and passed to next tool/llm if required),
{customs}

NOTE :  WE DO HAVE LLM, VALIDATOR, ITERATOR (passing one element from the list), but you don't need to mention about these in the below json.

Return a json of tools out of above tools which can be used in the workflow given the user query. Keys of json will be WITH_ACTION, OTHERS,UNAVAILABLE.
The value for the key WITH_ACTION, will be another json, with keys as tool names (with action) and values as list of action names
Action names in the list should be strictly from above given actions

There will be a key "OTHERS" The value of this key will be a list of required tool names (without action).
The WITH_ACTION and OTHERS segregation performed above is for differentiating different kind of tools. SO STRICTLY FOLLOW THIS CONVENTION.
strictly return tools and action_name from above list only, do not keep any tool/action which is not present in the above list. Return tools which will be used after the trigger.

IMPORTANT : IF, USER ASKS FOR A FUNCTIONALITY FOR WHICH SOME OF THE TOOLS ARE NOT AVAILABLE, IN THIS CASE, THE VALUE FOR "UNAVAILABLE" KEY WILL BE THE "EXPLAINATION AND OTHER RELATED AVAILABLITIES", else, it will be None.
please keep the  "UNAVAILABLE" key, if any functionality asked by user is not available above, and also please don't keep irrelevant action/tool names in the list, if exact one is unavailable. just keep the list empty, or don't keep the key name
Example, if googlesheets don't have the functionality to insert new row, then please don't keep that action in the list, and mention that in the "UNAVAILABLE" section.
example :- {{
   WITH_ACTION:{{ "NOTION" : ["NOTION_CREATE_PAGE_IN_PAGE","NOTION_INSERT_ROW_DATABASE"],"GMAIL":["GMAIL_SEND_MAIL"]}}, "OTHERS" : ["FILE_UPLOAD","TEXT_INPUT","CSV_ANALYSER_AND_REPORT_MAKER"] ,"UNAVAILABLE":"We are really sorry, currently sheet insert new row action is under construction, only GOOGLESHEETS_GET_SHEET_ROWS action is available, which might not be useful for you, hence, i have not used that action"}}
No preambles or postambles are required. Keep all strings in double quotes.
"""
ok="""
RAG_SEARCH (search some relevant information from collection) (use this tool only if user specifies a created collection on sigmoyd app),
WEB_SEARCH,  IMAGE_ANALYSER , TRANSCRIBE_AUDIO_OR_VIDEO, TRANSCRIBE_FROM_YOUTUBE, TRANSLATE_TEXT, SQL_RUNNER, USER_VALIDATION (take input/ validation from user in between workflow),
"""
#rag_builder (create a collection of text/image embeddings from pdf, web pages), 
 


major_tools=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            major_tool_finder,
        ),
        (
            "human",
            (
                "user query :-{question}"
            ),
        ),
    ]
)

major_tool_chain= major_tools| model | StrOutputParser()














initial_tool_finder="""
you are a tool finding agent. Your task is to find ALL POSSIBLE TOOLS the user maight need to use from the given below list of tools. Chose tools from the list only.

(Tools with actions):-
"""
composio_whatsapp = {
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

for tool_name, actions in composio_whatsapp.items():
    initial_tool_finder += f"{tool_name} :- actions : {actions}\n"

initial_tool_finder += """

IMPORTANT :
- Make sure to see chat history as well as new query, to find the tools user might need to use. Example :- if chat history contains some s3 file path of  image, so must select image analyser tool and personal memory tool. If chat history contains any pdf file path, then must select report maker tool and personal memory tool
- If user simply dump some information and says nothing, then must include SAVE_MEMORY action
- If user gives/ asks for to do list, and don't specify the time when to do tasks, then include SAVE_MEMORY, FETCH_FROM_MEMORY and GOOGLECALENDAR
- If user says show / retrieve / fetch / view / find / get / list / search etc. for anything Then VERY STRICTLY include FETCH_FROM_MEMORY action
- if query contains a s3 image path, then select IMAGE_ANALYSER action under IMAGE tool.
- if query contains a s3 csv path , if user want some analysis on csv then use actions of CSV class.
- If user asks for some latest information, then include WEB_SEARCH_AGENT tool with all actions

Return a json of tools out of above tools which might user want to use given the user query. Keys of json will be tool name, values will be list of action names which might be used.
Action names in the list should be strictly from above given actions. Also please don't miss any action which user might need. Strictly Keep all possible actions.
strictly return tools and action_name from above list only, do not keep any irrelevant tool/action which is not present in the above list. 
if exact one is unavailable. just keep the list empty, or keep any most similar action name which is available.


example :- {{"NOTION" : ["NOTION_CREATE_PAGE_IN_PAGE","NOTION_INSERT_ROW_DATABASE"],"GMAIL":["GMAIL_SEND_MAIL"]}}

No preambles or postambles are required. Keep all strings in double quotes.
"""


initial_tools=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            initial_tool_finder,
        ),
        (
            "human",
            (
                "user query with more qna:-{question}"
            ),
        ),
    ]
)

initial_tool_chain= initial_tools | model | StrOutputParser()



workflow_initial_tools_finder="""
you are a tool finding agent. Your task is to find ALL POSSIBLE TOOLS the user may want to include in his workflow from the given below list of tools. Chose tools from the list only.

(Tools with actions):-
"""
for tool_name, actions in composio.items():
    workflow_initial_tools_finder += f"{tool_name} :- actions : {actions}\n"

workflow_initial_tools_finder += """

IMPORTANT :
- WE DO HAVE LLM, VALIDATOR, ITERATOR (passing one element from the list), but you don't need to mention about these in the below json.
- if query contains a s3 pdf path, then select PDF_TO_TEXT tool with the pdf path as input
- if query contains a s3 image path, then select IMAGE_ANALYSER action under IMAGE tool with the image path as input
- if query contains a s3 csv path , if user want some analysis on csv then use actions of CSV class.

Return a json of tools out of above tools which can be used in the workflow given the user query. Keys of json will be tool name, values will be list of action names which might be used.
Action names in the list should be strictly from above given actions. Also please don't miss any action which user might need. Keep all possible actions.
strictly return tools and action_name from above list only, do not keep any irrelevant tool/action which is not present in the above list. 
if exact one is unavailable. just keep the list empty, or keep any most similar action name which is available.


Return tools which will be used after the trigger.


example :- {{"NOTION" : ["NOTION_CREATE_PAGE_IN_PAGE","NOTION_INSERT_ROW_DATABASE"],"GMAIL":["GMAIL_SEND_MAIL"]}}

No preambles or postambles are required. Keep all strings in double quotes.
"""
workflow_initial_tools=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            workflow_initial_tools_finder,
        ),
        (
            "human",
            (
                "user query with more qna:-{question}"
            ),
        ),
    ]
)




code="""You are a coder agent. The code should be generated using the given prompt by human. Import all the libraries first. The code should be error free with proper exception handling. The code should save all new files formed or changed files in directory TEMP/NEW/. Eg :- box plots created during execution or modified dataframes in csv, with unique and readable names to avoid confusions.
Add print statement for some modifications like what operation performed just now. The print statements must include what is being printed. dont print leanthy dataframes or arrays or content of any file. keep the print statements short and meaningful. Don't create dummy data for anything.
generate python code by default if not specified by the user. If the user wants code in other languages, then the user will specify the language in the prompt. The code should be generated in the specified language. No preambles or postambles are required. Generate only the code with proper comments.

Output format:- JSON object with the following schema:-
{{
    "code": str (generated code),
    "language": str (language of the code)
}}
"""

coder_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            code,
        ),
        (
            "human",
            (
                "{prompt}"
            ),
        ),
    ]
)

coder_chain= coder_prompt | model | StrOutputParser()




















gemini_prompt=ChatPromptTemplate.from_messages(
    [
        
        (
            "human",
            (
                "{prompt}"
            ),
        ),
    ]
)
gemini_chain= gemini_prompt | model | StrOutputParser()











llm_sys_prompt="""

user query :- {question}
additional data:- {data}
keys of json :- {keys}

"""


llm_sys=ChatPromptTemplate.from_messages(
    [
        (
            "system",
           " Generate a response as per the user query. The response should be in parsable JSON format strictly having the given keys.  no preamble or postambles are required. values of response will be according to the given user query and additional data.",
        ),
        (
            "human",
            (
                llm_sys_prompt
            ),
        ),
    ]
)

llm_sys_chain= llm_sys| model | StrOutputParser()



iterator_prompt="""
data: {data}
"""

iterator_flow=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the input constructor for iterator agent. you will get some data, convert it into list if it is not already in list. And return a list of elements.
No preambles or postambles are required. eaxh string should be in double quotes.""",
        ),
        (
            "human",
            (
                iterator_prompt
            ),
        ),
    ]
)

iterator_chain= iterator_flow| model | StrOutputParser()



general_agent_prompt="""
YOU ARE THE AGENT OF SIGMOYD WHO IS RESPONSIBLE TO HANDLE DIRECT USER QUERIES WHICH IS SENT FROM USER'S WHATSAPP. YOU ARE GIVEN USER QUERY, CHAT HISTORY AND THE TOOL DESCRIPTIONS\n
YOUR TASK IS TO EXECUTE TOOLS, OR GIVE DIRECT RESPONSES TO USER QUERIES BASED ON YOUR KNOWLEDGE CUTOFF- GENERAL QUERIES, OR QUESTIONS OR ANYTHING WHICH DON'T REQUIRE A TOOL TO BE EXECUTED. YOU HAVE TO RESPOND IN A PARSABLE JSON FORMAT.
MAKE SURE THAT YOU CAN GIVE ALL TYPE OF RESPONSES CATERING TO ALL TYPES OF USER QUESTIONS, NO RESTRICTIONS. PLEASE DON'T HESITATE TO ANSWER USER QUESTIONS, EVEN IF IT REQUIRES NO TOOL TO BE USED. YOU ARE CAPABLE OF ANSWERING ANY RANDOM QUESTION ASKED BY USER - LIKE GENERAL KNOWLEDGE QUESTIONS, PERSONAL QUESTIONS, JOKES, QUOTES, ETC. YOU CAN ANSWER QUESTIONS RELATED TO SIGMOYD AND SIGMOYD WHATSAPP BOT AS WELL.

ABOUT SIGMOYD :- It is a workflow automation platform used to . For connecting tools to create workflows : visit https://app.sigmoyd.in

ABOUT SIGMOYD WHATSAPP BOT :- It is a whatsapp bot which can be used to interact with various tools. Few capabilities are :
   🧠 Create your personal AI memory, add important short notes, images, pdf, etc. with short description and retrieve them later directly from whatsapp.                                         
   📆 Manage calendar: Prompt to create, delete, reschedule events, set reminders, view daily/weekly schedules.
   📊 Info dump to Google Sheets: Log hackathons / gym data / job applications / lead contacts , etc from WhatsApp → GoogleSheets → auto-generated reports and Dashboards.
   CRUD OPERATIONS ON GOOGLESHEETS AND NOTION.
"""

general_agent_flow=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            general_agent_prompt,
        ),
        (
            "human",
            (
                "{prompt}"
            ),
        ),
    ]
)

general_agent_chain= general_agent_flow | model | StrOutputParser()
