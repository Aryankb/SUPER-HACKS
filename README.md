

# SIGMOYD: Prompt-to-Orchestrate Framework

**Live Demo Available**:  [link](https://drive.google.com/file/d/10_FIgGWq08ZGp7OWQ9nH5qajTVCirSlx/view?usp=drivesdk)
Watch how Sigmoyd searches the latest **LinkedIn job posts**, generates **cold emails**, and sends them instantly to recruiters — **fully automated**!

---
### Deployed Endpoint URL (Beta version releasing soon)

[link](https://www.sigmoyd.in/whatsapp)

---

## What is Sigmoyd?

Ever thought of building your own **AI personal assistant** just by writing a **prompt**?

**No code. No drag & drop. Just describe what you want.**

Sigmoyd makes it possible with its powerful **Prompt-to-Orchestrate** framework.
![Image](https://github.com/user-attachments/assets/18eb427c-1d24-43f2-9e3c-99a49a9d7046)

---

## Key Features

### Orchestration Framework  
- Connects any tools (aka *Functions*) with **LLMs** in between  
- In-built **knowledge base** of all tool inputs/outputs  
- tools such as **CSV_AI** , **RAG_RETRIEVER** , etc.
- Smart orchestration with **zero hallucination workflows**

### MCP Integration  
- **MCP Client Layer**: Discovers and includes tools from any MCP server  
- Add MCP tools to any orchestrated flow seamlessly

### Visual Workflow Execution  
- See **data flow**, **validations**, **iterations**, and **triggers**  
- Gain deep insights into how the workflow runs under the hood
- real time logs for workflow running on background

### Trigger & Control Workflows  
- **Auto-triggers**: On New Mail Recieved, API calls, cron jobs (periodic runs), etc.  
- **Manual run option**  
- Enable/disable workflows dynamically  

### Built-in OAuth Integration  
One-click login for:
- Gmail, whole google suite
- Notion
- Slack
- LinkedIn  
... and more!

### Create Dynamic Knowledge Base for RAG
- Create collections using pdf, csv, ppt,web links etc.
- delete or add files to the collection
- retrieve relevant information from the collection using retrieval tool.
- use retrieval tool in between your workflows

---

## Use Cases

- **Customer Support Email arrived**: Classify incoming emails, retrieve relevant context from collection, generate and send the reply . Route the comples queries to relevant slack channel or google sheet.
- **Linkedin job cold email automation**: Find job postings from linkedin posts, scrape jd , recruiter emails, generate a personalized cold email with context from resume for each scraped job, send the email to recruiter with resume.
- **Sales Data Analysis**: Upload CSV → Ask questions → Get insights  

**... and so much more.**  
The possibilities are **limitless** with MCP + Tool integration.


---
## Architecture Diagram
![Image](https://github.com/user-attachments/assets/9079aa9e-9b9d-4776-8089-513708b68094)
![Image](https://github.com/user-attachments/assets/e39fb632-1d31-446b-bf71-dd85f3275a1e)

---

**Follow Sigmoyd** — a robust prompt-to-orchestrate framework.  
Be part of the future of **Agentic AI** & **Workflow Automation**.

---




# Setup Instructions

## Create a New Environment
```bash
python -m venv env
```

## Activate the Environment
### Windows:
```bash
env\Scripts\activate
```
### macOS/Linux:
```bash
source env/bin/activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Navigate to Agentic Directory
```bash
cd Agentic
```

## Run Uvicorn Server
```bash
uvicorn create_agents:app --reload
```
