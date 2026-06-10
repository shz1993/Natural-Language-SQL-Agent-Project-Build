# Natural Language SQL Agent

Ask questions about a PostgreSQL database in plain English – AI converts them to SQL, executes, and explains results.

## Live Demo
[\[Link to your Streamlit app\]](https://shz1993-natural-language-sql-agent-project-build-app-r01ojk.streamlit.app/)

## Features
- Chat interface with memory
- Automatic SQL error correction
- Supports any PostgreSQL schema (tested with Chinook)
- Displays both the generated SQL and the results

## Tech Stack
- Streamlit (frontend)
- OpenAI GPT-3.5-turbo (LLM)
- PostgreSQL (Neon)
- Psycopg2, Pandas

## How to Run Locally
1. Clone repo
2. Install requirements: `pip install -r requirements.txt`
3. Create `.streamlit/secrets.toml` with your DB and OpenAI keys
4. Run `streamlit run app.py`

## Example Questions
- "Who is the top customer by total purchase?"
- "Show me all tracks in the 'Rock' genre"
- "Which sales agent sold the most in 2013?"
- Show me the top 5 best-selling tracks by quantity."
- Who is the customer who spent the most money?"
- List all rock tracks longer than 5 minutes."
- Which employee has the most customers?"
- What is the total sales per country?"