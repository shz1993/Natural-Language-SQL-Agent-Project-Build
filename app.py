import streamlit as st
import psycopg2
import pandas as pd
import google.generativeai as genai
import re

# ---------- Database Connection ----------
def get_db_connection():
    """Returns a PostgreSQL connection using secrets."""
    return psycopg2.connect(
        host=st.secrets["POSTGRES_HOST"],
        port=st.secrets["POSTGRES_PORT"],
        database=st.secrets["POSTGRES_DB"],
        user=st.secrets["POSTGRES_USER"],
        password=st.secrets["POSTGRES_PASSWORD"],
        connect_timeout=10
    )

# ---------- Schema Fetching ----------
def get_table_schema(conn):
    """Fetch all table names and their columns for Chinook database."""
    schema_info = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema='public'
            ORDER BY table_name;
        """)
        tables = cur.fetchall()
        for (table,) in tables:
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name=%s;
            """, (table,))
            cols = cur.fetchall()
            cols_str = ", ".join([f"{c[0]} {c[1]}" for c in cols])
            schema_info.append(f"Table: {table} ({cols_str})")
    return "\n".join(schema_info)

# ---------- LLM: Natural Language -> SQL (using Gemini) ----------
def generate_sql(question, schema, conversation_history):
    """Ask Gemini to generate a valid PostgreSQL query."""
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
You are an expert at converting natural language questions into **PostgreSQL** queries.
Use the following database schema (Chinook sample DB):
{schema}

Rules:
- Only output the SQL query, no extra text.
- Use double quotes for table/column names if they contain mixed case.
- Return only the SQL, no markdown formatting.
- If the question is ambiguous, make reasonable assumptions.
- Always use LIMIT 10 unless the question asks for a specific number.

Previous conversation (for context only, not to repeat):
{conversation_history[-3:] if conversation_history else 'None'}

Question: {question}
SQL:
"""
    response = model.generate_content(prompt)
    sql = response.text.strip()
    sql = re.sub(r'^```sql\n?', '', sql)
    sql = re.sub(r'\n?```$', '', sql)
    return sql

# ---------- SQL Execution ----------
def execute_sql(conn, sql):
    """Run SQL and return (success, result/error)."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            if sql.strip().upper().startswith("SELECT"):
                rows = cur.fetchall()
                colnames = [desc[0] for desc in cur.description]
                df = pd.DataFrame(rows, columns=colnames)
                return True, df
            else:
                conn.commit()
                return True, f"Query executed successfully. {cur.rowcount} rows affected."
    except Exception as e:
        return False, str(e)

def correct_sql(question, sql, error_msg, schema):
    """Ask Gemini to fix the SQL based on the error."""
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
The following SQL query failed with an error.
Question: {question}
Failed SQL: {sql}
Error: {error_msg}
Schema: {schema}
Please provide a corrected PostgreSQL query. Output only the SQL.
"""
    response = model.generate_content(prompt)
    new_sql = response.text.strip()
    new_sql = re.sub(r'^```sql\n?', '', new_sql)
    new_sql = re.sub(r'\n?```$', '', new_sql)
    return new_sql

# ---------- LLM: Convert SQL Result to English ----------
def result_to_english(question, sql, result_df):
    """Convert the query result into a plain English answer."""
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    if result_df.empty:
        return "No results found for your query."
    result_str = result_df.head(10).to_string()
    prompt = f"""
Question: {question}
SQL used: {sql}
Query result (first rows):
{result_str}
Please answer the original question in one clear, concise English sentence.
If the result shows numbers, summarise them naturally.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# ---------- Main Streamlit App ----------
def main():
    st.set_page_config(page_title="AI SQL Agent (Free)", layout="wide")
    st.title("🗄️ Natural Language SQL Agent")
    st.markdown("Ask questions about the **Chinook** music store database (tracks, artists, customers, invoices, etc.)")
    st.info("🤖 Using **Google Gemini 1.5 Flash** (free tier) - 60 requests per minute")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conn" not in st.session_state:
        try:
            st.session_state.conn = get_db_connection()
            st.session_state.schema = get_table_schema(st.session_state.conn)
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            st.stop()
    
    for user_q, assistant_a in st.session_state.messages:
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            st.write(assistant_a)
    
    user_question = st.chat_input("Ask something like: Show me the top 5 tracks by sales")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.spinner("Generating SQL..."):
            sql = generate_sql(user_question, st.session_state.schema, st.session_state.messages)
            st.caption(f"🔍 Generated SQL: `{sql}`")
            
            success, result = execute_sql(st.session_state.conn, sql)
            attempts = 1
            while not success and attempts < 3:
                st.warning(f"SQL error: {result}. Trying to fix... (attempt {attempts})")
                sql = correct_sql(user_question, sql, result, st.session_state.schema)
                st.caption(f"🛠️ Corrected SQL: `{sql}`")
                success, result = execute_sql(st.session_state.conn, sql)
                attempts += 1
            
            if not success:
                answer = f"I couldn't generate a working SQL query. Last error: {result}"
            else:
                if isinstance(result, pd.DataFrame):
                    answer = result_to_english(user_question, sql, result)
                    with st.expander("📊 View query result"):
                        st.dataframe(result)
                else:
                    answer = result
        
        with st.chat_message("assistant"):
            st.write(answer)
        
        st.session_state.messages.append((user_question, answer))
        
        if len(st.session_state.messages) > 10:
            st.session_state.messages = st.session_state.messages[-10:]

if __name__ == "__main__":
    main()