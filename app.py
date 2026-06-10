import streamlit as st
import psycopg2
import pandas as pd
from openai import OpenAI
import re

# ---------- Database Connection ----------
def get_db_connection():
    """Returns a PostgreSQL connection using secrets."""
    return psycopg2.connect(
        host=st.secrets["POSTGRES_HOST"],
        port=st.secrets["POSTGRES_PORT"],
        database=st.secrets["POSTGRES_DB"],
        user=st.secrets["POSTGRES_USER"],
        password=st.secrets["POSTGRES_PASSWORD"]
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

# ---------- LLM: Natural Language -> SQL ----------
def generate_sql(question, schema, conversation_history, model="gpt-3.5-turbo"):
    """Ask OpenAI to generate a valid PostgreSQL query."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Build prompt with conversation memory
    messages = [
        {"role": "system", "content": f"""
You are an expert at converting natural language questions into **PostgreSQL** queries.
Use the following database schema (Chinook sample DB):
{schema}

Rules:
- Only output the SQL query, no extra text.
- Use double quotes for table/column names if they contain mixed case.
- Return only the SQL, no markdown formatting.
- If the question is ambiguous, make reasonable assumptions.
- Always use LIMIT 10 unless the question asks for a specific number.
"""}
    ]
    # Add previous conversation for context
    for q, a in conversation_history[-5:]:  # keep last 5 Q&A
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f"Question: {question}\nSQL:"})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=300
    )
    sql = response.choices[0].message.content.strip()
    # Cleanup markdown code fences if present
    sql = re.sub(r'^```sql\n?', '', sql)
    sql = re.sub(r'\n?```$', '', sql)
    return sql

# ---------- SQL Execution & Self-Correction ----------
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

def correct_sql(question, sql, error_msg, schema, model="gpt-3.5-turbo"):
    """Ask OpenAI to fix the SQL based on the error."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = f"""
The following SQL query failed with an error.
Question: {question}
Failed SQL: {sql}
Error: {error_msg}
Schema: {schema}
Please provide a corrected PostgreSQL query. Output only the SQL.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )
    new_sql = response.choices[0].message.content.strip()
    new_sql = re.sub(r'^```sql\n?', '', new_sql)
    new_sql = re.sub(r'\n?```$', '', new_sql)
    return new_sql

# ---------- LLM: Convert SQL Result to English ----------
def result_to_english(question, sql, result_df, model="gpt-3.5-turbo"):
    """Convert the query result into a plain English answer."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    # If result is empty
    if result_df.empty:
        return "No results found for your query."
    # Convert DataFrame to string representation (first 10 rows)
    result_str = result_df.head(10).to_string()
    prompt = f"""
Question: {question}
SQL used: {sql}
Query result (first rows):
{result_str}
Please answer the original question in one clear, concise English sentence.
If the result shows numbers, summarise them naturally.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# ---------- Main Streamlit App ----------
def main():
    st.set_page_config(page_title="AI SQL Agent", layout="wide")
    st.title("🗄️ Natural Language SQL Agent")
    st.markdown("Ask questions about the **Chinook** music store database (tracks, artists, customers, invoices, etc.)")
    
    # Initialize session memory
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of (user_msg, assistant_msg)
    if "conn" not in st.session_state:
        try:
            st.session_state.conn = get_db_connection()
            st.session_state.schema = get_table_schema(st.session_state.conn)
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            st.stop()
    
    # Display chat history
    for user_q, assistant_a in st.session_state.messages:
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            st.write(assistant_a)
    
    # Input box
    user_question = st.chat_input("Ask something like: Who was our top sales agent last month?")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.spinner("Generating SQL..."):
            # 1. Generate initial SQL
            sql = generate_sql(user_question, st.session_state.schema, st.session_state.messages)
            st.caption(f"🔍 Generated SQL: `{sql}`")
            
            # 2. Execute with self-correction loop (max 2 attempts)
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
                # 3. Convert result to English
                if isinstance(result, pd.DataFrame):
                    answer = result_to_english(user_question, sql, result)
                    # Show data table optionally
                    with st.expander("📊 View query result"):
                        st.dataframe(result)
                else:
                    answer = result  # for non-SELECT commands
        
        with st.chat_message("assistant"):
            st.write(answer)
        
        # Save to memory
        st.session_state.messages.append((user_question, answer))
        
        # Optional: keep only last 10 exchanges
        if len(st.session_state.messages) > 10:
            st.session_state.messages = st.session_state.messages[-10:]

if __name__ == "__main__":
    main()