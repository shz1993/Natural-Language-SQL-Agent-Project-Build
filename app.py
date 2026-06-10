import streamlit as st
import psycopg2
import pandas as pd
import re
import google.generativeai as genai

# ---------- Database Connection ----------
def get_db_connection():
    """Returns a PostgreSQL connection using secrets."""
    return psycopg2.connect(
        host=st.secrets["POSTGRES_HOST"],
        port=st.secrets["POSTGRES_PORT"],
        database=st.secrets["POSTGRES_DB"],
        user=st.secrets["POSTGRES_USER"],
        password=st.secrets["POSTGRES_PASSWORD"]
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

# ---------- Initialize Gemini ----------
def init_gemini():
    """Setup Gemini API with error handling."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        return None

# ---------- LLM: Natural Language -> SQL ----------
def generate_sql(question, schema, conversation_history, model):
    """Ask Gemini to generate a valid PostgreSQL query."""
    if model is None:
        return None
    
    prompt = f"""
You are an expert at converting natural language questions into PostgreSQL queries.

IMPORTANT: This database uses table and column names in CamelCase (first letter capitalized).
Use the EXACT names as shown below - they are case-sensitive but work without quotes:

Schema:
{schema}

Example of correct SQL:
SELECT Track.Name, SUM(InvoiceLine.Quantity) as TotalSold
FROM InvoiceLine
JOIN Track ON InvoiceLine.TrackId = Track.TrackId
GROUP BY Track.Name
ORDER BY TotalSold DESC
LIMIT 5;

Rules:
- ONLY output the SQL query, no extra text, no markdown
- Use the table and column names EXACTLY as shown in the schema
- Do NOT use double quotes around names
- Use LIMIT 10 unless specified otherwise

User question: {question}
SQL:
"""
    
    try:
        response = model.generate_content(prompt)
        sql = response.text.strip()
        
        # Cleanup markdown
        sql = re.sub(r'^```sql\n?', '', sql)
        sql = re.sub(r'^```\n?', '', sql)
        sql = re.sub(r'\n?```$', '', sql)
        
        return sql
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None

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
                return True, f"Query executed successfully."
    except Exception as e:
        return False, str(e)

def correct_sql(question, sql, error_msg, schema, model):
    """Ask Gemini to fix the SQL based on the error."""
    if model is None:
        return None
        
    prompt = f"""
The following SQL query failed. Please fix it.

IMPORTANT: This database uses table names in CamelCase like: InvoiceLine, Track, Customer.
Do NOT use double quotes. Use the exact names from the schema.

Question: {question}
Failed SQL: {sql}
Error: {error_msg}

Schema (use these exact names):
{schema}

Rules:
- Use the exact table and column names from the schema (CamelCase)
- Do NOT add double quotes
- Output only the corrected SQL

Corrected SQL:
"""
    try:
        response = model.generate_content(prompt)
        new_sql = response.text.strip()
        new_sql = re.sub(r'^```sql\n?', '', new_sql)
        new_sql = re.sub(r'^```\n?', '', new_sql)
        new_sql = re.sub(r'\n?```$', '', new_sql)
        return new_sql
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None

# ---------- Result to English ----------
def result_to_english(question, sql, result_df, model):
    """Convert the query result into a plain English answer."""
    if model is None or result_df.empty:
        return "No results found." if result_df.empty else "AI not available."
    
    result_str = result_df.head(10).to_string()
    
    prompt = f"""
Question: {question}
Query result:
{result_str}

Answer the question in one clear English sentence.
Answer:
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return f"Results:\n{result_str}"

# ---------- Main App ----------
def main():
    st.set_page_config(page_title="AI SQL Agent", layout="wide")
    st.title("🗄️ Natural Language SQL Agent")
    st.info("🤖 Using Google Gemini 2.5 Flash")
    
    # Initialize
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = init_gemini()
        if st.session_state.gemini_model is None:
            st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conn" not in st.session_state:
        try:
            st.session_state.conn = get_db_connection()
            st.session_state.schema = get_table_schema(st.session_state.conn)
            st.success("✅ Connected to database!")
            
            # Show tables in sidebar for debugging
            with st.sidebar:
                st.write("📋 Database Tables:")
                for table in ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 
                              'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 
                              'PlaylistTrack', 'Track']:
                    st.write(f"  - {table}")
        except Exception as e:
            st.error(f"Database error: {e}")
            st.stop()
    
    # Chat history
    for user_q, assistant_a in st.session_state.messages:
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            st.write(assistant_a)
    
    # Input
    user_question = st.chat_input("Ask something like: Show me the top 5 best-selling tracks")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.spinner("Generating SQL..."):
            sql = generate_sql(user_question, st.session_state.schema, 
                              st.session_state.messages, st.session_state.gemini_model)
            
            if sql:
                st.caption(f"🔍 SQL: `{sql}`")
                success, result = execute_sql(st.session_state.conn, sql)
                
                # Self-correction
                attempts = 1
                while not success and attempts < 3:
                    st.warning(f"Error: {result}. Fixing...")
                    sql = correct_sql(user_question, sql, result, 
                                     st.session_state.schema, st.session_state.gemini_model)
                    if sql:
                        st.caption(f"🛠️ Fixed SQL: `{sql}`")
                        success, result = execute_sql(st.session_state.conn, sql)
                    attempts += 1
                
                if success and isinstance(result, pd.DataFrame):
                    answer = result_to_english(user_question, sql, result, st.session_state.gemini_model)
                    with st.expander("📊 Data"):
                        st.dataframe(result)
                elif success:
                    answer = str(result)
                else:
                    answer = f"Failed: {result}"
            else:
                answer = "Failed to generate SQL."
        
        with st.chat_message("assistant"):
            st.write(answer)
        
        st.session_state.messages.append((user_question, answer))
        
        if len(st.session_state.messages) > 10:
            st.session_state.messages = st.session_state.messages[-10:]

if __name__ == "__main__":
    main()