import streamlit as st
import psycopg2
import pandas as pd
import re
from groq import Groq

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

# ---------- Initialize Groq ----------
def init_groq():
    """Setup Groq API with error handling."""
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        return client
    except Exception as e:
        st.error(f"Failed to initialize Groq: {e}")
        return None

# ---------- Helper: Convert SQL to lowercase ----------
def fix_table_names(sql):
    """Convert CamelCase table names to lowercase."""
    # List of common Chinook tables
    tables = ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 
              'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 
              'PlaylistTrack', 'Track']
    
    for table in tables:
        sql = re.sub(rf'\b{table}\b', table.lower(), sql, flags=re.IGNORECASE)
    
    # Fix column names to lowercase too
    columns = ['TrackId', 'AlbumId', 'ArtistId', 'CustomerId', 
               'InvoiceId', 'GenreId', 'PlaylistId', 'Quantity', 'Total']
    
    for col in columns:
        sql = re.sub(rf'\b{col}\b', col.lower(), sql, flags=re.IGNORECASE)
    
    return sql

# ---------- LLM: Natural Language -> SQL ----------
def generate_sql(question, schema, conversation_history, client):
    """Ask Groq to generate a valid PostgreSQL query."""
    if client is None:
        return None
    
    history_text = ""
    for q, a in conversation_history[-5:]:
        history_text += f"User: {q}\nAssistant: {a}\n"
    
    prompt = f"""You are an expert at converting natural language questions into PostgreSQL queries.

IMPORTANT: This database uses LOWERCASE table and column names.
Use the EXACT names as shown below:

Schema:
{schema}

Previous conversation:
{history_text}

Example of correct SQL:
SELECT track.name, SUM(invoiceline.quantity) as total_sold
FROM invoiceline
JOIN track ON invoiceline.trackid = track.trackid
JOIN album ON track.albumid = album.albumid
GROUP BY track.name
ORDER BY total_sold DESC
LIMIT 5;

Rules:
- ONLY output the SQL query, no extra text, no markdown
- Use LOWERCASE table and column names
- Use LIMIT 10 unless specified otherwise

User question: {question}
SQL:"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        sql = response.choices[0].message.content.strip()
        
        # Cleanup markdown
        sql = re.sub(r'^```sql\n?', '', sql)
        sql = re.sub(r'^```\n?', '', sql)
        sql = re.sub(r'\n?```$', '', sql)
        
        # Convert to lowercase table names
        sql = fix_table_names(sql)
        
        return sql
    except Exception as e:
        st.error(f"Groq API error: {e}")
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

def correct_sql(question, sql, error_msg, schema, client):
    """Ask Groq to fix the SQL based on the error."""
    if client is None:
        return None
        
    prompt = f"""The following SQL query failed. Please fix it.

IMPORTANT: This database uses LOWERCASE table names like: invoiceline, track, album, customer.
Do NOT use CamelCase. Use lowercase only.

Question: {question}
Failed SQL: {sql}
Error: {error_msg}

Schema (use lowercase names):
{schema}

Rules:
- Use lowercase table and column names
- Output only the corrected SQL

Corrected SQL:"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        new_sql = response.choices[0].message.content.strip()
        new_sql = re.sub(r'^```sql\n?', '', new_sql)
        new_sql = re.sub(r'^```\n?', '', new_sql)
        new_sql = re.sub(r'\n?```$', '', new_sql)
        new_sql = fix_table_names(new_sql)
        return new_sql
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return None

# ---------- Result to English ----------
def result_to_english(question, sql, result_df, client):
    """Convert the query result into a plain English answer."""
    if client is None or result_df.empty:
        return "No results found." if result_df.empty else "AI not available."
    
    result_str = result_df.head(10).to_string()
    
    prompt = f"""Question: {question}
Query result:
{result_str}

Answer the question in one clear English sentence.
Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return f"Results:\n{result_str}"

# ---------- Main App ----------
def main():
    st.set_page_config(page_title="AI SQL Agent", layout="wide")
    st.title("🗄️ Natural Language SQL Agent")
    st.info("🤖 Using Groq Llama 3 (100% Free)")
    
    # Initialize
    if "groq_client" not in st.session_state:
        st.session_state.groq_client = init_groq()
        if st.session_state.groq_client is None:
            st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conn" not in st.session_state:
        try:
            st.session_state.conn = get_db_connection()
            st.session_state.schema = get_table_schema(st.session_state.conn)
            st.success("✅ Connected to database!")
            
            with st.sidebar:
                st.write("📋 Database Tables:")
                cur = st.session_state.conn.cursor()
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;")
                for table in cur.fetchall():
                    st.write(f"  - {table[0]}")
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
                              st.session_state.messages, st.session_state.groq_client)
            
            if sql:
                st.caption(f"🔍 SQL: `{sql}`")
                success, result = execute_sql(st.session_state.conn, sql)
                
                attempts = 1
                while not success and attempts < 3:
                    st.warning(f"Error: {result}. Fixing...")
                    sql = correct_sql(user_question, sql, result, 
                                     st.session_state.schema, st.session_state.groq_client)
                    if sql:
                        st.caption(f"🛠️ Fixed SQL: `{sql}`")
                        success, result = execute_sql(st.session_state.conn, sql)
                    attempts += 1
                
                if success and isinstance(result, pd.DataFrame):
                    answer = result_to_english(user_question, sql, result, st.session_state.groq_client)
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