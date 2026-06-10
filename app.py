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
        # Gunakan model yang benar - 'gemini-1.5-flash' atau 'gemini-pro'
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        return None

# ---------- LLM: Natural Language -> SQL (Gemini) ----------
def generate_sql(question, schema, conversation_history, model):
    """Ask Gemini to generate a valid PostgreSQL query."""
    if model is None:
        return None
    
    # Build prompt with conversation memory
    prompt = f"""
You are an expert at converting natural language questions into **PostgreSQL** queries.
Use the following database schema (Chinook sample DB):

{schema}

Rules:
- Only output the SQL query, no extra text, no markdown formatting.
- Use double quotes for table/column names if they contain mixed case.
- Always use LIMIT 10 unless the question asks for a specific number.
- For questions about "top X", use ORDER BY and LIMIT X.

Previous conversation (for context):
"""
    # Add conversation history
    for q, a in conversation_history[-5:]:
        prompt += f"\nUser: {q}\nAssistant: {a}\n"
    
    prompt += f"\nUser question: {question}\nSQL query:"
    
    try:
        response = model.generate_content(prompt)
        sql = response.text.strip()
        
        # Cleanup markdown code fences if present
        sql = re.sub(r'^```sql\n?', '', sql)
        sql = re.sub(r'^```\n?', '', sql)
        sql = re.sub(r'\n?```$', '', sql)
        
        return sql
    except Exception as e:
        st.error(f"Gemini API error (generate_sql): {e}")
        return None

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

def correct_sql(question, sql, error_msg, schema, model):
    """Ask Gemini to fix the SQL based on the error."""
    if model is None:
        return None
        
    prompt = f"""
The following SQL query failed with an error.
Question: {question}
Failed SQL: {sql}
Error: {error_msg}
Schema: {schema}

Please provide a corrected PostgreSQL query. Output only the SQL, no explanation.
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
        st.error(f"Gemini API error (correct_sql): {e}")
        return None

# ---------- LLM: Convert SQL Result to English ----------
def result_to_english(question, sql, result_df, model):
    """Convert the query result into a plain English answer."""
    if model is None:
        return "AI model not available. Here's the raw data:"
    
    if result_df.empty:
        return "No results found for your query."
    
    # Convert DataFrame to string (first 10 rows)
    result_str = result_df.head(10).to_string()
    
    prompt = f"""
Question: {question}
SQL used: {sql}
Query result (first rows):
{result_str}

Please answer the original question in one clear, concise English sentence.
If the result shows numbers, summarise them naturally.
Answer:
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini API error (result_to_english): {e}")
        return f"Here are the results:\n\n{result_str}"

# ---------- Main Streamlit App ----------
def main():
    st.set_page_config(page_title="AI SQL Agent", layout="wide")
    st.title("🗄️ Natural Language SQL Agent")
    st.markdown("Ask questions about the **Chinook** music store database (tracks, artists, customers, invoices, etc.)")
    st.info("🤖 Using **Google Gemini 1.5 Flash** (free tier) - 60 requests per minute")
    
    # Initialize Gemini
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = init_gemini()
        if st.session_state.gemini_model is None:
            st.error("Failed to initialize Gemini. Please check your API key in Secrets.")
            st.stop()
    
    # Initialize session memory
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of (user_msg, assistant_msg)
    if "conn" not in st.session_state:
        try:
            st.session_state.conn = get_db_connection()
            st.session_state.schema = get_table_schema(st.session_state.conn)
            st.success("✅ Connected to PostgreSQL database!")
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
    user_question = st.chat_input("Ask something like: Show me the top 5 best-selling tracks by quantity")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.spinner("Generating SQL using Gemini..."):
            # 1. Generate initial SQL
            sql = generate_sql(user_question, st.session_state.schema, 
                              st.session_state.messages, st.session_state.gemini_model)
            
            if sql is None:
                answer = "Sorry, I couldn't generate SQL due to an API error. Please try again."
            else:
                st.caption(f"🔍 Generated SQL: `{sql}`")
                
                # 2. Execute with self-correction loop (max 2 attempts)
                success, result = execute_sql(st.session_state.conn, sql)
                attempts = 1
                while not success and attempts < 3:
                    st.warning(f"SQL error: {result}. Trying to fix... (attempt {attempts})")
                    sql = correct_sql(user_question, sql, result, 
                                     st.session_state.schema, st.session_state.gemini_model)
                    if sql is None:
                        break
                    st.caption(f"🛠️ Corrected SQL: `{sql}`")
                    success, result = execute_sql(st.session_state.conn, sql)
                    attempts += 1
                
                if not success or sql is None:
                    answer = f"I couldn't generate a working SQL query. Last error: {result if 'result' in locals() else 'API error'}"
                else:
                    # 3. Convert result to English
                    if isinstance(result, pd.DataFrame):
                        answer = result_to_english(user_question, sql, result, st.session_state.gemini_model)
                        # Show data table optionally
                        with st.expander("📊 View query result"):
                            st.dataframe(result)
                    else:
                        answer = result  # for non-SELECT commands
        
        with st.chat_message("assistant"):
            st.write(answer)
        
        # Save to memory
        st.session_state.messages.append((user_question, answer))
        
        # Keep only last 10 exchanges
        if len(st.session_state.messages) > 10:
            st.session_state.messages = st.session_state.messages[-10:]

if __name__ == "__main__":
    main()