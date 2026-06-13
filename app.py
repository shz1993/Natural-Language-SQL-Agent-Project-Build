import streamlit as st
import psycopg2
import pandas as pd
import re
from groq import Groq

def get_db_connection():
    return psycopg2.connect(
        host=st.secrets["POSTGRES_HOST"],
        port=st.secrets["POSTGRES_PORT"],
        database=st.secrets["POSTGRES_DB"],
        user=st.secrets["POSTGRES_USER"],
        password=st.secrets["POSTGRES_PASSWORD"]
    )

def init_groq():
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception as e:
        st.error(f"Failed to initialize Groq: {e}")
        return None

def generate_sql(question, client):
    if client is None:
        return None
    
    prompt = f"""Convert to PostgreSQL SQL. ALL TABLES ARE LOWERCASE: track, invoiceline, album, artist.

Question: {question}

Example: SELECT track.name, SUM(invoiceline.quantity) as total_sold 
FROM invoiceline 
JOIN track ON invoiceline.trackid = track.trackid 
GROUP BY track.name 
ORDER BY total_sold DESC 
LIMIT 5;

Return ONLY SQL. No markdown. Use lowercase names.

SQL:"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        sql = response.choices[0].message.content.strip()
        sql = re.sub(r'^```sql\n?', '', sql)
        sql = re.sub(r'^```\n?', '', sql)
        sql = re.sub(r'\n?```$', '', sql)
        # Remove any quotes
        sql = sql.replace('"', '')
        return sql.lower()
    except Exception as e:
        st.error(f"Groq error: {e}")
        return None

def execute_sql(conn, sql):
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description] if cur.description else []
            df = pd.DataFrame(rows, columns=colnames)
            return True, df
    except Exception as e:
        return False, str(e)

def main():
    st.set_page_config(page_title="AI SQL Agent", layout="wide")
    st.title("🗄️ Natural Language SQL Agent")
    st.info("🤖 Using Groq Llama 3 (100% Free)")
    
    if "groq_client" not in st.session_state:
        st.session_state.groq_client = init_groq()
        if st.session_state.groq_client is None:
            st.stop()
    
    if "conn" not in st.session_state:
        try:
            st.session_state.conn = get_db_connection()
            st.success("✅ Connected to database!")
            
            with st.sidebar:
                st.write("📋 Database Tables (lowercase):")
                tables = ['track', 'invoiceline', 'album', 'artist', 'customer', 'genre', 'invoice']
                for table in tables:
                    st.write(f"  - {table}")
        except Exception as e:
            st.error(f"Database error: {e}")
            st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for user_q, assistant_a in st.session_state.messages:
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            st.write(assistant_a)
    
    user_question = st.chat_input("Ask something like: Show me the top 5 best-selling tracks")
    
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.spinner("Generating SQL..."):
            sql = generate_sql(user_question, st.session_state.groq_client)
            
            if sql:
                st.caption(f"🔍 SQL: `{sql}`")
                
                success, result = execute_sql(st.session_state.conn, sql)
                
                if success and isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                    with st.chat_message("assistant"):
                        st.write(f"✅ Found {len(result)} results")
                    st.session_state.messages.append((user_question, f"Found {len(result)} results"))
                elif success:
                    st.success("Query executed successfully")
                    st.session_state.messages.append((user_question, "Query executed successfully"))
                else:
                    st.error(f"SQL Error: {result}")
                    with st.chat_message("assistant"):
                        st.write(f"Error: {result}")
                    st.session_state.messages.append((user_question, f"Error: {result}"))
            else:
                st.error("Failed to generate SQL")

if __name__ == "__main__":
    main()