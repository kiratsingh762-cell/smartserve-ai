import streamlit as st
import requests
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SmartServe AI",
    page_icon="[AI]",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
.header-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    color: white; padding: 25px; border-radius: 12px;
    text-align: center; margin-bottom: 25px;
}
.order-box {
    background: #d4edda; border-left: 4px solid #28a745;
    padding: 12px 15px; border-radius: 6px; margin: 8px 0;
    font-size: 14px;
}
.escalate-box {
    background: #fff3cd; border-left: 4px solid #ffc107;
    padding: 12px 15px; border-radius: 6px; margin: 8px 0;
    font-size: 14px;
}
.source-tag {
    background: #e9ecef; color: #495057; padding: 2px 8px;
    border-radius: 10px; font-size: 12px; margin: 2px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h2>SmartServe AI</h2>
    <p>Intelligent Customer Support | RAG + AS400 Powered</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### How to Use")
    st.markdown("""
    **General Support:**
    Ask any product or technical question.

    **Order Status:**
    Mention an Order ID like `ORD00042`
    to get live status from AS400.

    **Human Agent:**
    Say *"connect me to an agent"*
    for human escalation.
    """)

    st.markdown("---")
    st.markdown("### System Status")
    try:
        health = requests.get("http://localhost:8000/health", timeout=3).json()
        st.success(f"API: {health.get('api','unknown')}")
        st.info(f"Vectors: {health.get('vectors', 'N/A'):,}")
        st.info(f"AS400: {health.get('as400','N/A')}")
    except:
        st.error("API Offline — start the server")
        st.code("uvicorn src.api.main:app --reload --port 8000")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ── Chat History Init ─────────────────────────────────────────
# CONCEPT - st.session_state:
# Streamlit reruns the ENTIRE script on every user interaction.
# session_state is a dictionary that PERSISTS between reruns.
# Without it, chat history would be wiped on every message.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm SmartServe AI, your intelligent support assistant.\n\n"
                "I can help you with:\n"
                "- Product & technical issues\n"
                "- Order status (mention your Order ID)\n"
                "- Account & billing queries\n"
                "- General support questions\n\n"
                "How can I assist you today?"
            ),
            "as400_data": {},
            "escalate": False,
            "sources": []
        }
    ]

# ── Render Chat History ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Show AS400 order data if present
        if msg.get("as400_data") and msg["as400_data"].get("found"):
            d = msg["as400_data"]
            st.markdown(f"""
            <div class="order-box">
            <b>Live Order Data (AS400):</b><br>
            Order: <b>{d.get('order_id')}</b> |
            Status: <b>{d.get('status')}</b><br>
            Product: {d.get('product')} |
            Tracking: {d.get('tracking')}<br>
            Amount: ${d.get('amount')} |
            Date: {d.get('order_date')}
            </div>
            """, unsafe_allow_html=True)

        # Show escalation notice
        if msg.get("escalate"):
            st.markdown("""
            <div class="escalate-box">
            <b>Connecting you to a human agent...</b>
            A support representative will be with you shortly.
            </div>
            """, unsafe_allow_html=True)

        # Show sources
        if msg.get("sources"):
            for s in msg["sources"]:
                st.markdown(f'<span class="source-tag">{s}</span>',
                            unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input("Type your question here..."):

    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={"message": prompt, "session_id": "streamlit"},
                    timeout=30
                )
                data = response.json()

                # Display the answer
                st.write(data["answer"])

                # Display AS400 order data
                if data.get("as400_data") and data["as400_data"].get("found"):
                    d = data["as400_data"]
                    st.markdown(f"""
                    <div class="order-box">
                    <b>Live Order Data (AS400):</b><br>
                    Order: <b>{d.get('order_id')}</b> |
                    Status: <b>{d.get('status')}</b><br>
                    Product: {d.get('product')} |
                    Tracking: {d.get('tracking')}<br>
                    Amount: ${d.get('amount')} |
                    Date: {d.get('order_date')}
                    </div>
                    """, unsafe_allow_html=True)

                # Display escalation notice
                if data.get("escalate_to_human"):
                    st.markdown("""
                    <div class="escalate-box">
                    <b>Connecting you to a human agent...</b>
                    </div>
                    """, unsafe_allow_html=True)

                # Display source tags
                if data.get("sources"):
                    for s in data["sources"]:
                        st.markdown(f'<span class="source-tag">{s}</span>',
                                    unsafe_allow_html=True)

                # Save to history
                st.session_state.messages.append({
                    "role":      "assistant",
                    "content":   data["answer"],
                    "as400_data": data.get("as400_data", {}),
                    "escalate":  data.get("escalate_to_human", False),
                    "sources":   data.get("sources", [])
                })

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Run: uvicorn src.api.main:app --reload --port 8000")
            except Exception as e:
                st.error(f"Error: {str(e)}")