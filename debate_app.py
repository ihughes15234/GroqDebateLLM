import streamlit as st
from typing import Generator
from groq import Groq, BadRequestError

st.set_page_config(page_icon="🗣️", layout="wide", page_title="Groq LLM Debate")


def icon(emoji: str):
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🗣️")
st.subheader("Groq Multi-Agent Debate", divider="violet", anchor=False)

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- Model Definitions ---
models = {
    "llama-3.3-70b-versatile": {"name": "LLaMA3.3-70b-versatile", "tokens": 128000, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    "DeepSeek-R1-Distill-Llama-70B": {"name": "DeepSeek-R1-Distill-Llama-70B", "tokens": 128000, "developer": "DeepSeek"},
}
# --- Agent Definitions ---
initial_agent_instructions = {
    "agent1": {
        "instruction": "Only return Python code with no additional comments or explanations. You are an expert CAD drafter for FreeCad. Write Python code for the specified user design. If you get comments on how to improve the Python code, then implment it. Only return Python Code.",
        "avatar": "😀",
        "model": "llama-3.3-70b-versatile",  # Default model
    },
    "agent2": {
        "instruction": "You are an expert CAD drafter for FreeCad. Revise the given Python code for the specified user design. Please go through the code slowly and make sure designs are centered and follow the user design. State any assumptions explicitly.",
        "avatar": "🤔",
        "model": "DeepSeek-R1-Distill-Llama-70B",  # Default model
    },
}

# --- Session State ---
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []
if "current_agent_index" not in st.session_state:
    st.session_state.current_agent_index = 0
if "num_turns" not in st.session_state:
    st.session_state.num_turns = 3
if "current_turn" not in st.session_state:
    st.session_state.current_turn = 0
if "debate_running" not in st.session_state:
    st.session_state.debate_running = False
if "user_prompted" not in st.session_state:
    st.session_state.user_prompted = False

if "agent_instructions" not in st.session_state:
    st.session_state.agent_instructions = {
        key: value["instruction"] for key, value in initial_agent_instructions.items()
    }
if "agent_avatars" not in st.session_state:
    st.session_state.agent_avatars = {
        key: value["avatar"] for key, value in initial_agent_instructions.items()
    }
# Store agent models in session state
if "agent_models" not in st.session_state:
    st.session_state.agent_models = {
        key: value["model"] for key, value in initial_agent_instructions.items()
    }

# --- UI Elements ---
col1, col2 = st.columns(2)  # Reduced to two columns for model selection

with col1:
    st.subheader("Agent 1 Model")
    st.session_state.agent_models["agent1"] = st.selectbox(
        "Select model for Agent 1:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        key="model_agent1",  # Unique key
        index=list(models.keys()).index(st.session_state.agent_models["agent1"]) #set the default index
    )


with col2:
    st.subheader("Agent 2 Model")
    st.session_state.agent_models["agent2"] = st.selectbox(
        "Select model for Agent 2:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        key="model_agent2",  # Unique key
        index=list(models.keys()).index(st.session_state.agent_models["agent2"]) #set the default index
    )


max_tokens = st.slider(
    "Max Tokens:",
    min_value=512,
    max_value=128000,  # Set a reasonable max
    value=20000,  # Default value
    step=512,
    help="Maximum number of tokens for the response.",
)

num_turns = st.number_input(
    "Number of Debate Turns:", min_value=1, max_value = 5, value=st.session_state.num_turns, step=2
)
st.session_state.num_turns = num_turns

# --- Agent System Prompt Input ---
st.subheader("Agent System Prompts")
agent_cols = st.columns(len(initial_agent_instructions))

for i, agent_key in enumerate(initial_agent_instructions):
    with agent_cols[i]:
        agent_name = f"Agent {i+1}"
        st.session_state.agent_instructions[agent_key] = st.text_area(
            f"System prompt for {agent_name}:",
            value=st.session_state.agent_instructions[agent_key],
            key=f"prompt_{agent_key}",
            height=150,
        )

# --- Helper Functions ---
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def run_debate_turn():
    agent_keys = list(st.session_state.agent_instructions.keys())
    current_agent_key = agent_keys[st.session_state.current_agent_index]
    current_agent_instruction = st.session_state.agent_instructions[current_agent_key]
    current_agent_avatar = st.session_state.agent_avatars[current_agent_key]
    current_agent_model = st.session_state.agent_models[current_agent_key]  # Get model

    if st.session_state.user_prompted and st.session_state.current_turn == 0:
        messages = [{"role": "system", "content": current_agent_instruction}]
        messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.debate_history])
        st.session_state.user_prompted = False
    else:
        messages = []
        for msg in st.session_state.debate_history:
            if msg["role"] == "assistant":
                messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append({"role": msg["role"], "content": msg["content"]})

    agent_name = f"Agent {st.session_state.current_agent_index + 1}"
    temp_response = ""

    try:
        chat_completion = client.chat.completions.create(
            model=current_agent_model,  # Use the agent's selected model
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                temp_response += chunk.choices[0].delta.content

    except BadRequestError as e:
        if "context length" in str(e).lower():
            st.error(
                "The conversation history is too long. Reduce turns or choose a model with a larger context window.",
                icon="⚠️",
            )
        else:
            st.error(f"An error occurred: {e}", icon="🚨")
        st.session_state.debate_running = False
        return
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}", icon="🚨")
        st.session_state.debate_running = False
        return

    st.session_state.debate_history.append(
        {
            "role": "assistant",
            "content": temp_response,
            "agent": current_agent_key,
            "agent_name": agent_name,
            "avatar": current_agent_avatar,
        }
    )

    st.session_state.current_agent_index = (
        st.session_state.current_agent_index + 1
    ) % len(st.session_state.agent_instructions)
    st.session_state.current_turn += 1

    if st.session_state.current_turn >= st.session_state.num_turns:
        st.session_state.debate_running = False


# --- Main Debate Logic ---

if prompt := st.chat_input("Enter the initial topic for debate..."):
    st.session_state.debate_history.append(
        {"role": "user", "content": prompt, "agent": "user", "agent_name": "User", "avatar": "👨‍💻"}
    )
    st.session_state.user_prompted = True

    # Automatic Debate Loop (with delay)
    while st.session_state.current_turn < st.session_state.num_turns:
        run_debate_turn()


# --- Display Debate History ---
for message in st.session_state.debate_history:
    with st.chat_message(message["agent_name"], avatar=message["avatar"]):
        st.markdown(message["content"])

# --- "Reset Debate" Button: Always visible ---
if st.button("Reset Debate"):
    st.session_state.debate_history = []
    st.session_state.current_agent_index = 0
    st.session_state.current_turn = 0
    st.session_state.user_prompted = False
    st.session_state.agent_instructions = {
        key: initial_agent_instructions[key]["instruction"]
        for key in initial_agent_instructions
    }
    st.session_state.agent_models = {
        key: initial_agent_instructions[key]["model"]
        for key in initial_agent_instructions
    }