import os
from dotenv import load_dotenv

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb
from agno.tools import tool


extracted_data = {}
# Output schema for structured data extraction
class EventDetails(BaseModel):
    """Details of a single life event/experience"""
    event_id: int
    event_number: int  # 1, 2, 3 for tracking multiple events
    event_overview: Optional[str] = None
    when_happened: Optional[str] = None
    what_happened: Optional[str] = None
    peak_moment: Optional[str] = None
    is_complete: bool = False


# Tools for data management and confirmation
@tool
def update_single_field(session_state, field_name: str, field_value: str) -> str:
    """Update a single field in the event details"""
    if "current_event" not in session_state:
        session_state["current_event"] = {}

    session_state["current_event"][field_name] = field_value
    return f"Updated {field_name}: {field_value}"


@tool
def update_multiple_fields(session_state, updates: Dict[str, str]) -> str:
    """Update multiple fields at once from user input"""
    if "current_event" not in session_state:
        session_state["current_event"] = {}

    for field_name, field_value in updates.items():
        session_state["current_event"][field_name] = field_value

    return f"Updated multiple fields: {list(updates.keys())}"


@tool
def confirm_data_from_user(session_state) -> Dict[str, Any]:
    """Show collected data to user for confirmation"""
    current_event = session_state.get("current_event", {})
    return {
        "status": "awaiting_confirmation",
        "collected_data": current_event,
        "message": "Please confirm if this information is correct"
    }


@tool
def confirm_completeness(session_state) -> Dict[str, Any]:
    """Check if all required data is collected and mark complete if ready"""
    current_event = session_state.get("current_event", {})
    required_fields = ["event_overview", "when_happened", "what_happened", "peak_moment"]
    missing_fields = []

    for field in required_fields:
        if field not in current_event or not current_event[field] or current_event[field].strip() == "":
            missing_fields.append(field)

    if not missing_fields:
        # Create final EventDetails object
        event_details = EventDetails(
            event_id=session_state.get("experience_no", 1),
            event_number=session_state.get("experience_no", 1),
            event_overview=current_event.get("event_overview"),
            when_happened=current_event.get("when_happened"),
            what_happened=current_event.get("what_happened"),
            peak_moment=current_event.get("peak_moment"),
            is_complete=True
        )

        session_state["completed_event"] = event_details.model_dump()
        session_state["session_complete"] = True  # ADD THIS LINE

        return {
            "status": "complete",
            "message": "All required data collected successfully!",
            "event_details": event_details.model_dump()
        }
    else:
        return {
            "status": "incomplete",
            "missing_fields": missing_fields,
            "message": f"Still need: {', '.join(missing_fields)}"
        }


# Create the coach agent
db = SqliteDb(db_file="tmp/coach_sessions.db")

coach_agent = Agent(
    model=OpenAIChat(id="gpt-4"),
    description="Empathetic self-analysis coach that gathers life experience data through supportive conversation",
    db=db,
    enable_user_memories=True,
    add_history_to_context=True,
    add_session_state_to_context=True,  # Enable session state in context
    tools=[update_single_field, update_multiple_fields, confirm_data_from_user, confirm_completeness],
    instructions=[
        "You are a warm, empathetic, and supportive self-analysis coach helping users explore their life experiences.",
        "IMPORTANT: Stay completely neutral and non-judgmental. Experiences can be positive, negative, or mixed - all are valid and valuable.",
        "Show genuine empathy and understanding. Use phrases like 'I understand', 'That sounds meaningful', 'Thank you for sharing'.",
        "Based on experience_no from session state, acknowledge previous work if > 1 (e.g., 'We've explored your first experience together, now let's look at your second one').",
        "If asked about previous data, gently say 'We've processed that information and it's safely stored. Let's focus on this current experience.'",
        "Gather these 4 key pieces ONE AT A TIME: What happened, Overview, When it happened, Peak moment.",
        "Ask follow-up questions with care and curiosity, not interrogation. Guide users gently if they're struggling.",
        "If user provides multiple pieces of info at once, acknowledge everything and extract what you can using update_multiple_fields.",
        "Use update_single_field for individual responses, update_multiple_fields when user gives multiple details.",
        "Always use confirm_data_from_user to show collected data before marking complete.",
        "Only use confirm_completeness after user confirms the data is correct.",
        "Start with friendly greeting using their name from session state: 'Hello {name}, I'm here to support you as we explore your experiences together. When you're ready, could you tell me about a life experience that comes to mind - something that has stayed with you?'",
        "For peak moments, ask gently: 'In this experience, was there a particular moment that felt most significant to you? Perhaps when you felt most engaged, or when something shifted, or when you realized something important? Take your time - there's no right or wrong answer.'",
        "Handle difficult moments with extra care: 'I can sense this might be challenging to talk about. Please share only what feels comfortable for you. We can take this at your own pace.'",
        "If user seems stuck or says 'I don't know', offer gentle prompts: 'Sometimes it helps to think about... Would any of these resonate with you?'",
        "If user wants to change previously given information, respond supportively: 'Of course, let's update that. It's completely normal to want to refine what you've shared.'",
        "Once all data confirmed, express gratitude: 'Thank you so much for trusting me with this experience. Let me show you what we've captured together.'",
        "Remember: You're creating a safe space for vulnerable sharing. Every response should feel supportive and non-judgmental."
    ]
)

# Initialize with profile data
profile_data = {
    "name": "Sarah",
    "age": 28,
    "current_occupation": "Marketing Manager",
    "desired_career": "Product Manager",
    "work_experience": "5 years in marketing, led 3 major campaigns",
    "experience_no": 1,
    "stage": "Gather information for experience_1",
    "current_event": {}  # Initialize empty event data
}

# Start the conversation with session state
print("=== Starting Empathetic Coach Session ===")
response = coach_agent.run(
    "Hello, I'm ready to start exploring my experiences",
    session_state=profile_data,
    user_id="sarah_user",
    session_id="experience_session_1"
)
print(f"Coach: {response.content}")

# Interactive chat loop with streaming
print("\n=== Interactive Chat (type 'quit' to exit) ===")
while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ['quit', 'exit', 'bye']:
        print(
            "Coach: Thank you for sharing your experiences with me today. It takes courage to explore these meaningful moments. Take care! 💙")
        break

    if user_input:
        print("Coach: ", end="", flush=True)
        for chunk in coach_agent.run(
                user_input,
                stream=True,
                user_id="sarah_user",
                session_id="experience_session_1"
        ):
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)

        print()

        # CHECK FOR COMPLETION
        # if coach_agent.session_state and coach_agent.session_state.get("session_complete"):
        #     print("\n" + "=" * 50)
        #     print("SESSION COMPLETE - EXTRACTED DATA:")
        #     print("=" * 50)
        #     completed_event = coach_agent.session_state.get("completed_event")
        #     if completed_event:
        #         # Create dictionary with only the required fields
        #         extracted_data = {
        #             "event_overview": completed_event.get("event_overview"),
        #             "when_happened": completed_event.get("when_happened"),
        #             "what_happened": completed_event.get("what_happened"),
        #             "peak_moment": completed_event.get("peak_moment")
        #         }
        #
        #     break


# At this point, the loop is finished. Now check for completed session:
completed_event = coach_agent.session_state.get("session_complete") if coach_agent.session_state else None

if completed_event:
    extracted_data = {
        "event_overview": completed_event.get("event_overview"),
        "when_happened": completed_event.get("when_happened"),
        "what_happened": completed_event.get("what_happened"),
        "peak_moment": completed_event.get("peak_moment")
    }

    print("\n=== EXTRACTED EVENT DATA ===")
    print(extracted_data)
    print("=" * 50)
else:
    print("\nNo completed event data was found.")