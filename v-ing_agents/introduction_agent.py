import os
from dotenv import load_dotenv

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(OPENAI_API_KEY)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

from typing import Optional
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb
from copy import deepcopy

profile_data_to_store_to_db = {}


# Define structured output model
class UserProfile(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    current_occupation: Optional[str] = None
    desired_career: Optional[str] = None
    work_experience: Optional[str] = None


# Create database for session persistence
db = SqliteDb(db_file="tmp/self_analysis_coach.db")

# In-process profile memory to prevent loss across tool calls/turns
# This accumulates fields and is merged with the agent's session_state each update.
profile_memory = UserProfile().model_dump()

def _merge_profile(session_state: dict) -> dict:
    """
    Merge the in-process memory with the session state's user_profile.
    Non-None and non-empty values from session_state take precedence.
    """
    merged = deepcopy(profile_memory)
    in_state = (session_state or {}).get("user_profile", {}) or {}
    for k, v in in_state.items():
        if v is not None and v != "":
            merged[k] = v
    return merged


def _persist_memory(profile: dict) -> None:
    """
    Persist non-empty values into the in-process memory.
    """
    for k, v in (profile or {}).items():
        if v is not None and v != "":
            profile_memory[k] = v


def update_profile(session_state, field: str, value: str) -> str:
    """Update a specific field in the user profile, merging with existing state."""
    # Coerce age to int when possible
    if field == "age":
        try:
            value = int(str(value).strip().split()[0])
        except Exception:
            pass
    # Merge with memory so previously captured fields aren’t lost
    profile = _merge_profile(session_state)
    profile[field] = value
    session_state["user_profile"] = profile
    _persist_memory(profile)
    print(f"DEBUG: Updated {field} = {value}")
    print(f"DEBUG: Current profile (merged): {session_state['user_profile']}")
    return f"Updated {field}: {value}"


def update_multiple_fields(session_state, updates: dict) -> str:
    """Update multiple fields at once, merging with existing state."""
    profile = _merge_profile(session_state)
    for field, value in (updates or {}).items():
        if value is None or value == "":
            continue
        if field == "age":
            try:
                value = int(str(value).strip().split()[0])
            except Exception:
                pass
        profile[field] = value
    session_state["user_profile"] = profile
    _persist_memory(profile)
    print(f"DEBUG: Updated multiple fields: {updates}")
    print(f"DEBUG: Current profile (merged): {session_state['user_profile']}")
    return f"Updated multiple fields: {list(updates.keys())}"


def confirm_profile(session_state) -> str:
    """Mark profile as confirmed"""
    session_state['profile_confirmed'] = True
    print(f"DEBUG: Profile confirmed. Final state: {session_state['user_profile']}")
    return "Profile confirmed!"


def check_profile_completeness(session_state) -> str:
    """Check which fields are still missing"""
    profile = session_state['user_profile']
    missing_fields = []

    if not profile.get('name'):
        missing_fields.append('name')
    if not profile.get('age'):
        missing_fields.append('age')
    if not profile.get('current_occupation'):
        missing_fields.append('current_occupation')
    if not profile.get('desired_career'):
        missing_fields.append('desired_career')
    if not profile.get('work_experience'):
        missing_fields.append('work_experience')

    return f"Missing fields: {missing_fields}" if missing_fields else "Profile is complete"


def run_coaching_session(name: Optional[str] = None, age: Optional[int] = None):
    """
    Run the coaching session with optional name and age parameters

    Args:
        name: User's name (if already known)
        age: User's age (if already known)
    """

    # Initialize user profile with provided data
    initial_profile = UserProfile().model_dump()
    if name:
        initial_profile['name'] = name
    if age:
        initial_profile['age'] = age

    print(f"DEBUG: Initial profile: {initial_profile}")
    # Seed in-process memory with any provided initial values
    _persist_memory(initial_profile)

    # Create the Self Analysis Coach agent
    coach_agent = Agent(
        name="Self Analysis Coach",
        model=OpenAIChat(id="gpt-4o-mini"),
        db=db,
        session_id="coaching_session",
        add_history_to_context=True,
        session_state={
            "user_profile": initial_profile,
            "profile_confirmed": False
        },
        add_session_state_to_context=True,
        enable_agentic_state=True,
        tools=[update_profile, update_multiple_fields, confirm_profile, check_profile_completeness],
        instructions=[
            "You are a personal Self Analysis Coach helping users explore their career path.",
            "",
            "GOAL: Collect these 5 pieces of information:",
            "1. Name",
            "2. Age",
            "3. Current occupation or student status",
            "4. Desired career path",
            "5. Work experience or part-time job history",
            "",
            "CURRENT PROFILE STATUS:",
            f"Name: {initial_profile.get('name', 'Not provided')}",
            f"Age: {initial_profile.get('age', 'Not provided')}",
            f"Current Occupation: {initial_profile.get('current_occupation', 'Not provided')}",
            f"Desired Career: {initial_profile.get('desired_career', 'Not provided')}",
            f"Work Experience: {initial_profile.get('work_experience', 'Not provided')}",
            "",
            "IMPORTANT CONVERSATION RULES:",
            "",
            "1. HANDLING MULTIPLE DATA AT ONCE:",
            "   - If user provides multiple pieces of information in one response, extract ALL relevant data",
            "   - Use update_multiple_fields tool to update several fields at once",
            "   - Example: 'I'm John, 25, working as a developer' → update name, age, and occupation",
            "",
            "2. HANDLING VAGUE ANSWERS:",
            "   - If answer is too vague, ask specific follow-up questions",
            "   - Examples of vague answers: 'I work', 'I'm in tech', 'I study'",
            "   - Ask for specifics: 'What's your specific role?', 'Which field of tech?', 'What do you study?'",
            "",
            "3. WORK EXPERIENCE HANDLING:",
            "   - If user mentions current job, also ask about work experience duration",
            "   - If they say 'working for 3 years', update both current_occupation AND work_experience",
            "",
            "4. CONFIRMATION PROCESS:",
            "   - Use check_profile_completeness to see what's missing",
            "   - Only use confirm_profile when ALL 5 fields are filled",
            "   - Always show a summary before confirming",
            "",
            "5. GREETING BEHAVIOR:",
            "   - If name provided: greet by name",
            "   - If no name: ask for name first",
            "   - Be warm and encouraging throughout",
            "",
            "EXAMPLE CONVERSATION FLOWS:",
            "",
            "User: 'Hi, I'm Sarah, 28, working as a software engineer for 5 years, want to become a tech lead'",
            "→ Extract: name=Sarah, age=28, current_occupation=software engineer, work_experience=5 years, desired_career=tech lead",
            "→ Use update_multiple_fields with all data",
            "→ Confirm all information is correct",
            "",
            "User: 'I work in tech'",
            "→ Too vague! Ask: 'What's your specific role in tech? Are you a developer, designer, analyst, etc.?'",
            "",
            "Always use the tools to update the session state as you collect information!"
        ],
        markdown=True
    )

    print("🎯 Welcome to your Self Analysis Coaching Session!")
    print("Type 'exit' to end the session at any time.\n")

    # Dynamic greeting and first question based on provided data
    if name and age:
        print(f"🤖 Coach: Hello {name}! I'm your personal Self Analysis Coach. I see you're {age} years old.")
        print("Let me help you explore your career path. What's your current occupation or are you a student?")
    elif name:
        print(f"🤖 Coach: Hello {name}! I'm your personal Self Analysis Coach.")
        print("Could you please tell me your age?")
    elif age:
        print(f"���� Coach: Hello! I'm your personal Self Analysis Coach. I see you're {age} years old.")
        print("What's your name?")
    else:
        print("🤖 Coach: Hello! I'm your personal Self Analysis Coach.")
        print("Let's start by getting to know you better. What's your name?")

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n🎯 Thank you for the coaching session! Take care!")
            break

        if not user_input:
            continue

        coach_agent.print_response(user_input, stream=True)

        # Check if profile is complete
        current_state = coach_agent.get_session_state()
        print(f"DEBUG: Session state after response: {current_state}")

        if current_state and current_state.get('profile_confirmed', False):
            # Merge before final rendering to avoid empty outputs
            profile = _merge_profile(current_state)
            print(f"DEBUG: Final profile data (merged): {profile}")

            print("\n✅ All information collected successfully!")
            print("📊 Final Profile Summary:")

            # Display all collected information
            profile_fields = [
                ('Name', 'name'),
                ('Age', 'age'),
                ('Current Occupation', 'current_occupation'),
                ('Desired Career', 'desired_career'),
                ('Work Experience', 'work_experience')
            ]

            for display_name, field_key in profile_fields:
                value = profile.get(field_key)
                if value is not None and value != "":
                    print(f"{display_name}: {value}")
                    profile_data_to_store_to_db[display_name] = value


            print(
                "\n🎯 Thank you for sharing your information! In the next step, we will discuss your goals and aspirations in more detail.")
            break


# Example usage scenarios
if __name__ == "__main__":
    # Test the scenario with comprehensive data handling
    print("=== Test: Comprehensive Data Handling ===")
    # run_coaching_session(name="John", age=30)
    # run_coaching_session(age=30)
    # run_coaching_session(name="John")
    run_coaching_session()

"""
📊 Final Profile Summary:
Name: John
Age: 30
Current Occupation: engineer
Desired Career: poetry
Work Experience: 4 years
"""


print("*"*50)
print("Profile Data", profile_data_to_store_to_db)