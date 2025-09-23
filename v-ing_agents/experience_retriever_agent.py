import os
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.tools import tool

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")


class EventDetails(BaseModel):
    """Structured details describing a single life experience."""

    event_id: int
    event_number: int
    event_phase: str
    name: Optional[str] = None
    age: Optional[str] = None
    current_occupation: Optional[str] = None
    desired_career: Optional[str] = None
    work_experience: Optional[str] = None
    event_overview: Optional[str] = None
    when_happened: Optional[str] = None
    what_happened: Optional[str] = None
    peak_moment: Optional[str] = None
    is_complete: bool = False


REQUIRED_EVENT_FIELDS = (
    "event_overview",
    "when_happened",
    "what_happened",
    "peak_moment",
)

CONFIRMATION_PHRASES = {
    "yes",
    "yes.",
    "yes!",
    "yes, that's right",
    "yes that's right",
    "yes, correct",
    "correct",
    "correct.",
    "correct!",
    "that's correct",
    "that's right",
    "that is correct",
    "looks good",
    "looks good.",
    "looks good to me",
    "sounds good",
    "sounds good.",
    "sounds right",
    "all good",
    "all good.",
    "perfect",
    "perfect.",
    "confirmed",
    "confirmed.",
    "absolutely",
    "absolutely.",
}


def _is_confirmation_phrase(text: str) -> bool:
    return text.strip().lower() in CONFIRMATION_PHRASES


def _all_required_fields_present(session_state: Dict[str, Any]) -> bool:
    current_event = session_state.get("current_event", {})
    for field in REQUIRED_EVENT_FIELDS:
        value = current_event.get(field)
        if not value or not str(value).strip():
            return False
    return True


def _ensure_event_container(session_state: Dict[str, Any]) -> Dict[str, Any]:
    if "current_event" not in session_state:
        session_state["current_event"] = {}
    return session_state["current_event"]


@tool
def update_single_field(session_state: Dict[str, Any], field_name: str, field_value: str) -> str:
    """Store a single field gathered from the user."""

    field_name = field_name.strip()
    field_value = (field_value or "").strip()

    if field_name not in REQUIRED_EVENT_FIELDS:
        return (
            "Invalid field. Use one of: "
            + ", ".join(REQUIRED_EVENT_FIELDS)
        )

    if not field_value:
        return f"No update made to {field_name}."

    current_event = _ensure_event_container(session_state)
    current_event[field_name] = field_value
    return f"Stored {field_name}."


@tool
def update_multiple_fields(session_state: Dict[str, Any], updates: Dict[str, str]) -> str:
    """Store multiple experience fields at once."""

    current_event = _ensure_event_container(session_state)
    applied: Dict[str, str] = {}

    for field, value in updates.items():
        if field in REQUIRED_EVENT_FIELDS and value and value.strip():
            current_event[field] = value.strip()
            applied[field] = value.strip()

    if not applied:
        return "No valid fields provided."

    return "Stored fields: " + ", ".join(sorted(applied.keys()))


@tool
def confirm_data_from_user(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Present the collected data back to the user for confirmation."""

    current_event = session_state.get("current_event", {})

    return {
        "status": "awaiting_confirmation",
        "collected_data": current_event,
        "message": (
            "Please review the summary above and let me know if everything is accurate "
            "or if we should adjust anything."
        ),
    }


@tool
def confirm_completeness(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Mark the experience complete when all fields are captured."""

    current_event = session_state.get("current_event", {})
    missing_fields = [
        field for field in REQUIRED_EVENT_FIELDS
        if not current_event.get(field)
    ]

    if missing_fields:
        return {
            "status": "incomplete",
            "missing_fields": missing_fields,
            "message": "Still need: " + ", ".join(missing_fields),
        }

    event_details = _finalize_completed_event(session_state)

    return {
        "status": "complete",
        "message": "All experience details are confirmed and complete.",
        "event_details": event_details.model_dump(),
    }


def _finalize_completed_event(session_state: Dict[str, Any]) -> EventDetails:
    experience_no = session_state.get("experience_no", 1)
    event_phase = session_state.get("stage") or f"experience_{experience_no}"

    event_details = EventDetails(
        event_id=experience_no,
        event_number=experience_no,
        event_phase=event_phase,
        name=session_state.get("name"),
        age=str(session_state.get("age")) if session_state.get("age") is not None else None,
        current_occupation=session_state.get("current_occupation"),
        desired_career=session_state.get("desired_career"),
        work_experience=session_state.get("work_experience"),
        event_overview=session_state.get("current_event", {}).get("event_overview"),
        when_happened=session_state.get("current_event", {}).get("when_happened"),
        what_happened=session_state.get("current_event", {}).get("what_happened"),
        peak_moment=session_state.get("current_event", {}).get("peak_moment"),
        is_complete=True,
    )

    session_state["completed_event"] = event_details.model_dump()
    session_state["session_complete"] = True
    return event_details


def _build_instruction_block(profile_data: Dict[str, Any]) -> Tuple[str, ...]:
    experience_no = profile_data.get("experience_no", "Not provided")
    stage = profile_data.get("stage", "Not provided")

    return (
        "ROLE: You are an empathetic experience coach helping the user capture one meaningful life event.",
        "PROFILE CONTEXT:",
        f"- Experience number: {experience_no}",
        f"- Stage/phase label: {stage}",
        "- You already have the user's profile data in session_state; never ask for it again.",
        "WORKFLOW FOR EVERY USER MESSAGE:",
        "1. Parse the message for details about event_overview, when_happened, what_happened, and peak_moment.",
        "2. Immediately store any new details using update_multiple_fields (for several items) or update_single_field (for one item) BEFORE replying.",
        "3. Ask warm, specific follow-up questions until all four fields contain concrete detail.",
        "4. Once the four fields are filled, call confirm_data_from_user to show the summary and ask for confirmation.",
        "5. After the user clearly confirms the summary, call confirm_completeness exactly once and acknowledge the completion.",
        "6. Never end the conversation without calling confirm_completeness after a confirmation.",
        "STYLE: Be concise, compassionate, and focused on the experience. Do not mention tool usage."
    )


def create_experience_coach(profile_data: Dict[str, Any]) -> Agent:
    instruction_block = list(_build_instruction_block(profile_data))

    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description="Empathetic coach that gathers structured experience details through conversation.",
        db=SqliteDb(db_file="tmp/coach_sessions.db"),
        enable_user_memories=True,
        add_history_to_context=True,
        add_session_state_to_context=True,
        tools=[update_single_field, update_multiple_fields, confirm_data_from_user, confirm_completeness],
        instructions=instruction_block,
        markdown=True,
    )


def _initial_session_state(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    base_state = dict(profile_data)
    base_state.setdefault("experience_no", 1)
    base_state.setdefault("stage", f"experience_{base_state['experience_no']}")
    base_state.setdefault("current_event", {})
    base_state.setdefault("session_complete", False)
    base_state["profile_data"] = dict(profile_data)
    return base_state


def _print_event_summary(event_details: EventDetails) -> None:
    print("\n=== EXPERIENCE SUMMARY ===")
    print(f"Event Number : {event_details.event_number}")
    print(f"Event Phase  : {event_details.event_phase}")
    print(f"Overview     : {event_details.event_overview}")
    print(f"When         : {event_details.when_happened}")
    print(f"What         : {event_details.what_happened}")
    print(f"Peak Moment  : {event_details.peak_moment}")
    print("==========================")


def run_experience_session(
    profile_data: Dict[str, Any],
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[EventDetails]:
    """Run the conversational experience retriever session."""

    if not profile_data.get("name"):
        raise ValueError("profile_data must include the user's name.")

    agent = create_experience_coach(profile_data)

    session_id = session_id or f"experience_session_{profile_data.get('experience_no', 1)}"
    agent.user_id = user_id or f"{profile_data['name'].lower().replace(' ', '_')}_user"
    agent.session_state = _initial_session_state(profile_data)

    print("=== Experience Retrieval Session ===")
    print("Coach: Thank you for being here. When you're ready, please walk me through a meaningful life experience that has stayed with you.")

    first_turn = True

    while True:
        try:
            user_message = input("\nYou: ")
        except EOFError:
            print("\nCoach: The session ended unexpectedly. Let's continue another time.")
            return None

        user_message = user_message.strip()
        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit", "bye", "stop"}:
            print("\nCoach: Thank you for sharing with me today. We can pick this up any time.")
            return None

        run_kwargs = {"user_id": agent.user_id, "session_id": agent.session_id, "stream": False}
        if first_turn:
            run_kwargs["session_state"] = agent.session_state
            first_turn = False

        response = agent.run(user_message, **run_kwargs)
        reply_text = getattr(response, "content", str(response))
        if reply_text:
            print(f"\nCoach: {reply_text.strip()}")

        session_state = agent.get_session_state() or {}

        if session_state.get("session_complete"):
            completed_event = session_state.get("completed_event")
            if completed_event:
                event_details = EventDetails(**completed_event)
                _print_event_summary(event_details)
                return event_details

            print("\nCoach: I marked the session complete but could not prepare the summary. Let's revisit later.")
            return None

        if _is_confirmation_phrase(user_message) and _all_required_fields_present(session_state):
            event_details = _finalize_completed_event(session_state)
            print("\nCoach: All experience details are confirmed and complete.")
            _print_event_summary(event_details)
            return event_details



def _sample_profile() -> Dict[str, Any]:
    return {
        "name": "Sarah",
        "age": 28,
        "current_occupation": "Marketing Manager",
        "desired_career": "Product Manager",
        "work_experience": "5 years in marketing, led 3 major campaigns",
        "experience_no": 1,
        "stage": "Gather information for experience_1",
    }


if __name__ == "__main__":
    run_experience_session(_sample_profile())