"""
Session 3 - Demo 10: The Full Agent (async)
This is the complete agent with the agentic loop — fully async.

Before running:
  pip install google-genai python-dotenv fastmcp
  Create a .env file next to this script with:
    GEMINI_API_KEY=your-key-here
    GEMINI_MODEL=gemini-2.5-flash-lite
"""
import asyncio
import json
import math
import os
import re

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

# ============================================================
# Configuration
# ============================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
THROTTLE_SECONDS = 10  # free-tier RPM guard

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Create a .env file with GEMINI_API_KEY=...")

client = genai.Client(api_key=GEMINI_API_KEY)


# ── Pydantic schemas for tool arguments & responses ────────────────────────────

class CalculateArgs(BaseModel):
    expression: str

class WeatherArgs(BaseModel):
    city: str

class SearchNotesArgs(BaseModel):
    query: str

class ToolCall(BaseModel):
    tool_name: str
    tool_arguments: dict

class FinalAnswer(BaseModel):
    answer: str


async def call_llm(prompt: str) -> str:
    """Send a prompt to Gemini asynchronously; throttle to respect free-tier RPM."""
    print(f"  [waiting {THROTTLE_SECONDS}s to respect rate limits...]", flush=True)
    await asyncio.sleep(THROTTLE_SECONDS)
    response = await client.aio.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


# ============================================================
# System Prompt — This is what turns an LLM into an agent
# ============================================================
system_prompt = """You are a helpful AI agent that can use tools to answer questions accurately.

You have access to the following tools:

1. calculate(expression: str) -> str
   Evaluate a mathematical expression using Python syntax.
   Examples: calculate("2**10"), calculate("math.sqrt(144)"), calculate("sum(math.exp(x) for x in [1,1,2,3,5,8])")

2. get_weather(city: str) -> str
   Get the current weather for a city.
   Examples: get_weather("Mumbai"), get_weather("London")

3. search_notes(query: str) -> str
   Search through user's notes for relevant information.
   Examples: search_notes("meeting"), search_notes("project ideas")

You must respond in ONE of these two JSON formats:

If you need to use a tool:
{"tool_name": "<name>", "tool_arguments": {"<arg_name>": "<value>"}}

If you have the final answer:
{"answer": "<your final answer>"}

IMPORTANT RULES:
- Respond with ONLY the JSON. No other text. No markdown code fences.
- Use tools when you need real data or precise calculations.
- After receiving a tool result, either use another tool or provide your final answer.
- For complex calculations, break them down into steps if needed.
- ALWAYS use the calculate tool for math — do NOT try to compute in your head.
"""


# ============================================================
# Tools — async functions the agent can call
# ============================================================

async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    args = CalculateArgs(expression=expression)
    try:
        allowed = {
            "math": math, "abs": abs, "round": round, "pow": pow,
            "sum": sum, "min": min, "max": max, "range": range, "list": list,
        }
        result = eval(args.expression, {"__builtins__": {}}, allowed)
        return json.dumps({"result": str(result)})
    except Exception as e:
        return json.dumps({"error": f"Calculation failed: {e}"})


async def get_weather(city: str) -> str:
    """Get current weather for a city (simulated; swap for a real API call)."""
    args = WeatherArgs(city=city)
    weather_data = {
        "Mumbai":        {"temp": "32°C", "condition": "Humid, Partly Cloudy", "humidity": "78%"},
        "Delhi":         {"temp": "28°C", "condition": "Clear Sky",            "humidity": "45%"},
        "London":        {"temp": "15°C", "condition": "Rainy",                "humidity": "85%"},
        "New York":      {"temp": "22°C", "condition": "Sunny",                "humidity": "55%"},
        "Tokyo":         {"temp": "26°C", "condition": "Windy",                "humidity": "60%"},
        "San Francisco": {"temp": "18°C", "condition": "Foggy",                "humidity": "72%"},
    }
    if args.city in weather_data:
        return json.dumps({"weather": weather_data[args.city]})
    return json.dumps({"error": f"Weather data not available for {args.city}"})


async def search_notes(query: str) -> str:
    """Search through user's notes (simulated)."""
    args = SearchNotesArgs(query=query)
    notes = [
        {"title": "Meeting Agenda",  "content": "Discuss Q3 targets, review agent architecture, plan MCP integration"},
        {"title": "Shopping List",   "content": "Milk, eggs, bread, coffee, batteries"},
        {"title": "Project Ideas",   "content": "Build a stock monitoring agent, voice-based assistant, browser automation tool"},
        {"title": "Travel Plans",    "content": "Tokyo trip in December, need to book flights and hotel by November"},
        {"title": "Learning Notes",  "content": "Finish transformer session, practice async Python, read MCP docs"},
    ]
    q = args.query.lower()
    results = [n for n in notes if q in n["title"].lower() or q in n["content"].lower()]
    return json.dumps({"results": results or "No notes found matching your query"})


# Tool registry — maps tool names to async functions
tools: dict[str, any] = {
    "calculate":    calculate,
    "get_weather":  get_weather,
    "search_notes": search_notes,
}


# ============================================================
# Response Parser — Handles messy LLM output
# ============================================================

def parse_llm_response(text: str) -> dict:
    """Parse the LLM's response, handling common formatting issues"""
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (opening fence)
        lines = lines[1:]
        # Remove last line if it's a closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        # Remove language identifier
        if text.startswith("json"):
            text = text[4:].strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse LLM response: {text[:200]}")


# ============================================================
# The Agent Loop — This is where the magic happens
# ============================================================

async def run_agent(user_query: str, max_iterations: int = 5, verbose: bool = True) -> str | None:
    """
    Async agent loop: User query → LLM → [Tool call → Result → LLM]* → Final answer.
    Tools run as coroutines; the loop awaits each tool result before the next LLM call.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  User: {user_query}")
        print(f"{'='*60}")

    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_query},
    ]

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
            elif msg["role"] == "tool":
                prompt += f"Tool Result: {msg['content']}\n\n"

        response_text = await call_llm(prompt)
        if verbose:
            print(f"LLM: {response_text.strip()}")

        try:
            parsed = parse_llm_response(response_text)
        except (ValueError, json.JSONDecodeError) as e:
            if verbose:
                print(f"Parse error: {e} — asking LLM to retry...")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": "Please respond with valid JSON only. No markdown, no extra text."})
            continue

        if "answer" in parsed:
            answer = FinalAnswer(**parsed).answer
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Agent Answer: {answer}")
                print(f"{'='*60}")
            return answer

        if "tool_name" in parsed:
            try:
                call = ToolCall(**parsed)
            except Exception as e:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "tool", "content": json.dumps({"error": f"Bad tool call schema: {e}"})})
                continue

            if verbose:
                print(f"→ Calling tool: {call.tool_name}({call.tool_arguments})")

            if call.tool_name not in tools:
                error_msg = json.dumps({"error": f"Unknown tool: {call.tool_name}. Available: {list(tools.keys())}"})
                if verbose:
                    print(f"→ Error: {error_msg}")
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "tool", "content": error_msg})
                continue

            tool_result = await tools[call.tool_name](**call.tool_arguments)
            if verbose:
                print(f"→ Result: {tool_result}")

            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "tool", "content": tool_result})

    print("\nMax iterations reached. Agent could not complete the task.")
    if verbose:
        print(f"\n{'='*60}\nFull conversation history:\n{'='*60}")
        for i, msg in enumerate(messages):
            print(f"[{i}] {msg['role']}: {msg['content'][:100]}...")
    return None


# ============================================================
# Run the agent!
# ============================================================

async def main() -> None:
    print("\n" + "=" * 60)
    print("  SESSION 3: YOUR FIRST AI AGENT (async)")
    print("=" * 60)

    print("\n\n>>> TEST 1: Simple weather query")
    await run_agent("What is the weather in Mumbai?")

    print("\n\n>>> TEST 2: Math that LLMs get wrong")
    await run_agent("What is 2 raised to the power of 10, plus the square root of 144?")

    print("\n\n>>> TEST 3: Multi-step reasoning")
    await run_agent(
        "Search my notes for travel plans, then check the weather in "
        "the city I'm planning to visit."
    )

    print("\n\n>>> TEST 4: The classic — sum of exponentials of Fibonacci")
    await run_agent(
        "Calculate the sum of exponential values (e^x) of the "
        "first 6 Fibonacci numbers: 1, 1, 2, 3, 5, 8"
    )

    print("\n\n>>> TEST 5: No tools needed")
    await run_agent("What is the capital of France?")


if __name__ == "__main__":
    asyncio.run(main())
