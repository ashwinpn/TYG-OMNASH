import time  # To measure latency
import openai  # OpenAI API

def zero_shot_no_reasoning(role, resumeA, transcriptA, resumeB, transcriptB, api_key):
    """
    Evaluate and rank two candidates based on resumes and transcripts.
    Prints token usage, cost, and latency.
    Args:
        role (str): Role to evaluate candidates for.
        resumeA (str): Resume of Candidate A.
        transcriptA (str): Transcript of Candidate A.
        resumeB (str): Resume of Candidate B.
        transcriptB (str): Transcript of Candidate B.
        api_key (str): OpenAI API key.
    Returns:
        str: Winner ('A' or 'B').
    """

    # Set API key
    openai.api_key = api_key

    # Create the prompt
    prompt = f"""
We are evaluating two candidates for the role of {role}.
Here are their details:

Candidate A:
Resume: {resumeA}
Transcript: {transcriptA}

Candidate B:
Resume: {resumeB}
Transcript: {transcriptB}

Evaluate the strengths and weaknesses of each candidate based on their resumes and interview performance.
Return a JSON with the format:
{{
    "winner": "A" or "B"
}}
Make sure the 'winner' field is either 'A' or 'B'.
"""

    # Measure start time for latency
    start_time = time.time()

    # Call GPT-4 API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )

    # Measure end time for latency
    end_time = time.time()

    # Parse the response
    result = response.choices[0].message.content.strip()

    # Extract and print token usage
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # Cost calculation (assuming GPT-4 pricing as of now)
    # GPT-4 pricing: $0.03 per 1k prompt tokens, $0.06 per 1k completion tokens
    cost = (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)

    # Print stats
    print(f"Tokens Used: Prompt = {prompt_tokens}, Completion = {completion_tokens}, Total = {total_tokens}")
    print(f"Cost: ${cost:.4f}")
    print(f"Latency: {end_time - start_time:.2f} seconds")

    # Return the result
    return result
