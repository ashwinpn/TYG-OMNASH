def zero_shot_no_reasoning(role, resumeA, transcriptA, resumeB, transcriptB, api_key):
    """
    Evaluate and rank two candidates based on their resumes and transcripts.
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
    # Set API Key
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

    # Call GPT-4 API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )

    # Parse and return the winner
    return response['choices'][0]['message']['content'].strip()
