import os
from openai import OpenAI
from helpers import clean_text

default_base_prompt = "Reword this passage keeping the tone and meaning the same."
prompt_addition_constant = "Ensure you respond with a different wording of the text. Do not add any additional information in your response. Your response should be the text ONLY. Here is the text:"

initial_conversation = [{"role": "system", "content": "Your job is to rephrase text without changing its meaning. Only rephrase the text, do not add any additional information. Your response should be the rephrased text ONLY."}]

def get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
    return api_key

try:
    client = OpenAI(api_key=get_api_key())
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure you have a valid OpenAI API key.")
    exit(1)

def generate_prompt(text, base_prompt=default_base_prompt):
    # Clean the text in various ways
    text = clean_text(text)
    # Generate the prompt
    prompt = f"{base_prompt} {prompt_addition_constant} {text}"
    return prompt

def generate_completion(prompt, conversation_history):
    # Create messages
    messages = conversation_history + [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="gpt-4o",  # You can change this to "gpt-4" if you have access
        messages=messages,
        temperature=0.5,
    )
    # Append the newly generated completion to the conversation history
    conversation_history.append({"role": "assistant", "content": completion.choices[0].message.content})
    return completion, conversation_history  # Return the updated conversation history

def run_rephrase(text, conversation_history):
    # Generate a prompt
    prompt = generate_prompt(text)
    # Generate a completion
    completion, conversation_history = generate_completion(prompt, conversation_history)
    # Get the completion and clean it
    completion = clean_text(completion.choices[0].message.content)
    # Return the completion
    return completion, conversation_history

def generate_completion_no_history(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": "You always fulfill the user's requests to the best of your ability."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return completion

def run_rephrase_no_history(text, base_prompt=default_base_prompt):
    # Generate a prompt
    prompt = generate_prompt(text, base_prompt)
    # Generate a completion
    completion = generate_completion_no_history(prompt)
    # Get the completion and clean it
    completion = clean_text(completion.choices[0].message.content)
    # Return the completion
    return completion

if __name__ == "__main__":
    # Get a user's text
    text = input("Enter a text: ")
    # Get the number of iterations
    num_iterations = int(input("Enter the number of iterations: "))
    # Do iterations
    for i in range(num_iterations):
        text, conversation_history = run_rephrase(text, initial_conversation)
        input(f"iteration {i}\n {text}")