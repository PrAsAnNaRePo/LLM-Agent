import json
import os
import re
import time
from gradio_client import Client
from langchain.tools import DuckDuckGoSearchRun
import colorama

colorama.init()

class Agent:
    def __init__(
            self,
            client_id: str,
            ) -> None:
        self.client = Client(client_id)
        self.search = DuckDuckGoSearchRun()

        self.math_prompt = """[INST] <<SYS>>
Assistant is the well known math professor who can solve very complex problems in any topics step by step.
- Try to solve the problem always step by step.
- Try to help the user to understand the problem clearly.
- Use comments to explain the problem.
- Use latex to write the math equations.
- The assistant will help to write solutions that are easy to read and understand.
<</SYS>>
"""
        self.code_prompt = """[INST] <<SYS>>
Assistant is a well-known programmer who can help you with programming problems in Python, Java, C++, and other languages.
- The assistant will help you break down complex problems into smaller, more manageable steps.
- The assistant will provide comments in the code to help you understand what each line of code is doing.
- The assistant will provide links to relevant documentation or tutorials to help you learn more about a particular topic.
- The assistant will help you write code that is easy to read and understand.
- The assistant will help you write code that is efficient and optimized.
- The assistant will help you write code that is well-documented and commented.
- The assistant will help you write code that is modular and reusable.
- The assistant will help you write code that is testable and maintainable.
<</SYS>>
"""
        self.casual_message_prompt = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. you SHOULD NOT spread false information.
Be respectful and honest. you are a friendly assistant.
<</SYS>>
"""
        self.search_prompt = """[INST] <<SYS>>
You are a helpfull assistant. Assistant is a expert in answer the question from the google search result. Assistant do not spread false information.
- If the search result inapproriate for the question, just leave the search content and answer by yourself.
- Given the search summary answer the question asked by the user.
<</SYS>>
"""
        self.history = """[INST] <<SYS>>
Based on the given input or instruction you have to decide the plan of action.
- You have to decide the plan of action based on the given input.

The available plans are:
- "Search": Useful for when you need to search something that you don't know.
    Example: "How to solve a complex problems easily? plan should be taken: Search"
- "Math": Useful for when you need to solve a math problem.
    Example: "Solve the math problem: 2+2. plan should be taken: Math"
- "Code": Useful for when user asked to code or code problems.
    Example: "Write a code to print hello world. plan should be taken: Code"
- "Message": Useful for when you need to send a Casual message to the user.
    Example: "Send a message to the user: Hello. plan should be taken: Message"


NOTE:
- DON'T use any other plans.
- DON'T use any other format.
- DON'T always use the search plan for simple questions that you know.
- USE Code tool when user asked to generate or wite code.
- USE Math tool when user asked to solve math problems.
- Assistant SHOULD NOT respond with any other format.
- Assistant MUST respond with the plan of action NOT any other information.
- Available plans are: "Search", "Math", "Code", "Message". you SHOULD NOT use any other plans.
<</SYS>>
"""

    def run(self, prompt):
        start_time = time.time()
        response = self.client.predict(self.history + prompt +' [/INST] Prefered Action:', 2, api_name="/predict")
        print("--- %s seconds ---" % (time.time() - start_time))
        res = response.split("[/INST]")[-1].strip().split(':')[-1].strip()

        print(colorama.Fore.YELLOW + f"Action: {res}" + colorama.Style.RESET_ALL)
        
        # input("\nproceed? ")
        
        if res == "Search":
            search_result = self.search.run(prompt)
            search_prompt = self.search_prompt + "Here is the search result:\n" + search_result + f"\n{prompt} [/INST]"
            response = self.client.predict(search_prompt, 3000, api_name="/predict")
            print(colorama.Fore.GREEN + response.split("[/INST]")[-1].strip() + colorama.Style.RESET_ALL)
        

        if res == "Math":
            math_prompt = self.math_prompt + prompt + f" [/INST]"
            response = self.client.predict(math_prompt, 2048, api_name="/predict")
            print(colorama.Fore.GREEN + response.split("[/INST]")[-1].strip() + colorama.Style.RESET_ALL)
           

        if res == "Message":
            message_prompt = self.casual_message_prompt + prompt + f" [/INST]"
            response = self.client.predict(message_prompt, 3000, api_name="/predict")
            print(colorama.Fore.GREEN + response.split("[/INST]")[-1].strip() + colorama.Style.RESET_ALL)
           

        if res == "Code":
            code_prompt = self.code_prompt + prompt + f" [/INST]"
            response = self.client.predict(code_prompt, 3000, api_name="/predict")
            print(colorama.Fore.GREEN + response.split("[/INST]")[-1].strip() + colorama.Style.RESET_ALL)
      

agent = Agent("http://127.0.0.1:7860")
while True:
    prompt = input("Enter the prompt: ")
    agent.run(prompt)

