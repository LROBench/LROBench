import json
from openai import OpenAI
import asyncio
import tiktoken
import time
import os 

class LLMCaller:
    def __init__(self):
        try:
            with open("./src/conf/conf.json", "r", encoding="utf-8") as file:
                conf = json.load(file) 
        except FileNotFoundError:
            print("Configuration file './src/conf/conf.json' is not found")
        self.model = os.environ.get('MODEL', 'gpt-5')
        self.api_key = os.environ.get('API_KEY', '')
        self.base_url = os.environ.get('BASE_URL', 'https://api.openai.com/v1') 

        self.temperature = float(conf['temperature'])
        self.max_token = int(conf['max_token'])
        self.stream = conf['stream'].lower() == 'true'
        self.max_tries = int(conf['max_tries'])

        self.total_tokens_used = 0  # 累计 token 消耗
        self.input_tokens = 0 # input tokens
        self.output_tokens = 0
        self.semaphore = asyncio.Semaphore(10)
        
        self.kwargs = {
            "model": self.model,
            "max_tokens": self.max_token,
        }
        if self.model not in ['gpt-5-2025-08-07']:
            self.kwargs["temperature"] = self.temperature

        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            timeout = 40*60
        )


    def call(self, query):
        tries = 0
        while tries < self.max_tries:
            try:
                if not self.stream:
                    response = self.client.chat.completions.create(
                        messages=query,
                        **self.kwargs
                    )
                    if response.usage:
                        self.total_tokens_used += response.usage.total_tokens
                        self.input_tokens += response.usage.prompt_tokens
                        self.output_tokens += response.usage.completion_tokens
                    return response.choices[0].message.content
                else:
                    response = self.client.chat.completions.create(
                        messages=query,
                        stream=True,
                        **self.kwargs
                    )

                    full_response = ""
                    for chunk in response:
                        if hasattr(chunk, 'choices') and chunk.choices is not None and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                full_response += delta.content
                                # yield delta.content
                    # print(full_response)

                    encoding = tiktoken.encoding_for_model("gpt-4o")
                    input_token = len(encoding.encode("".join([f"{m['role']}: {m['content']}\n" for m in query])))
                    output_token = len(encoding.encode(full_response))
                    self.total_tokens_used += input_token + output_token
                    self.input_tokens += input_token
                    self.output_tokens += output_token

                    return full_response
                
                
            except Exception as e:
                tries += 1
                print(f"Error occurred while processing query: {e}. Retrying ({tries}/{self.max_tries})...")
                time.sleep(1)
        

    async def async_call(self, query):
        async with self.semaphore:
            tries = 0
            while tries < self.max_tries:
                try:
                    if not self.stream:
                        response = await asyncio.to_thread(
                            lambda: self.client.chat.completions.create(
                                messages=query, 
                                **self.kwargs
                            )
                        )
                        
                        if response.usage:
                            self.total_tokens_used += response.usage.total_tokens
                            self.input_tokens += response.usage.prompt_tokens
                            self.output_tokens += response.usage.completion_tokens
                        return response.choices[0].message.content
                    else:
                        response = await asyncio.to_thread(
                            lambda: self.client.chat.completions.create(
                                messages=query, 
                                stream=True,
                                **self.kwargs
                            )
                        )
                        full_response = ""
                        for chunk in response:
                            if hasattr(chunk, 'choices') and chunk.choices is not None and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    full_response += delta.content

                        encoding = tiktoken.encoding_for_model("gpt-4o")
                        input_token = len(encoding.encode("".join([f"{m['role']}: {m['content']}\n" for m in query])))
                        output_token = len(encoding.encode(full_response))
                        self.total_tokens_used += input_token + output_token
                        self.input_tokens += input_token
                        self.output_tokens += output_token

                        return full_response
                except Exception as e:
                    tries += 1
                    print(f"Error occurred while processing query: {e}. Retrying ({tries}/{self.max_tries})...")
                    await asyncio.sleep(1) 


    async def call_batch_async(self, queries):
        async def delayed_call(query):
            await asyncio.sleep(5)
            return await self.async_call(query)
        tasks = [delayed_call(query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results


    def get_total_tokens_used(self):
        return self.total_tokens_used, self.input_tokens, self.output_tokens

