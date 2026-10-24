"""Exercise the installed Anthropic-to-chat tool bridge without a GPU or API keys."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from soulx_agent.llm_runner import load_llm_plan


async def main():
    import litellm

    paths = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            paths.append(self.path)
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions", self.path
            assert request["tools"][0]["type"] == "function"
            response = {
                "id": "chatcmpl-check",
                "object": "chat.completion",
                "created": 1,
                "model": "bridge-check",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-check",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    settings = load_llm_plan().gateway_config()["litellm_settings"]
    litellm.use_chat_completions_url_for_anthropic_messages = settings[
        "use_chat_completions_url_for_anthropic_messages"
    ]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = await litellm.anthropic_messages(
            model="openai/bridge-check",
            api_key="local",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            max_tokens=32,
            stream=False,
            messages=[{"role": "user", "content": "Use lookup"}],
            tools=[
                {
                    "name": "lookup",
                    "description": "Look up a value",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        assert response["stop_reason"] == "tool_use", response
        assert response["content"][0]["type"] == "tool_use", response
        assert paths == ["/v1/chat/completions"], paths
        print("Installed Anthropic-to-chat bridge: function tool conversion passed", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    asyncio.run(main())
