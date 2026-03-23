"""
patch_web_server.py

Run once from project root:
    python patch_web_server.py

Adds a /api/chat endpoint to web_server.py so the browser can call
Groq via the server without exposing the API key in the HTML.
"""

import os, sys

WS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "web_server.py")

if not os.path.exists(WS):
    print(f"ERROR: {WS} not found")
    sys.exit(1)

with open(WS, "r", encoding="utf-8") as f:
    content = f.read()

# Check if already patched
if "/api/chat" in content:
    print("Already patched — /api/chat endpoint exists.")
    sys.exit(0)

# Find a good place to insert — after the imports section
# Add the new endpoint after the existing /health endpoint

new_endpoint = '''

@app.post("/api/chat")
async def chat_proxy(request: Request):
    """
    Proxy endpoint so the browser can call Groq without
    exposing the API key in the HTML file.
    """
    import os, requests as req_lib
    body = await request.json()
    messages = body.get("messages", [])
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {"content": "API key not configured on server."}
    try:
        resp = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "max_tokens": 200, "messages": messages},
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return {"content": content}
    except Exception as e:
        return {"content": f"AI unavailable: {e}"}

'''

# Insert before the last few lines (the uvicorn.run call or similar)
# Find the health endpoint and add after it
if '@app.get("/health")' in content:
    # Find the end of the health function
    idx = content.find('@app.get("/health")')
    # Find the next blank line after the function body
    end = content.find('\n\n', idx + 50)
    if end == -1:
        end = len(content)
    content = content[:end] + new_endpoint + content[end:]
    with open(WS, "w", encoding="utf-8") as f:
        f.write(content)
    print("✓ Added /api/chat endpoint to web_server.py")
else:
    # Just append before the last line
    lines = content.rstrip().split('\n')
    content = '\n'.join(lines) + new_endpoint
    with open(WS, "w", encoding="utf-8") as f:
        f.write(content)
    print("✓ Appended /api/chat endpoint to web_server.py")