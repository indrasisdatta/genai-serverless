The Model Context Protocol (MCP) is an open-standard protocol designed to solve the "data silo" problem in AI. It creates a universal interface between AI models (like Claude or Gemini) and the external data sources they need to access (like GitHub, Slack, or local databases).

Think of it as USB-C for AI. Just as USB-C allows any device to connect to any peripheral, MCP allows any AI model to connect to any data source without writing custom integrations for every single combination.

## Core Architecture: The Three Pillars
The MCP architecture follows a simple Client-Host-Server model.

- **The Host (The AI Application)**
  - This is the program the user interacts with (e.g., Claude Desktop, IDEs, or a custom-built AI agent).

  - The Host is responsible for managing the lifecycle of the connection and providing the UI for the user.

- The Client (The Connector)

  - The Client lives inside the Host application.

  - It maintains a 1:1 connection with an MCP Server. It translates the AI's requests into the MCP protocol and vice versa.

- The Server (The Data Provider)

  - These are lightweight programs that expose specific capabilities.

  - A server connects to a specific resource (like a Postgres database) and "advertises" what it can do to the Client.

### Key Components of a Server
When a Server connects to a Client, it exposes three main primitives:

 - **Resources:** These are "read-only" data sources. Think of them like files or API endpoints (e.g., a README file or a log stream).

- **Tools:** These are "executable" functions. The AI can call these to do something (e.g., "Create a Jira ticket" or "Run this Python script").

- **Prompts:** These are pre-defined templates that help the user interact with the data (e.g., a "Code Review" prompt that automatically pulls in recent git commits).

## Technical Flow: How it Works
 - The communication usually happens over JSON-RPC 2.0. 
 - The standard transport layers are stdio (for local processes) or HTTP/SSE (for remote services).

- Discovery: When the Host starts, the Client connects to the Server. The Server sends a list of its available Tools and Resources.

 - Reasoning: The user asks a question. The AI model looks at the available Tools and decides which one to use.

 - Execution: The Client sends a request to the Server to execute the tool.

 - Response: The Server sends the result back to the Client, which feeds it into the AI's context window.

## Why It’s a Game Changer
Security: Servers run locally or in controlled environments. You don't have to give your secret API keys to the LLM provider; the Server handles the credentials.

Interoperability: You can write a Server once in TypeScript or Python, and it will immediately work across any IDE or AI app that supports MCP.

Context Control: Instead of dumping thousands of files into an LLM, the model can surgically "fetch" exactly what it needs via MCP tools.