# V2A Inspect

## Project Structure
This project consists of two parts. The AI inference server, and the main python module where the agent runtime exists.

### AI Inference Server

The server runtime that runs AI inference. Server code is in `server/`

#### Supported Models

1. SAM3: Image segmentation and tracking with natural language prompts
2. DINO v2: Image embedding generation
3. SigLIP2: Image and text embedding generation (for simillarity search)

#### Code Structure
- `models/`: Holds the API request and response models.
- `inference/`: The AI model inference code
- `runtime.py`: Entrypoint for `v2a-inspect-server` CLI command
- `settings.py`: Pydantic setting config
- `app.py`: FastAPI routes


### Agent Runtime

The agent runtime is where agentic video to multitrack audio pipeline runs. Code is in `src/`

#### Code Structure
- `client`: API client for the AI inference server. Designed to not depend on anything.
- `config`: Pydantic setting config
- `models`: Pydantic models used for data
- `prompts`: Prompt manager for the agent. Includes prompts as .txt file that is embedded to the python module when built.
- `preprocessing`: Preprocessing pipeline.
