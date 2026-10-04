# Memory Systems Lab Notes

## Configuration convention

`load_config()` reads `.env` with `python-dotenv` when it is installed and
falls back to the process environment otherwise. This preserves the required
offline benchmark mode: no API key or provider SDK is needed until a live agent
explicitly calls `build_chat_model()`.

| Purpose | Environment variable | Default |
| --- | --- | --- |
| Primary provider / model | `LLM_PROVIDER`, `LLM_MODEL` | `openai`, `gpt-4o-mini` |
| Judge provider / model | `JUDGE_LLM_PROVIDER`, `JUDGE_LLM_MODEL` | primary provider/model |
| Sampling | `LLM_TEMPERATURE` | `0` |
| Compact settings | `COMPACT_THRESHOLD_TOKENS`, `COMPACT_KEEP_MESSAGES` | `800`, `6` |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | — |
| Custom OpenAI-compatible API | `CUSTOM_API_KEY`, `CUSTOM_BASE_URL` | — |
| Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | — |
| Anthropic | `ANTHROPIC_API_KEY` | — |
| Ollama | `OLLAMA_BASE_URL` | `http://localhost:11434` |
| OpenRouter | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` | — |

The 800-token compact threshold leaves normal conversations untouched while
forcing compaction for the supplied roughly 3,000-token stress input. Six
recent messages are retained verbatim after each compaction.

## Provider references

- https://docs.langchain.com/oss/python/integrations/chat/openai/
- https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai/
- https://docs.langchain.com/oss/python/integrations/chat/anthropic/
- https://docs.langchain.com/oss/python/integrations/chat/ollama/
- https://docs.langchain.com/oss/python/integrations/chat/openrouter/
