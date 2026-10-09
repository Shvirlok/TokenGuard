import type { PriceTable, ProfileId } from './types'

export const DEFAULT_PROXY_URL = 'http://127.0.0.1:8080'

export interface ProfileDefinition {
  id: ProfileId
  name: string
  description: string
  tag: string
  accent: 'success' | 'warning' | 'info'
  hourly_limit: number
  daily_limit: number
  loop_threshold: number
}

export const PROFILES: ProfileDefinition[] = [
  {
    id: 'careful',
    name: 'Careful',
    description: '$5/hr cap, 2-loop cutoff, alerts ON',
    tag: 'Default Recommended',
    accent: 'success',
    hourly_limit: 5,
    daily_limit: 50,
    loop_threshold: 2,
  },
  {
    id: 'standard',
    name: 'Standard Agent',
    description: '$15/hr cap, 4-loop cutoff',
    tag: 'Autonomous Workflows',
    accent: 'warning',
    hourly_limit: 15,
    daily_limit: 100,
    loop_threshold: 4,
  },
  {
    id: 'passive',
    name: 'Passive Monitor',
    description: 'No loop blocks, tracking only',
    tag: 'Observability Only',
    accent: 'info',
    hourly_limit: 1000,
    daily_limit: 5000,
    loop_threshold: 0,
  },
]

export const DEMO_PRICES: PriceTable = {
  'gpt-4o': { input: 2.5, output: 10 },
  'gpt-4o-mini': { input: 0.15, output: 0.6 },
  'o3-mini': { input: 1.1, output: 4.4 },
  'claude-3-7-sonnet': { input: 3, output: 15 },
  'claude-3-5-sonnet': { input: 3, output: 15 },
  'claude-3-5-haiku': { input: 0.8, output: 4 },
  'gemini-2.0-flash': { input: 0.1, output: 0.4 },
  'gemini-1.5-pro': { input: 1.25, output: 5 },
  'qwen-max': { input: 2.8, output: 8.4 },
  'qwen-plus': { input: 0.4, output: 1.2 },
  'deepseek-chat': { input: 0.27, output: 1.1 },
  'deepseek-reasoner': { input: 0.55, output: 2.19 },
  'llama-3.3-70b-versatile': { input: 0.59, output: 0.79 },
  'mistral-large-latest': { input: 2, output: 6 },
}

const proxy = (base: string) => base || DEFAULT_PROXY_URL

export function getSnippets(base: string) {
  const url = proxy(base)
  const openaiCompat = (comment: string, key: string, env: string, model: string, msg: string) => `from openai import OpenAI

# ${comment}
client = OpenAI(
    base_url="${url}/v1",
    api_key="${key}"  # or ${env} env var
)

response = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "${msg}"}]
)
print(response.choices[0].message.content)`

  return [
    {
      id: 'cli',
      label: 'Zero-Code CLI',
      code: `# Run any Python agent / script protected through TokenGuard:
tokenguard run python my_agent.py

# Or customize repeat sensitivity on the fly:
tokenguard run --max-repeats 2 python my_agent.py`,
    },
    {
      id: 'claude',
      label: 'Claude SDK',
      code: `import anthropic

# Point Anthropic client directly to local TokenGuard proxy:
client = anthropic.Anthropic(
    base_url="${url}",
    api_key="sk-ant-your-key"  # or ANTHROPIC_API_KEY env var
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello Claude via TokenGuard!"}]
)
print(response.content[0].text)`,
    },
    {
      id: 'gemini',
      label: 'Gemini',
      code: openaiCompat(
        'Zero-config Google Gemini (auto-routed to Gemini OpenAI gateway):',
        'AIzaSy-your-gemini-key',
        'GEMINI_API_KEY',
        'gemini-2.0-flash',
        'Hello Gemini via TokenGuard!',
      ),
    },
    {
      id: 'qwen',
      label: 'Qwen / DashScope',
      code: openaiCompat(
        'Zero-config Qwen / Alibaba DashScope:',
        'sk-your-dashscope-key',
        'DASHSCOPE_API_KEY',
        'qwen-max',
        'Hello Qwen via TokenGuard!',
      ),
    },
    {
      id: 'openai',
      label: 'OpenAI',
      code: openaiCompat(
        'Point standard OpenAI client to TokenGuard proxy:',
        'sk-your-key',
        'OPENAI_API_KEY',
        'gpt-4o',
        'Hello OpenAI via TokenGuard!',
      ),
    },
    {
      id: 'deepseek',
      label: 'DeepSeek',
      code: openaiCompat(
        'Zero-config DeepSeek: automatically routed to api.deepseek.com',
        'sk-your-deepseek-key',
        'DEEPSEEK_API_KEY',
        'deepseek-chat',
        'Hello DeepSeek via TokenGuard!',
      ),
    },
    {
      id: 'groq',
      label: 'Groq (Llama)',
      code: openaiCompat(
        'Zero-config Groq (Llama / Gemma / Mixtral):',
        'gsk_your-groq-key',
        'GROQ_API_KEY',
        'llama-3.3-70b-versatile',
        'Ultra-fast inference via Groq!',
      ),
    },
    {
      id: 'mistral',
      label: 'Mistral',
      code: openaiCompat(
        'Zero-config Mistral AI:',
        'your-mistral-key',
        'MISTRAL_API_KEY',
        'mistral-large-latest',
        'Hello Mistral via TokenGuard!',
      ),
    },
    {
      id: 'langchain',
      label: 'LangChain',
      code: `from langchain_openai import ChatOpenAI

# Any LangChain agent, protected by TokenGuard:
llm = ChatOpenAI(
    model="gpt-4o",
    base_url="${url}/v1",
    api_key="sk-your-key",
)

print(llm.invoke("Hello LangChain via TokenGuard!").content)`,
    },
    {
      id: 'curl',
      label: 'cURL',
      code: `# Standard chat completions (works with gpt-4o, gemini, qwen, claude, deepseek):
curl ${url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello via TokenGuard!"}]
  }'`,
    },
  ]
}
