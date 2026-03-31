/**
 * OpenRouter API Adapter
 *
 * This adapter converts Claude Code's API calls to work with OpenRouter API
 * OpenRouter provides access to multiple LLMs including free models
 * API: https://openrouter.ai/docs
 */

import type {
  BetaMessage,
  BetaMessageStreamParams,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'

interface OpenRouterConfig {
  apiKey: string
  baseURL?: string
  model?: string
  siteName?: string
  siteUrl?: string
}

interface OpenRouterMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface OpenRouterChatCompletionRequest {
  model: string
  messages: OpenRouterMessage[]
  temperature?: number
  max_tokens?: number
  stream?: boolean
  route?: 'fallback'
}

interface OpenRouterChatCompletionResponse {
  id: string
  model: string
  created: number
  choices: Array<{
    index: number
    message: {
      role: string
      content: string
    }
    finish_reason: string
  }>
  usage: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

/**
 * Convert Claude message format to OpenRouter format
 */
function convertClaudeMessagesToOpenRouter(messages: any[]): OpenRouterMessage[] {
  const openRouterMessages: OpenRouterMessage[] = []

  for (const msg of messages) {
    if (msg.role === 'user' || msg.role === 'assistant') {
      // Extract text content from Claude's content blocks
      let textContent = ''
      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === 'text') {
            textContent += block.text
          } else if (block.type === 'tool_use') {
            // Convert tool use to text description
            textContent += `\n[Tool: ${block.name}]\n${JSON.stringify(block.input, null, 2)}\n`
          } else if (block.type === 'tool_result') {
            // Convert tool result to text
            textContent += `\n[Result]\n${block.content}\n`
          }
        }
      } else if (typeof msg.content === 'string') {
        textContent = msg.content
      }

      openRouterMessages.push({
        role: msg.role,
        content: textContent
      })
    }
  }

  return openRouterMessages
}

/**
 * Convert OpenRouter response to Claude format
 */
function convertOpenRouterResponseToClaude(openRouterResponse: OpenRouterChatCompletionResponse): BetaMessage {
  const choice = openRouterResponse.choices[0]

  return {
    id: openRouterResponse.id,
    type: 'message' as const,
    role: 'assistant' as const,
    content: [{
      type: 'text' as const,
      text: choice.message.content
    }],
    model: openRouterResponse.model,
    stop_reason: choice.finish_reason === 'stop' ? 'end_turn' : 'max_tokens',
    stop_sequence: null,
    usage: {
      input_tokens: openRouterResponse.usage.prompt_tokens,
      output_tokens: openRouterResponse.usage.completion_tokens,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0
    }
  }
}

/**
 * Main OpenRouter API Client
 */
export class OpenRouterAPIClient {
  private apiKey: string
  private baseURL: string
  private model: string
  private siteName: string
  private siteUrl: string

  constructor(config: OpenRouterConfig) {
    this.apiKey = config.apiKey || process.env.OPENROUTER_API_KEY || ''
    this.baseURL = config.baseURL || process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1'
    this.model = config.model || process.env.OPENROUTER_MODEL || 'openai/gpt-3.5-turbo'
    this.siteName = config.siteName || process.env.OPENROUTER_SITE_NAME || 'Claude Code'
    this.siteUrl = config.siteUrl || process.env.OPENROUTER_SITE_URL || 'https://github.com/anthropics/claude-code'

    if (!this.apiKey) {
      throw new Error('OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.')
    }
  }

  /**
   * Create a chat completion (non-streaming)
   */
  async createMessage(params: BetaMessageStreamParams): Promise<BetaMessage> {
    const openRouterMessages = convertClaudeMessagesToOpenRouter(params.messages)

    // Add system prompt if provided
    if (params.system) {
      const systemContent = Array.isArray(params.system)
        ? params.system.map(block => block.type === 'text' ? block.text : '').join('\n')
        : params.system

      openRouterMessages.unshift({
        role: 'system',
        content: systemContent
      })
    }

    const request: OpenRouterChatCompletionRequest = {
      model: this.model,
      messages: openRouterMessages,
      temperature: params.temperature,
      max_tokens: params.max_tokens,
      stream: false
    }

    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': this.siteUrl,
        'X-Title': this.siteName
      },
      body: JSON.stringify(request)
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`OpenRouter API error: ${response.status} - ${error}`)
    }

    const openRouterResponse: OpenRouterChatCompletionResponse = await response.json()
    return convertOpenRouterResponseToClaude(openRouterResponse)
  }

  /**
   * Create a streaming chat completion
   */
  async *createMessageStream(params: BetaMessageStreamParams): AsyncGenerator<any> {
    const openRouterMessages = convertClaudeMessagesToOpenRouter(params.messages)

    // Add system prompt if provided
    if (params.system) {
      const systemContent = Array.isArray(params.system)
        ? params.system.map(block => block.type === 'text' ? block.text : '').join('\n')
        : params.system

      openRouterMessages.unshift({
        role: 'system',
        content: systemContent
      })
    }

    const request: OpenRouterChatCompletionRequest = {
      model: this.model,
      messages: openRouterMessages,
      temperature: params.temperature,
      max_tokens: params.max_tokens,
      stream: true
    }

    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': this.siteUrl,
        'X-Title': this.siteName
      },
      body: JSON.stringify(request)
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`OpenRouter API error: ${response.status} - ${error}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('Failed to get response reader')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') continue

            try {
              const parsed = JSON.parse(data)
              const delta = parsed.choices?.[0]?.delta

              if (delta?.content) {
                yield {
                  type: 'content_block_delta',
                  index: 0,
                  delta: {
                    type: 'text_delta',
                    text: delta.content
                  }
                }
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }
}

/**
 * Factory function to create OpenRouter client compatible with Claude Code's expectations
 */
export function createOpenRouterClient(config: OpenRouterConfig) {
  const client = new OpenRouterAPIClient(config)

  return {
    beta: {
      messages: {
        create: (params: BetaMessageStreamParams) => client.createMessage(params),
        stream: (params: BetaMessageStreamParams) => client.createMessageStream(params)
      }
    }
  }
}
