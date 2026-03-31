/**
 * Kimi API Adapter
 *
 * This adapter converts Claude Code's API calls to work with Kimi AI's API
 * Kimi uses OpenAI-compatible endpoints at https://api.moonshot.ai/v1
 */

import type {
  BetaMessage,
  BetaMessageStreamParams,
  BetaRawMessageStreamEvent
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import type { Stream } from '@anthropic-ai/sdk/streaming.mjs'

interface KimiConfig {
  apiKey: string
  baseURL?: string
  model?: string
}

interface KimiMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface KimiChatCompletionRequest {
  model: string
  messages: KimiMessage[]
  temperature?: number
  max_tokens?: number
  stream?: boolean
}

interface KimiChatCompletionResponse {
  id: string
  object: string
  created: number
  model: string
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
 * Convert Claude message format to Kimi format
 */
function convertClaudeMessagesToKimi(messages: any[]): KimiMessage[] {
  const kimiMessages: KimiMessage[] = []

  for (const msg of messages) {
    if (msg.role === 'user' || msg.role === 'assistant') {
      // Extract text content from Claude's content blocks
      let textContent = ''
      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === 'text') {
            textContent += block.text
          } else if (block.type === 'tool_use') {
            // Convert tool use to text description for Kimi
            textContent += `\n[Tool: ${block.name}]\n${JSON.stringify(block.input, null, 2)}\n`
          } else if (block.type === 'tool_result') {
            // Convert tool result to text
            textContent += `\n[Result]\n${block.content}\n`
          }
        }
      } else if (typeof msg.content === 'string') {
        textContent = msg.content
      }

      kimiMessages.push({
        role: msg.role,
        content: textContent
      })
    }
  }

  return kimiMessages
}

/**
 * Convert Kimi response to Claude format
 */
function convertKimiResponseToClaude(kimiResponse: KimiChatCompletionResponse): BetaMessage {
  const choice = kimiResponse.choices[0]

  return {
    id: kimiResponse.id,
    type: 'message' as const,
    role: 'assistant' as const,
    content: [{
      type: 'text' as const,
      text: choice.message.content
    }],
    model: kimiResponse.model,
    stop_reason: choice.finish_reason === 'stop' ? 'end_turn' : 'max_tokens',
    stop_sequence: null,
    usage: {
      input_tokens: kimiResponse.usage.prompt_tokens,
      output_tokens: kimiResponse.usage.completion_tokens,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0
    }
  }
}

/**
 * Main Kimi API Client
 */
export class KimiAPIClient {
  private apiKey: string
  private baseURL: string
  private model: string

  constructor(config: KimiConfig) {
    this.apiKey = config.apiKey || process.env.MOONSHOT_API_KEY || ''
    this.baseURL = config.baseURL || process.env.KIMI_BASE_URL || 'https://api.moonshot.ai/v1'
    this.model = config.model || process.env.KIMI_MODEL || 'kimi-k2.5'

    if (!this.apiKey) {
      throw new Error('Kimi API key is required. Set MOONSHOT_API_KEY environment variable.')
    }
  }

  /**
   * Create a chat completion (non-streaming)
   */
  async createMessage(params: BetaMessageStreamParams): Promise<BetaMessage> {
    const kimiMessages = convertClaudeMessagesToKimi(params.messages)

    // Add system prompt if provided
    if (params.system) {
      const systemContent = Array.isArray(params.system)
        ? params.system.map(block => block.type === 'text' ? block.text : '').join('\n')
        : params.system

      kimiMessages.unshift({
        role: 'system',
        content: systemContent
      })
    }

    const request: KimiChatCompletionRequest = {
      model: this.model,
      messages: kimiMessages,
      temperature: params.temperature,
      max_tokens: params.max_tokens,
      stream: false
    }

    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`Kimi API error: ${response.status} - ${error}`)
    }

    const kimiResponse: KimiChatCompletionResponse = await response.json()
    return convertKimiResponseToClaude(kimiResponse)
  }

  /**
   * Create a streaming chat completion
   */
  async *createMessageStream(params: BetaMessageStreamParams): AsyncGenerator<any> {
    const kimiMessages = convertClaudeMessagesToKimi(params.messages)

    // Add system prompt if provided
    if (params.system) {
      const systemContent = Array.isArray(params.system)
        ? params.system.map(block => block.type === 'text' ? block.text : '').join('\n')
        : params.system

      kimiMessages.unshift({
        role: 'system',
        content: systemContent
      })
    }

    const request: KimiChatCompletionRequest = {
      model: this.model,
      messages: kimiMessages,
      temperature: params.temperature,
      max_tokens: params.max_tokens,
      stream: true
    }

    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`Kimi API error: ${response.status} - ${error}`)
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
 * Factory function to create Kimi client compatible with Claude Code's expectations
 */
export function createKimiClient(config: KimiConfig) {
  const client = new KimiAPIClient(config)

  return {
    beta: {
      messages: {
        create: (params: BetaMessageStreamParams) => client.createMessage(params),
        stream: (params: BetaMessageStreamParams) => client.createMessageStream(params)
      }
    }
  }
}
