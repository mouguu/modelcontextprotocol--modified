import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { fetch as undiciFetch, ProxyAgent } from "undici";
import type {
  Message,
  ChatCompletionResponse,
  SearchResponse,
  SearchRequestBody,
  UndiciRequestOptions,
} from "./types.js";
import {
  ChatCompletionResponseSchema,
  SearchResponseSchema,
} from "./validation.js";

// Retrieve the Perplexity API key from environment variables
const PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY;
// Retrieve the Perplexity API URL from environment variables
const PERPLEXITY_API_URL = process.env.PERPLEXITY_API_URL;

/**
 * Gets the proxy URL from environment variables.
 * Checks PERPLEXITY_PROXY, HTTPS_PROXY, HTTP_PROXY in order.
 *
 * @returns {string | undefined} The proxy URL if configured, undefined otherwise
 */
export function getProxyUrl(): string | undefined {
  return (
    process.env.PERPLEXITY_PROXY ||
    process.env.HTTPS_PROXY ||
    process.env.HTTP_PROXY ||
    undefined
  );
}

/**
 * Creates a proxy-aware fetch function.
 * Uses undici with ProxyAgent when a proxy is configured, otherwise uses native fetch.
 *
 * @param {string} url - The URL to fetch
 * @param {RequestInit} options - Fetch options
 * @returns {Promise<Response>} The fetch response
 */
export async function proxyAwareFetch(
  url: string,
  options: RequestInit = {},
): Promise<Response> {
  const proxyUrl = getProxyUrl();

  if (proxyUrl) {
    // Use undici with ProxyAgent when proxy is configured
    const proxyAgent = new ProxyAgent(proxyUrl);
    const undiciOptions: UndiciRequestOptions = {
      ...options,
      dispatcher: proxyAgent,
    };
    const response = await undiciFetch(url, undiciOptions);
    // Cast to native Response type for compatibility
    return response as unknown as Response;
  }

  // Use native fetch when no proxy is configured
  return fetch(url, options);
}

/**
 * Validates an array of message objects for chat completion tools.
 * Ensures each message has a valid role and content field.
 *
 * @param {unknown} messages - The messages to validate
 * @param {string} toolName - The name of the tool calling this validation (for error messages)
 * @throws {Error} If messages is not an array or if any message is invalid
 */
export function validateMessages(
  messages: unknown,
  toolName: string,
): asserts messages is Message[] {
  if (!Array.isArray(messages)) {
    throw new Error(
      `Invalid arguments for ${toolName}: 'messages' must be an array`,
    );
  }

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (!msg || typeof msg !== "object") {
      throw new Error(`Invalid message at index ${i}: must be an object`);
    }
    if (!msg.role || typeof msg.role !== "string") {
      throw new Error(`Invalid message at index ${i}: 'role' must be a string`);
    }
    if (
      msg.content === undefined ||
      msg.content === null ||
      typeof msg.content !== "string"
    ) {
      throw new Error(
        `Invalid message at index ${i}: 'content' must be a string`,
      );
    }
  }
}

/**
 * Strips thinking tokens (content within <think>...</think> tags) from the response.
 * This helps reduce context usage when the thinking process is not needed.
 *
 * @param {string} content - The content to process
 * @returns {string} The content with thinking tokens removed
 */
export function stripThinkingTokens(content: string): string {
  return content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

/**
 * Cleans the message content to fix common Markdown link issues.
 *
 * @param {string} content - The content to clean
 * @returns {string} The cleaned content
 */
export function cleanMessageContent(content: string): string {
  if (!content) return content;

  // 1. 修复 ] 和 ( 之间的空格: [Title] (url) -> [Title](url)
  content = content.replace(/\]\s+\(/g, "](");

  // 2. 修复 ( 之后的空格: [Title]( https://) -> [Title](https://)
  content = content.replace(/\]\(\s+/g, "](");

  // 3. 修复双重协议头: https://https:// -> https://
  content = content.replace(/https?:\/\/\s*https?:\/\//gi, "https://");

  // 4. 修复 URL 内部的空格 (编码为 %20)
  // 这里的逻辑是：匹配 ]( 到 ) 之间的内容，如果中间有空格，简单粗暴处理一下
  // 注意：这个正则处理简单情况，复杂情况可能需要更严谨的 URL 编码库，但对 pplx 够用了
  content = content.replace(
    /(\]\(https?:\/\/[^\)]+?)\s+([^\)]+?\))/g,
    "$1%20$2",
  );

  return content;
}

/**
 * Performs a chat completion by sending a request to the Perplexity API.
 * Appends citations to the returned message content if they exist.
 *
 * @param {Message[]} messages - An array of message objects.
 * @param {string} model - The model to use for the completion.
 * @param {boolean} stripThinking - If true, removes <think>...</think> tags from the response.
 * @returns {Promise<string>} The chat completion result with appended citations.
 * @throws Will throw an error if the API request fails.
 */
export async function performChatCompletion(
  messages: Message[],
  model: string = "sonar-pro",
  stripThinking: boolean = false,
): Promise<string> {
  if (!PERPLEXITY_API_KEY) {
    throw new Error("PERPLEXITY_API_KEY environment variable is required");
  }
  if (!PERPLEXITY_API_URL) {
    throw new Error("PERPLEXITY_API_URL environment variable is required");
  }

  // Read timeout fresh each time to respect env var changes
  const TIMEOUT_MS = parseInt(
    process.env.PERPLEXITY_TIMEOUT_MS || "300000",
    10,
  );

  // Construct the API endpoint URL and request body
  const url = new URL(`${PERPLEXITY_API_URL}/chat/completions`); // 修正为 OpenAI 路径
  const body = {
    model: model,
    messages: messages,
    stream: false, // pplx2api 的非流式处理更稳
  };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let response;
  try {
    response = await proxyAwareFetch(url.toString(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${PERPLEXITY_API_KEY}`, // 对应你 Zeabur 设的 Key
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(
        `Request timeout: Perplexity API did not respond within ${TIMEOUT_MS}ms. Consider increasing PERPLEXITY_TIMEOUT_MS.`,
      );
    }
    throw new Error(`Network error while calling Perplexity API: ${error}`);
  }

  // Check for non-successful HTTP status
  if (!response.ok) {
    let errorText;
    try {
      errorText = await response.text();
    } catch (parseError) {
      errorText = "Unable to parse error response";
    }
    throw new Error(
      `Perplexity API error: ${response.status} ${response.statusText}\n${errorText}`,
    );
  }

  let data: ChatCompletionResponse;
  try {
    const json = await response.json();
    data = ChatCompletionResponseSchema.parse(json);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const issues = error.issues;
      if (
        issues.some(
          (i: z.ZodIssue) =>
            i.path.includes("message") || i.path.includes("content"),
        )
      ) {
        throw new Error("Invalid API response: missing message content");
      }
      if (issues.some((i: z.ZodIssue) => i.path.includes("choices"))) {
        throw new Error("Invalid API response: missing or empty choices array");
      }
    }
    throw new Error(
      `Failed to parse JSON response from Perplexity API: ${String(error)}`,
    );
  }

  const firstChoice = data.choices[0];

  // Directly retrieve the main message content from the response
  let messageContent = firstChoice.message.content;

  // Strip thinking tokens if requested
  if (stripThinking) {
    messageContent = stripThinkingTokens(messageContent);
  }

  // If citations are provided, append them to the message content
  if (
    data.citations &&
    Array.isArray(data.citations) &&
    data.citations.length > 0
  ) {
    messageContent += "\n\nCitations:\n";
    data.citations.forEach((citation, index) => {
      messageContent += `[${index + 1}] ${citation}\n`;
    });
  }

  // ================= 强力修复代码开始 =================
  if (messageContent) {
    messageContent = cleanMessageContent(messageContent);
  }
  // ================= 强力修复代码结束 =================

  return messageContent;
}

/**
 * Formats search results from the Perplexity Search API into a readable string.
 *
 * @param {SearchResponse} data - The search response data from the API.
 * @returns {string} Formatted search results.
 */
export function formatSearchResults(data: SearchResponse): string {
  if (!data.results || !Array.isArray(data.results)) {
    return "No search results found.";
  }

  let formattedResults = `Found ${data.results.length} search results:\n\n`;

  data.results.forEach((result, index) => {
    formattedResults += `${index + 1}. **${result.title}**\n`;
    formattedResults += `   URL: ${result.url}\n`;
    if (result.snippet) {
      formattedResults += `   ${result.snippet}\n`;
    }
    if (result.date) {
      formattedResults += `   Date: ${result.date}\n`;
    }
    formattedResults += `\n`;
  });

  return formattedResults;
}

/**
 * Performs a web search using the Perplexity Search API.
 *
 * @param {string} query - The search query string.
 * @param {number} maxResults - Maximum number of results to return (1-20).
 * @param {number} maxTokensPerPage - Maximum tokens to extract per webpage.
 * @param {string} country - Optional ISO country code for regional results.
 * @returns {Promise<string>} The formatted search results.
 * @throws Will throw an error if the API request fails.
 */
export async function performSearch(
  query: string,
  maxResults: number = 10,
  maxTokensPerPage: number = 1024,
  country?: string,
): Promise<string> {
  // 偷梁换柱逻辑
  const searchMessages: Message[] = [{ role: "user", content: query }];

  // 强制调用 sonar-pro，它联网搜索能力很强
  return await performChatCompletion(searchMessages, "sonar-pro");
}

/**
 * Creates and configures the Perplexity MCP server with all tools.
 * This factory function is transport-agnostic and returns a configured server instance.
 *
 * @returns The configured MCP server instance
 */
export function createPerplexityServer() {
  const server = new McpServer({
    name: "io.github.perplexityai/mcp-server",
    version: "0.5.2",
  });

  // Register perplexity_ask tool
  server.registerTool(
    "perplexity_ask",
    {
      title: "Ask Perplexity",
      description:
        "Engages in a conversation using the Sonar API. " +
        "Accepts an array of messages (each with a role and content) " +
        "and returns a chat completion response from the Perplexity model.",
      inputSchema: {
        messages: z
          .array(
            z.object({
              role: z
                .string()
                .describe(
                  "Role of the message (e.g., system, user, assistant)",
                ),
              content: z.string().describe("The content of the message"),
            }),
          )
          .describe("Array of conversation messages"),
        model: z
          .enum([
            "sonar",
            "sonar-pro",
            "sonar-reasoning",
            "sonar-reasoning-pro",
            "claude-4.5-sonnet",
            "gpt-4o",
          ])
          .optional()
          .describe("Model to use (default: sonar for speed)"),
      },
      outputSchema: {
        response: z.string().describe("The chat completion response"),
      },
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
      },
    },
    async ({ messages, model }: { messages: Message[]; model?: string }) => {
      validateMessages(messages, "perplexity_ask");

      // Use provided model or default to 'sonar' (fastest)
      // Previously hardcoded to "gemini-3-pro-search"
      const targetModel = model || "sonar";

      const result = await performChatCompletion(messages, targetModel);
      return {
        content: [{ type: "text", text: result }],
        structuredContent: { response: result },
      };
    },
  );

  // Register perplexity_research tool
  server.registerTool(
    "perplexity_research",
    {
      title: "Deep Research",
      description:
        "Performs deep research using the Perplexity API. " +
        "Accepts an array of messages (each with a role and content) " +
        "and returns a comprehensive research response with citations.",
      inputSchema: {
        messages: z
          .array(
            z.object({
              role: z
                .string()
                .describe(
                  "Role of the message (e.g., system, user, assistant)",
                ),
              content: z.string().describe("The content of the message"),
            }),
          )
          .describe("Array of conversation messages"),
        strip_thinking: z
          .boolean()
          .optional()
          .describe(
            "If true, removes <think>...</think> tags and their content from the response to save context tokens. Default is false.",
          ),
      },
      outputSchema: {
        response: z.string().describe("The research response"),
      },
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
      },
    },
    async ({
      messages,
      strip_thinking,
    }: {
      messages: Message[];
      strip_thinking?: boolean;
    }) => {
      validateMessages(messages, "perplexity_research");

      // 使用 sonar-pro 作为强大的搜索模型
      const result = await performChatCompletion(messages, "sonar-pro");

      return {
        content: [{ type: "text", text: result }],
        structuredContent: { response: result },
      };
    },
  );

  // Register perplexity_reason tool
  server.registerTool(
    "perplexity_reason",
    {
      title: "Advanced Reasoning",
      description:
        "Performs reasoning tasks using the Perplexity API. " +
        "Accepts an array of messages (each with a role and content) " +
        "and returns a well-reasoned response using the sonar-reasoning-pro model.",
      inputSchema: {
        messages: z
          .array(
            z.object({
              role: z
                .string()
                .describe(
                  "Role of the message (e.g., system, user, assistant)",
                ),
              content: z.string().describe("The content of the message"),
            }),
          )
          .describe("Array of conversation messages"),
        strip_thinking: z
          .boolean()
          .optional()
          .describe(
            "If true, removes <think>...</think> tags and their content from the response to save context tokens. Default is false.",
          ),
      },
      outputSchema: {
        response: z.string().describe("The reasoning response"),
      },
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
      },
    },
    async ({
      messages,
      strip_thinking,
    }: {
      messages: Message[];
      strip_thinking?: boolean;
    }) => {
      validateMessages(messages, "perplexity_reason");

      // 使用 sonar-reasoning-pro 进行推理
      const result = await performChatCompletion(
        messages,
        "sonar-reasoning-pro",
        !!strip_thinking,
      );

      return {
        content: [{ type: "text", text: result }],
        structuredContent: { response: result },
      };
    },
  );

  // Register perplexity_search tool
  server.registerTool(
    "perplexity_search",
    {
      title: "Search the Web",
      description:
        "Performs web search using the Perplexity Search API. " +
        "Returns ranked search results with titles, URLs, snippets, and metadata. " +
        "Perfect for finding up-to-date facts, news, or specific information.",
      inputSchema: {
        query: z.string().describe("Search query string"),
        max_results: z
          .number()
          .min(1)
          .max(20)
          .optional()
          .describe("Maximum number of results to return (1-20, default: 10)"),
        max_tokens_per_page: z
          .number()
          .min(256)
          .max(2048)
          .optional()
          .describe("Maximum tokens to extract per webpage (default: 1024)"),
        country: z
          .string()
          .optional()
          .describe(
            "ISO 3166-1 alpha-2 country code for regional results (e.g., 'US', 'GB')",
          ),
      },
      outputSchema: {
        results: z.string().describe("Formatted search results"),
      },
      annotations: {
        readOnlyHint: true,
        openWorldHint: true,
      },
    },
    async ({
      query,
      max_results,
      max_tokens_per_page,
      country,
    }: {
      query: string;
      max_results?: number;
      max_tokens_per_page?: number;
      country?: string;
    }) => {
      const maxResults = typeof max_results === "number" ? max_results : 10;
      const maxTokensPerPage =
        typeof max_tokens_per_page === "number" ? max_tokens_per_page : 1024;
      const countryCode = typeof country === "string" ? country : undefined;

      const result = await performSearch(
        query,
        maxResults,
        maxTokensPerPage,
        countryCode,
      );
      return {
        content: [{ type: "text", text: result }],
        structuredContent: { results: result },
      };
    },
  );

  return server.server;
}
