#!/usr/bin/env node

import express from "express";
import cors from "cors";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createPerplexityServer } from "./server.js";
import { logger } from "./logger.js";

// Check for required API key (Server to Perplexity)
const PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY;
if (!PERPLEXITY_API_KEY) {
  logger.error("PERPLEXITY_API_KEY environment variable is required");
  process.exit(1);
}

// Check for Client Authentication Key (Client to This Server)
// If set, clients must provide Authorization header
const MCP_SERVER_API_KEY = process.env.MCP_SERVER_API_KEY;

const app = express();
const PORT = parseInt(process.env.PORT || "8080", 10);
const BIND_ADDRESS = process.env.BIND_ADDRESS || "0.0.0.0";
const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS?.split(",") || ["*"];

// CORS configuration for browser-based MCP clients
app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true);

      if (ALLOWED_ORIGINS.includes("*")) {
        return callback(null, true);
      }

      if (ALLOWED_ORIGINS.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error(`Origin ${origin} not allowed by CORS`));
      }
    },
    exposedHeaders: ["Mcp-Session-Id", "mcp-protocol-version"],
    allowedHeaders: ["Content-Type", "mcp-session-id", "Authorization"], // Allow Authorization header
  })
);

app.use(express.json());

// --- Authentication Middleware ---
app.use("/mcp", (req, res, next) => {
  // If no server-side API key is configured, allow all requests
  if (!MCP_SERVER_API_KEY) {
    return next();
  }

  const authHeader = req.headers.authorization;

  if (!authHeader) {
    logger.warn("Client attempted connection without Authorization header");
    res
      .status(401)
      .json({ error: "Unauthorized: Missing Authorization header" });
    return;
  }

  // Support "Bearer <token>" or direct "<token>" format
  const token = authHeader.startsWith("Bearer ")
    ? authHeader.slice(7).trim()
    : authHeader.trim();

  if (token !== MCP_SERVER_API_KEY) {
    logger.warn("Client attempted connection with invalid API key");
    res.status(403).json({ error: "Forbidden: Invalid API Key" });
    return;
  }

  next();
});
// ------------------------------

const mcpServer = createPerplexityServer();

/**
 * POST: client-to-server messages (requests, responses, notifications)
 * GET: SSE stream for server-to-client messages (notifications, requests)
 */
app.all("/mcp", async (req, res) => {
  try {
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });

    res.on("close", () => {
      transport.close();
    });

    await mcpServer.connect(transport);

    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    logger.error("Error handling MCP request", { error: String(error) });
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: { code: -32603, message: "Internal server error" },
        id: null,
      });
    }
  }
});

/**
 * Health check endpoint
 */
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "perplexity-mcp-server",
    auth_enabled: !!MCP_SERVER_API_KEY,
  });
});

/**
 * Start the HTTP server
 */
app
  .listen(PORT, BIND_ADDRESS, () => {
    logger.info(
      `Perplexity MCP Server listening on http://${BIND_ADDRESS}:${PORT}/mcp`
    );
    if (MCP_SERVER_API_KEY) {
      logger.info(
        "🔒 Server Access Control: ENABLED (Clients must provide Authorization header)"
      );
    } else {
      logger.warn("⚠️  Server Access Control: DISABLED (Publicly accessible)");
    }
  })
  .on("error", (error) => {
    logger.error("Server error", { error: String(error) });
    process.exit(1);
  });
