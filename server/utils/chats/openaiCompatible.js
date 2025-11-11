const { v4: uuidv4 } = require("uuid");
const { DocumentManager } = require("../DocumentManager");
const { WorkspaceChats } = require("../../models/workspaceChats");
const { getVectorDbClass, getLLMProvider } = require("../helpers");
const { writeResponseChunk } = require("../helpers/chat/responses");
const { chatPrompt, sourceIdentifier } = require("./index");

const { PassThrough } = require("stream");

// Helper function to safely parse JSON with validation and repair attempts
function safeJSONParse(jsonString, context = "") {
  if (!jsonString || typeof jsonString !== 'string') {
    console.warn(`[safeJSONParse] Invalid input: ${context}`, { 
      type: typeof jsonString, 
      value: jsonString?.substring(0, 100) 
    });
    return { success: false, error: "Invalid or empty JSON string", data: null };
  }

  // Trim whitespace
  jsonString = jsonString.trim();
  
  if (jsonString.length === 0) {
    return { success: false, error: "Empty JSON string after trim", data: null };
  }

  try {
    const parsed = JSON.parse(jsonString);
    return { success: true, data: parsed, error: null };
  } catch (parseError) {
    console.warn(`[safeJSONParse] Parse failed: ${context}`, {
      error: parseError.message,
      jsonPreview: jsonString.substring(0, 200),
      length: jsonString.length
    });

    // Attempt to fix common JSON issues
    try {
      // Try to close incomplete objects/arrays
      let fixed = jsonString;
      
      // Count braces and brackets
      const openBraces = (fixed.match(/{/g) || []).length;
      const closeBraces = (fixed.match(/}/g) || []).length;
      const openBrackets = (fixed.match(/\[/g) || []).length;
      const closeBrackets = (fixed.match(/]/g) || []).length;
      
      // Add missing closing braces
      if (openBraces > closeBraces) {
        fixed += '}'.repeat(openBraces - closeBraces);
      }
      
      // Add missing closing brackets
      if (openBrackets > closeBrackets) {
        fixed += ']'.repeat(openBrackets - closeBrackets);
      }
      
      // Try to close incomplete strings
      const quotes = (fixed.match(/"/g) || []).length;
      if (quotes % 2 !== 0) {
        fixed += '"';
      }

      const repairedParsed = JSON.parse(fixed);
      console.info(`[safeJSONParse] Successfully repaired JSON: ${context}`);
      return { success: true, data: repairedParsed, error: null, repaired: true };
    } catch (repairError) {
      return { 
        success: false, 
        error: `Parse failed: ${parseError.message}. Repair failed: ${repairError.message}`,
        data: null,
        originalError: parseError,
        jsonPreview: jsonString.substring(0, 200)
      };
    }
  }
}

// Helper to extract complete SSE messages from buffer
function extractSSEMessages(buffer) {
  const messages = [];
  const lines = buffer.split('\n');
  let currentMessage = '';
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    // SSE messages end with double newline
    if (line === '' && currentMessage) {
      messages.push(currentMessage);
      currentMessage = '';
    } else if (line.startsWith('data: ')) {
      currentMessage = line.substring(6); // Remove 'data: ' prefix
    }
  }
  
  // Return messages and remaining incomplete data
  return {
    messages,
    remaining: currentMessage || lines[lines.length - 1] || ''
  };
}

async function chatSync({
  workspace,
  systemPrompt = null,
  history = [],
  prompt = null,
  attachments = [],
  temperature = null,
}) {
  const uuid = uuidv4();
  const chatMode = workspace?.chatMode ?? "chat";
  const LLMConnector = getLLMProvider({
    provider: workspace?.chatProvider,
    model: workspace?.chatModel,
  });
  const VectorDb = getVectorDbClass();
  const hasVectorizedSpace = await VectorDb.hasNamespace(workspace.slug);
  const embeddingsCount = await VectorDb.namespaceCount(workspace.slug);

  // User is trying to query-mode chat a workspace that has no data in it - so
  // we should exit early as no information can be found under these conditions.
  if ((!hasVectorizedSpace || embeddingsCount === 0) && chatMode === "query") {
    const textResponse =
      workspace?.queryRefusalResponse ??
      "There is no relevant information in this workspace to answer your query.";

    await WorkspaceChats.new({
      workspaceId: workspace.id,
      prompt: String(prompt),
      response: {
        text: textResponse,
        sources: [],
        type: chatMode,
        attachments,
      },
      include: false,
    });

    return formatJSON(
      {
        id: uuid,
        type: "textResponse",
        sources: [],
        close: true,
        error: null,
        textResponse,
      },
      { model: workspace.slug, finish_reason: "abort" }
    );
  }

  // If we are here we know that we are in a workspace that is:
  // 1. Chatting in "chat" mode and may or may _not_ have embeddings
  // 2. Chatting in "query" mode and has at least 1 embedding
  let contextTexts = [];
  let sources = [];
  let pinnedDocIdentifiers = [];
  await new DocumentManager({
    workspace,
    maxTokens: LLMConnector.promptWindowLimit(),
  })
    .pinnedDocs()
    .then((pinnedDocs) => {
      pinnedDocs.forEach((doc) => {
        const { pageContent, ...metadata } = doc;
        pinnedDocIdentifiers.push(sourceIdentifier(doc));
        contextTexts.push(doc.pageContent);
        sources.push({
          text:
            pageContent.slice(0, 1_000) +
            "...continued on in source document...",
          ...metadata,
        });
      });
    });

  const vectorSearchResults =
    embeddingsCount !== 0
      ? await VectorDb.performSimilaritySearch({
          namespace: workspace.slug,
          input: String(prompt),
          LLMConnector,
          similarityThreshold: workspace?.similarityThreshold,
          topN: workspace?.topN,
          filterIdentifiers: pinnedDocIdentifiers,
          rerank: workspace?.vectorSearchMode === "rerank",
        })
      : {
          contextTexts: [],
          sources: [],
          message: null,
        };

  // Failed similarity search if it was run at all and failed.
  if (!!vectorSearchResults.message) {
    return formatJSON(
      {
        id: uuid,
        type: "abort",
        textResponse: null,
        sources: [],
        close: true,
        error: vectorSearchResults.message,
      },
      { model: workspace.slug, finish_reason: "abort" }
    );
  }

  // For OpenAI Compatible chats, we cannot do backfilling so we simply aggregate results here.
  contextTexts = [...contextTexts, ...vectorSearchResults.contextTexts];
  sources = [...sources, ...vectorSearchResults.sources];

  // If in query mode and no context chunks are found from search, backfill, or pins -  do not
  // let the LLM try to hallucinate a response or use general knowledge and exit early
  if (chatMode === "query" && contextTexts.length === 0) {
    const textResponse =
      workspace?.queryRefusalResponse ??
      "There is no relevant information in this workspace to answer your query.";

    await WorkspaceChats.new({
      workspaceId: workspace.id,
      prompt: String(prompt),
      response: {
        text: textResponse,
        sources: [],
        type: chatMode,
        attachments,
      },
      include: false,
    });

    return formatJSON(
      {
        id: uuid,
        type: "textResponse",
        sources: [],
        close: true,
        error: null,
        textResponse,
      },
      { model: workspace.slug, finish_reason: "no_content" }
    );
  }

  // Compress & Assemble message to ensure prompt passes token limit with room for response
  // and build system messages based on inputs and history.
  const messages = await LLMConnector.compressMessages({
    systemPrompt: systemPrompt ?? (await chatPrompt(workspace)),
    userPrompt: String(prompt),
    contextTexts,
    chatHistory: history,
    attachments,
  });

  // Send the text completion.
  const { textResponse, metrics } = await LLMConnector.getChatCompletion(
    messages,
    {
      temperature:
        temperature ?? workspace?.openAiTemp ?? LLMConnector.defaultTemp,
    }
  );

  if (!textResponse) {
    return formatJSON(
      {
        id: uuid,
        type: "textResponse",
        sources: [],
        close: true,
        error: "No text completion could be completed with this input.",
        textResponse: null,
      },
      { model: workspace.slug, finish_reason: "no_content", usage: metrics }
    );
  }

  const { chat } = await WorkspaceChats.new({
    workspaceId: workspace.id,
    prompt: String(prompt),
    response: {
      text: textResponse,
      sources,
      type: chatMode,
      metrics,
      attachments,
    },
  });

  return formatJSON(
    {
      id: uuid,
      type: "textResponse",
      close: true,
      error: null,
      chatId: chat.id,
      textResponse,
      sources,
    },
    { model: workspace.slug, finish_reason: "stop", usage: metrics }
  );
}

async function streamChat({
  workspace,
  response,
  systemPrompt = null,
  history = [],
  prompt = null,
  attachments = [],
  temperature = null,
}) {
  const uuid = uuidv4();
  const chatMode = workspace?.chatMode ?? "chat";
  const LLMConnector = getLLMProvider({
    provider: workspace?.chatProvider,
    model: workspace?.chatModel,
  });
  const VectorDb = getVectorDbClass();
  const hasVectorizedSpace = await VectorDb.hasNamespace(workspace.slug);
  const embeddingsCount = await VectorDb.namespaceCount(workspace.slug);

  // Buffer for incomplete chunks
  let chunkBuffer = '';
  let chunkCount = 0;
  let parseErrorCount = 0;

  // We don't want to write a new method for every LLM to support openAI calls
  // via the `handleStreamResponseV2` method handler. So here we create a passthrough
  // that on writes to the main response, transforms the chunk to OpenAI format.
  // The chunk is coming in the format from `writeResponseChunk` but in the AnythingLLM
  // response chunk schema, so we here we mutate each chunk.
  const responseInterceptor = new PassThrough({});
  responseInterceptor.on("data", (chunk) => {
    chunkCount++;
    
    try {
      // Convert chunk to string and add to buffer
      const chunkStr = chunk.toString();
      chunkBuffer += chunkStr;

      // Extract complete SSE messages
      const { messages, remaining } = extractSSEMessages(chunkBuffer);
      chunkBuffer = remaining;

      // Process each complete message
      for (const message of messages) {
        if (!message || message.trim().length === 0) {
          continue;
        }

        // Attempt to parse the JSON
        const parseResult = safeJSONParse(message, `chunk #${chunkCount}`);
        
        if (parseResult.success) {
          const originalData = parseResult.data;
          
          if (parseResult.repaired) {
            console.warn(`[responseInterceptor] Used repaired JSON for chunk #${chunkCount}`);
          }

          // Transform to OpenAI format
          const modified = formatJSON(originalData, {
            chunked: true,
            model: workspace.slug,
          });

          // Write to response
          response.write(`data: ${JSON.stringify(modified)}\n\n`);
        } else {
          // JSON parsing failed even after repair attempts
          parseErrorCount++;
          console.error(`[responseInterceptor] JSON parse error in chunk #${chunkCount}:`, {
            error: parseResult.error,
            preview: parseResult.jsonPreview,
            chunkStr: chunkStr.substring(0, 200),
            bufferSize: chunkBuffer.length
          });

          // Only fail hard if we have too many consecutive errors
          if (parseErrorCount > 5) {
            console.error(`[responseInterceptor] Too many parse errors (${parseErrorCount}), aborting stream`);
            response.write(`data: ${JSON.stringify({
              error: "Stream parsing failed due to malformed data",
              details: parseResult.error,
              model: workspace.slug
            })}\n\n`);
            responseInterceptor.destroy();
          }
        }
      }
      
      // Reset error count on successful parse
      if (messages.length > 0) {
        parseErrorCount = 0;
      }

    } catch (e) {
      console.error(`[responseInterceptor] Unexpected error in chunk #${chunkCount}:`, {
        error: e.message,
        stack: e.stack,
        chunkPreview: chunk.toString().substring(0, 200),
        bufferSize: chunkBuffer.length
      });
    }
  });

  // Handle stream end to process any remaining buffered data
  responseInterceptor.on("end", () => {
    if (chunkBuffer.trim().length > 0) {
      console.warn(`[responseInterceptor] Stream ended with remaining buffer:`, {
        bufferSize: chunkBuffer.length,
        preview: chunkBuffer.substring(0, 200)
      });

      // Try to parse remaining buffer
      const parseResult = safeJSONParse(chunkBuffer, "final buffer");
      if (parseResult.success) {
        const modified = formatJSON(parseResult.data, {
          chunked: true,
          model: workspace.slug,
        });
        response.write(`data: ${JSON.stringify(modified)}\n\n`);
      }
    }
  });

  // Handle errors
  responseInterceptor.on("error", (err) => {
    console.error(`[responseInterceptor] Stream error:`, {
      error: err.message,
      stack: err.stack
    });
  });

  // User is trying to query-mode chat a workspace that has no data in it - so
  // we should exit early as no information can be found under these conditions.
  if ((!hasVectorizedSpace || embeddingsCount === 0) && chatMode === "query") {
    const textResponse =
      workspace?.queryRefusalResponse ??
      "There is no relevant information in this workspace to answer your query.";

    await WorkspaceChats.new({
      workspaceId: workspace.id,
      prompt: String(prompt),
      response: {
        text: textResponse,
        sources: [],
        type: chatMode,
        attachments,
      },
      include: false,
    });

    writeResponseChunk(
      response,
      formatJSON(
        {
          id: uuid,
          type: "textResponse",
          sources: [],
          close: true,
          error: null,
          textResponse,
        },
        { chunked: true, model: workspace.slug, finish_reason: "abort" }
      )
    );
    return;
  }

  // If we are here we know that we are in a workspace that is:
  // 1. Chatting in "chat" mode and may or may _not_ have embeddings
  // 2. Chatting in "query" mode and has at least 1 embedding
  let contextTexts = [];
  let sources = [];
  let pinnedDocIdentifiers = [];
  await new DocumentManager({
    workspace,
    maxTokens: LLMConnector.promptWindowLimit(),
  })
    .pinnedDocs()
    .then((pinnedDocs) => {
      pinnedDocs.forEach((doc) => {
        const { pageContent, ...metadata } = doc;
        pinnedDocIdentifiers.push(sourceIdentifier(doc));
        contextTexts.push(doc.pageContent);
        sources.push({
          text:
            pageContent.slice(0, 1_000) +
            "...continued on in source document...",
          ...metadata,
        });
      });
    });

  const vectorSearchResults =
    embeddingsCount !== 0
      ? await VectorDb.performSimilaritySearch({
          namespace: workspace.slug,
          input: String(prompt),
          LLMConnector,
          similarityThreshold: workspace?.similarityThreshold,
          topN: workspace?.topN,
          filterIdentifiers: pinnedDocIdentifiers,
          rerank: workspace?.vectorSearchMode === "rerank",
        })
      : {
          contextTexts: [],
          sources: [],
          message: null,
        };

  // Failed similarity search if it was run at all and failed.
  if (!!vectorSearchResults.message) {
    writeResponseChunk(
      response,
      formatJSON(
        {
          id: uuid,
          type: "abort",
          textResponse: null,
          sources: [],
          close: true,
          error: vectorSearchResults.message,
        },
        { chunked: true, model: workspace.slug, finish_reason: "abort" }
      )
    );
    return;
  }

  // For OpenAI Compatible chats, we cannot do backfilling so we simply aggregate results here.
  contextTexts = [...contextTexts, ...vectorSearchResults.contextTexts];
  sources = [...sources, ...vectorSearchResults.sources];

  // If in query mode and no context chunks are found from search, backfill, or pins -  do not
  // let the LLM try to hallucinate a response or use general knowledge and exit early
  if (chatMode === "query" && contextTexts.length === 0) {
    const textResponse =
      workspace?.queryRefusalResponse ??
      "There is no relevant information in this workspace to answer your query.";

    await WorkspaceChats.new({
      workspaceId: workspace.id,
      prompt: String(prompt),
      response: {
        text: textResponse,
        sources: [],
        type: chatMode,
        attachments,
      },
      include: false,
    });

    writeResponseChunk(
      response,
      formatJSON(
        {
          id: uuid,
          type: "textResponse",
          sources: [],
          close: true,
          error: null,
          textResponse,
        },
        { chunked: true, model: workspace.slug, finish_reason: "no_content" }
      )
    );
    return;
  }

  // Compress & Assemble message to ensure prompt passes token limit with room for response
  // and build system messages based on inputs and history.
  const messages = await LLMConnector.compressMessages({
    systemPrompt: systemPrompt ?? (await chatPrompt(workspace)),
    userPrompt: String(prompt),
    contextTexts,
    chatHistory: history,
    attachments,
  });

  if (!LLMConnector.streamingEnabled()) {
    writeResponseChunk(
      response,
      formatJSON(
        {
          id: uuid,
          type: "textResponse",
          sources: [],
          close: true,
          error: "Streaming is not available for the connected LLM Provider",
          textResponse: null,
        },
        {
          chunked: true,
          model: workspace.slug,
          finish_reason: "streaming_disabled",
        }
      )
    );
    return;
  }

  const stream = await LLMConnector.streamGetChatCompletion(messages, {
    temperature:
      temperature ?? workspace?.openAiTemp ?? LLMConnector.defaultTemp,
  });
  const completeText = await LLMConnector.handleStream(
    responseInterceptor,
    stream,
    {
      uuid,
      sources,
    }
  );

  if (completeText?.length > 0) {
    const { chat } = await WorkspaceChats.new({
      workspaceId: workspace.id,
      prompt: String(prompt),
      response: {
        text: completeText,
        sources,
        type: chatMode,
        metrics: stream.metrics,
        attachments,
      },
    });

    writeResponseChunk(
      response,
      formatJSON(
        {
          uuid,
          type: "finalizeResponseStream",
          close: true,
          error: false,
          chatId: chat.id,
          textResponse: "",
        },
        {
          chunked: true,
          model: workspace.slug,
          finish_reason: "stop",
          usage: stream.metrics,
        }
      )
    );
    return;
  }

  writeResponseChunk(
    response,
    formatJSON(
      {
        uuid,
        type: "finalizeResponseStream",
        close: true,
        error: false,
        textResponse: "",
      },
      {
        chunked: true,
        model: workspace.slug,
        finish_reason: "stop",
        usage: stream.metrics,
      }
    )
  );
  return;
}

function formatJSON(
  chat,
  { chunked = false, model, finish_reason = null, usage = {} }
) {
  const data = {
    id: chat.uuid ?? chat.id,
    object: "chat.completion",
    created: Math.floor(Number(new Date()) / 1000),
    model: model,
    choices: [
      {
        index: 0,
        [chunked ? "delta" : "message"]: {
          role: "assistant",
          content: chat.textResponse,
        },
        logprobs: null,
        finish_reason: finish_reason,
      },
    ],
    usage,
  };

  return data;
}

module.exports.OpenAICompatibleChat = {
  chatSync,
  streamChat,
};
