const { QdrantClient } = require("@qdrant/js-client-rest");
const axios = require("axios");
const express = require("express");
const bodyParser = require("body-parser");
require("dotenv").config();

const app = express();
app.use(bodyParser.json());
const PORT = process.env.PORT || 3000;

// Initialize Qdrant client with compatibility check disabled
const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
  checkCompatibility: false,
});


const documents = [
  { id: 1, text: "The sun is a star at the center of the solar system." },
  { id: 2, text: "Earth is the third planet from the sun." },
  { id: 3, text: "The moon orbits Earth and affects tides." },
  { id: 4, text: "My name is rahul rai" }
];


const cohere = require("cohere-ai");
cohere.init(process.env.COHERE_API_KEY);


async function getEmbedding(texts) {
  try {
    const response = await cohere.embed({
      texts, 
      model: "embed-english-v3.0",
      input_type: "search_document",
    });

    const embeddings = response?.body?.embeddings;

    if (!embeddings || !Array.isArray(embeddings)) {
      throw new Error("No embeddings returned from Cohere.");
    }

    console.log("Embeddings length:", embeddings.length);
    return embeddings;
  } catch (err) {
    console.error(
      "❌ Error generating embeddings:",
      err.response?.body || err.message
    );
    throw err;
  }
}

// getEmbedding("Hello world from Node.js!");

async function indexDocuments() {
  const collectionName = "rag_bot_collection";

  
  try {
    await qdrantClient.getCollection(collectionName);
  } catch (error) {
    await qdrantClient.createCollection(collectionName, {
      vectors: { size: 1024, distance: "Cosine" },
    });
  }

  // Get embeddings for documents
  const texts = documents.map((doc) => doc.text);
  const embeddings = await getEmbedding(texts);

  // Prepare points for Qdrant
  const points = documents.map((doc, index) => ({
    id: doc.id,
    vector: embeddings[index],
    payload: { text: doc.text },
  }));

  // Upsert points to Qdrant
  await qdrantClient.upsert(collectionName, { points });
  console.log("Documents indexed successfully");
}

// Function to retrieve relevant documents
async function retrieveDocuments(query, limit = 2) {
  const queryEmbedding = await getEmbedding([query]);
  const collectionName = "rag_bot_collection";

  const searchResults = await qdrantClient.search(collectionName, {
    vector: queryEmbedding[0],
    limit,
    with_payload: true,
  });

  return searchResults.map((result) => result.payload.text);
}

// Function to generate response using Hugging Face LLM

const COHERE_API_URL = "https://api.cohere.ai/generate";

async function generateResponse(query, context) {
  const prompt = `
You are a helpful assistant. Use the following context to answer the user's query accurately. If the context doesn't fully cover the query, provide a general answer based on your knowledge, but prioritize the context.

**Context**:
${context.join("\n")}

**Query**:
${query}

**Answer**:
`;

  try {
    const response = await axios.post(
      COHERE_API_URL,
      {
        prompt: prompt,
        max_tokens: 150,
        temperature: 0.7,
        k: 0,
        p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.COHERE_API_KEY}`,
          "Content-Type": "application/json",
          "Cohere-Version": "2022-12-06" 
        },
      }
    );

    // ✅ Check if generations and text exist
    if (
      response?.data?.generations &&
      Array.isArray(response.data.generations) &&
      response.data.generations.length > 0 &&
      response.data.generations[0].text
    ) {
      return response.data.generations[0].text.trim();
    } else {
      console.error("Cohere API returned unexpected response:", response.data);
      return "I'm sorry, I couldn't generate a response at the moment.";
    }

  } catch (error) {
    console.error("Error generating response:", error.response?.data || error.message);
    return "There was an error generating the response.";
  }
}




async function handleQuery(query) {
  try {
    // Retrieve relevant documents
    const context = await retrieveDocuments(query);
    console.log("Retrieved context:", context);

    // Generate response
    const response = await generateResponse(query, context);
    console.log("Response:", response);
    return response;
  } catch (error) {
    console.error("Error handling query:", error);
    throw new Error("Sorry, an error occurred while processing your query.");
  }
}

// API endpoint to handle queries
app.post("/query", async (req, res) => {
  const { query } = req.body;

  if (!query || typeof query !== "string") {
    return res
      .status(400)
      .json({ error: "Query is required and must be a string" });
  }

  try {
    const response = await handleQuery(query);
    res.json({ query, response });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get("/health", async (req, res) => {
  let qdrantConnected = false;
  try {
    await qdrantClient.getCollections();
    qdrantConnected = true;
  } catch (error) {
    console.error("Health check Qdrant error:", error.message);
  }
  res.json({ status: "Server running", qdrantConnected });
});

// Start the server and index documents
async function startServer() {
  try {
    console.log("Attempting to connect to Qdrant at:", process.env.QDRANT_URL);
    const collections = await qdrantClient.getCollections();
    console.log("Connected to Qdrant successfully. Collections:", collections);
    await indexDocuments();
    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
    });
  } catch (error) {
    console.error("Failed to initialize Qdrant:", error.message);
    if (error.response) {
      console.error("Qdrant response:", error.response.data);
      console.error("Qdrant status:", error.response.status);
      console.error("Qdrant headers:", error.response.headers);
    }
    console.log(
      `Server will start without Qdrant. Use /health to check status.`
    );
    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
    });
  }
}

startServer();
