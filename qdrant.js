const { QdrantClient } = require('@qdrant/js-client-rest');
const axios = require('axios');
const express = require('express');
const bodyParser = require('body-parser');
require('dotenv').config();

// Debug environment variables
console.log('QDRANT_API_KEY length:', process.env.QDRANT_API_KEY ? process.env.QDRANT_API_KEY.length : 'NOT SET');
console.log('QDRANT_API_KEY first 10 chars:', process.env.QDRANT_API_KEY ? process.env.QDRANT_API_KEY.slice(0, 10) : 'NOT SET');
console.log('Environment variables:', {
  QDRANT_URL: process.env.QDRANT_URL,
  QDRANT_API_KEY: process.env.QDRANT_API_KEY ? '[REDACTED]' : 'NOT SET',
  HF_API_KEY: process.env.HF_API_KEY ? '[REDACTED]' : 'NOT SET',
  PORT: process.env.PORT,
});

// Test Qdrant connection manually
async function testQdrantConnection() {
  try {
    const headers = { 'api-key': process.env.QDRANT_API_KEY };
    console.log('Axios request headers:', { 'api-key': process.env.QDRANT_API_KEY ? '[REDACTED]' : 'NOT SET' });
    const response = await axios.get(process.env.QDRANT_URL, { headers });
    console.log('Manual Qdrant test response:', response.data);
  } catch (error) {
    console.error('Manual Qdrant test failed:', error.message);
    if (error.response) {
      console.error('Error details:', error.response.data);
      console.error('Error status:', error.response.status);
      console.error('Error headers:', error.response.headers);
    }
  }
}
testQdrantConnection();

// Initialize Express app
const app = express();
app.use(bodyParser.json());
const PORT = process.env.PORT || 3000;

// Initialize Qdrant client
const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
  checkCompatibility: false,
});

// Use a more standard sentence-transformers model
const HF_API_URL_EMBEDDINGS = 'https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2';
const HF_API_URL_LLM = 'https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct';
const HF_API_KEY = process.env.HF_API_KEY;

// Sample documents to index
const documents = [
  { id: 1, text: "The sun is a star at the center of the solar system." },
  { id: 2, text: "Earth is the third planet from the sun." },
  { id: 3, text: "The moon orbits Earth and affects tides." },
];

// Function to get embeddings from Hugging Face
async function getEmbeddings(texts) {
  try {
    // Special format required for paraphrase models
    const payload = {
      inputs: {
        source_sentence: texts[0],
        sentences: texts.length > 1 ? texts.slice(1) : [texts[0]]
      }
    };
    
    const response = await axios.post(
      HF_API_URL_EMBEDDINGS,
      payload,
      {
        headers: {
          Authorization: `Bearer ${HF_API_KEY}`,
          'Content-Type': 'application/json',
        },
      }
    );
    
    // The response will be similarity scores, not embeddings
    // So this approach won't work for getting standalone embeddings
    console.log('Similarity scores:', response.data);
    throw new Error('This model returns similarity scores, not embeddings');
    
  } catch (error) {
    console.error('Error with paraphrase model:', error.message);
    throw error;
  }
}

// Function to index documents in Qdrant
async function indexDocuments() {
  const collectionName = 'rag_bot_collection';

  // Create collection if it doesn't exist
  try {
    await qdrantClient.getCollection(collectionName);
  } catch (error) {
    await qdrantClient.createCollection(collectionName, {
      vectors: { size: 384, distance: 'Cosine' },
    });
  }

  // Get embeddings for documents
  const texts = documents.map(doc => doc.text);
  const embeddings = await getEmbeddings(texts);

  // Prepare points for Qdrant
  const points = documents.map((doc, index) => ({
    id: doc.id,
    vector: embeddings[index],
    payload: { text: doc.text },
  }));

  // Upsert points to Qdrant
  await qdrantClient.upsert(collectionName, { points });
  console.log('Documents indexed successfully');
}

// Function to retrieve relevant documents
async function retrieveDocuments(query, limit = 2) {
  try {
    // Get embedding for the query
    const queryEmbedding = await getEmbeddings([query]);
    const collectionName = 'rag_bot_collection';

    const searchResults = await qdrantClient.search(collectionName, {
      vector: queryEmbedding[0],
      limit,
      with_payload: true,
    });

    return searchResults.map(result => result.payload.text);
  } catch (error) {
    console.error("Error retrieving documents:", error.message);
    return []; // Return empty context if retrieval fails
  }
}

// Function to generate response using Hugging Face LLM
async function generateResponse(query, context) {
  const prompt = `
You are a helpful assistant. Use the following context to answer the user's query accurately. If the context doesn't fully cover the query, provide a general answer based on your knowledge, but prioritize the context.

**Context**:
${context.join('\n')}

**Query**:
${query}

**Answer**:
`;

  try {
    const response = await axios.post(
      HF_API_URL_LLM,
      {
        inputs: prompt,
        parameters: { max_new_tokens: 150, temperature: 0.7 },
      },
      {
        headers: {
          Authorization: `Bearer ${HF_API_KEY}`,
          'Content-Type': 'application/json',
        },
      }
    );
    return response.data[0].generated_text.trim();
  } catch (error) {
    console.error('Error generating response:', error.message);
    throw error;
  }
}

// Function to handle user query
async function handleQuery(query) {
  try {
    // Retrieve relevant documents
    const context = await retrieveDocuments(query);
    console.log('Retrieved context:', context);

    // Generate response
    const response = await generateResponse(query, context);
    console.log('Response:', response);
    return response;
  } catch (error) {
    console.error('Error handling query:', error.message);
    throw new Error('Sorry, an error occurred while processing your query.');
  }
}

// API endpoint to handle queries
app.post('/query', async (req, res) => {
  const { query } = req.body;

  if (!query || typeof query !== 'string') {
    return res.status(400).json({ error: 'Query is required and must be a string' });
  }

  try {
    const response = await handleQuery(query);
    res.json({ query, response });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get('/health', async (req, res) => {
  let qdrantConnected = false;
  let qdrantError = null;
  let hfConnected = false;
  let hfError = null;
  
  // Test Qdrant connection
  try {
    await qdrantClient.getCollections();
    qdrantConnected = true;
  } catch (error) {
    qdrantError = error.message;
    console.error('Health check Qdrant error:', error.message);
  }
  
  // Test Hugging Face connection
  try {
    const response = await axios.post(
      HF_API_URL_EMBEDDINGS,
      { inputs: ["This is a test"] },
      {
        headers: {
          Authorization: `Bearer ${HF_API_KEY}`,
          'Content-Type': 'application/json',
        },
      }
    );
    hfConnected = true;
  } catch (error) {
    hfError = error.message;
    console.error('Health check Hugging Face error:', error.message);
  }
  
  res.json({ 
    status: 'Server running', 
    qdrantConnected, 
    qdrantError, 
    hfConnected, 
    hfError 
  });
});

// Start the server and index documents
async function startServer() {
  try {
    console.log('Attempting to connect to Qdrant at:', process.env.QDRANT_URL);
    const collections = await qdrantClient.getCollections();
    console.log('Connected to Qdrant successfully. Collections:', collections);
    await indexDocuments();
    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
    });
  } catch (error) {
    console.error('Failed to initialize Qdrant:', error.message);
    if (error.response) {
      console.error('Qdrant response:', error.response.data);
      console.error('Qdrant status:', error.response.status);
      console.error('Qdrant headers:', error.response.headers);
    }
    console.log(`Server will start without Qdrant. Use /health to check status.`);
    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
    });
  }
}

startServer();