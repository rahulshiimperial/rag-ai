const fs = require('fs').promises;
const axios = require('axios');
require('dotenv').config();

const { client, COLLECTION_NAME } = require('./qdrant');
const { generateEmbedding } = require('./embeddings');
const { chunkText } = require('./utils');

  const HF_COMPLETION_MODEL = 'mistralai/Mistral-7B-Instruct-v0.1';
const HF_CHAT_URL = `https://api-inference.huggingface.co/models/${HF_COMPLETION_MODEL}`;



async function indexData(filePath) {
  try {
    
    const fileContent = await fs.readFile(filePath, 'utf-8');
    const chunks = chunkText(fileContent);
    console.log(chunks)
    // Generate points with embeddings
    const points = await Promise.all(
      chunks.map(async (chunk, idx) => {
        const vector = await generateEmbedding("what is javascript ? why it is used ?");
        return {
          id: idx,
          vector,
          payload: { text: chunk },
        };
      })
    );

    // Upload to Qdrant
    await client.upsert(COLLECTION_NAME, { points });

    console.log('✅ Data indexed successfully');
  } catch (error) {
    console.error('❌ Failed to index data:', error);
    throw error;
  }
}

async function generateResponseFromHuggingFace(prompt) {
  try {
    const response = await axios.post(
      HF_CHAT_URL,
      {
        inputs: prompt,
        parameters: {
          max_new_tokens: 300,
          return_full_text: false,
        },
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
        },
      }
    );

    const generated = response.data[0]?.generated_text || 'No response generated';
    return generated.replace(prompt, '').trim();
  } catch (error) {
    console.error('❌ Hugging Face generation error:', error.response?.data || error.message);
    throw error;
  }
}

async function queryChatbot(userQuery) {
  try {
    const queryEmbedding = await generateEmbedding(userQuery);

    const searchResults = await client.search(COLLECTION_NAME, {
      vector: queryEmbedding,
      limit: 3,
      with_payload: true,
    });

    const context = searchResults.map(result => result.payload.text).join('\n');

    const prompt = `Use the context below to answer the question.\n\nContext:\n${context}\n\nQuestion: ${userQuery}\nAnswer:`;

    const response = await generateResponseFromHuggingFace(prompt);
    return response;
  } catch (error) {
    console.error('❌ Error processing query:', error);
    throw error;
  }
}

module.exports = { indexData, queryChatbot };
