const axios = require('axios');
require('dotenv').config();

const HF_API_URL = 'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2';

async function generateEmbedding(text) {
  try {
    // Handle array input by extracting the first string
    const inputText = Array.isArray(text) ? text[0] : text;
    if (typeof inputText !== 'string' || inputText.trim() === '') {
      throw new Error('Input must be a non-empty string');
    }

    const response = await axios.post(
      HF_API_URL,
      { inputs: [inputText] }, // Use array of strings for feature extraction
      {
        headers: {
          Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    // Response is [[vector]] for single input array
    if (!Array.isArray(response.data) || !Array.isArray(response.data[0]) || response.data[0].length !== 384) {
      throw new Error(`Invalid embedding response: ${JSON.stringify(response.data).slice(0, 100)}`);
    }

    const embedding = response.data[0]; // Extract the first vector
    return embedding; // 384-dimensional vector
  } catch (error) {
    console.error('Error generating embedding:', error.response?.data || error.message);
    throw error;
  }
}

module.exports = { generateEmbedding };