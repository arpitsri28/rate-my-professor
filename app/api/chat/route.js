import { NextResponse } from "next/server"
import { Pinecone } from "@pinecone-database/pinecone"
import OpenAI from "openai"
import { SYSTEM_ENTRYPOINTS } from "next/dist/shared/lib/constants"
import fetch from 'node-fetch'

const systemPrompt = `
Welcome to the Rate My Professor Assistant! I can help you find information about professors based on your specific needs. Just type your query about a professor's teaching style, courses, research, or any other aspect, and I will provide you with the top three professors who match your criteria.

Please enter your query about the professor or course you are interested in:

For example, you can ask:
- "Who is the best professor for machine learning in engineering?"
- "Can you recommend a professor known for an interactive teaching style in biology?"
- "I need a professor who gives detailed feedback on assignments in economics."

Based on your query, here are the top three professors:

1. Dr. Jane Smith - Engineering
   - Expertise in Machine Learning and Data Science.
   - Known for a hands-on approach and in-depth lectures.
   - Rated highly for engaging course materials and supportive teaching style.

2. Dr. John Doe - Biology
   - Specializes in Molecular Biology and Genetics.
   - Students appreciate his interactive labs and accessible teaching methods.
   - Frequently collaborates with students on research projects.

3. Dr. Alice Johnson - Economics
   - Focuses on Economic Theory and Policy.
   - Praised for clear explanations and detailed feedback on assignments.
   - Offers valuable insights into real-world economic applications.

If you need more detailed information about any of these professors or have another query, feel free to ask!
`

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY
  });
  const index = pc.index('rag2').namespace('ns1');

  const text = data[data.length -1].content;

  // Fetch the embedding from your FastAPI service
  const response = await fetch('http://localhost:8000/embeddings/', { 
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: text})
  });

  if (!response.ok) {
    throw new Error('Failed to fetch embeddings from FastAPI');
  }

  const embeddingResult = await response.json();
  const embedding = embeddingResult.probabilities; // Ensure this matches the output of your FastAPI

  // Query the Pinecone index with the embedding
  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding
  });

  let resultString = '\n\nReturned results from vector db (done automatically):';
  results.matches.forEach((match) => (
    resultString += ` \n
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n
    `
  ));

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  const completion = await openai.chat.completions.create({
    messages: [
      {role: 'system', content: systemPrompt},
      ...lastDataWithoutLastMessage,
      {role: 'user', content: lastMessageContent}
    ],
    model: 'gpt-4o-mini',
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
