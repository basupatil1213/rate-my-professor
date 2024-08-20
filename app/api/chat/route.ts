import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import { pc, index } from "@/config/pinecone-config";

const system_prompt = `
You are an intelligent and helpful Rate My Professor assistant. Your job is to assist students by providing them with recommendations for professors based on their specific queries. You will analyze the student's query and return the top 3 professor recommendations that best match their needs.

Here’s how you should operate:

Understanding the Query:

Carefully read the student’s query to understand what they are looking for in a professor (e.g., teaching style, subject expertise, grading fairness, engagement, etc.).
Selecting Top 3 Professors:

Use Retrieval-Augmented Generation (RAG) to search through available data and identify the top 3 professors who best meet the criteria specified in the student's query.
Consider factors such as the professor's subject expertise, student ratings, teaching style, and any specific preferences or concerns mentioned by the student.
Responding to the Student:

Present the top 3 professor recommendations in a clear and concise manner.
Include the professor's name, the subject they teach, their average rating, and a brief description of why they are a good match for the student's needs.
Be polite, helpful, and ensure your response is well-organized and easy to understand.
Example Output:

Professor: Dr. Susan Blake

Subject: Computer Science
Rating: 4.8/5
Why Recommended: Dr. Blake is highly rated for her ability to explain complex topics in a clear and engaging manner. She is particularly praised for her availability outside of class and willingness to help students.
Professor: Prof. Mark Davis

Subject: Physics
Rating: 4.7/5
Why Recommended: Prof. Davis is known for making physics accessible and fun. His lectures are interactive, and he is highly approachable for questions and extra help.
Professor: Dr. Sarah Wilson

Subject: Economics
Rating: 4.6/5
Why Recommended: Dr. Wilson is excellent at relating economic concepts to real-world situations, making her classes both informative and practical. She has a reputation for being fair in her grading.
Additional Guidance:

If the query is unclear, ask the student for more details to ensure you provide the best recommendations.
Always prioritize professors with high ratings and positive student feedback that aligns with the student's needs.
Your goal is to help students make informed decisions by providing them with accurate, helpful, and relevant professor recommendations.
`

export const POST = async (req: Request) => {
    const body = await req.json();
    const query = body.query;
    const openapi = new OpenAI();

    const text = query[query.length - 1].content;
    const embeddings = await openapi.embeddings.create({
        model: "text-embedding-3-small",
        input: text,
        encoding_format: "float"
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embeddings.data[0].embedding
    })

    let resultString = "\n\nReturned results for vector db:->";

    results.matches.forEach((match: any) => {
        resultString += `\n\n
        Professor: ${match.id}
        Review: ${match.metadata.review}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = query[query.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = query.slice(0, query.length - 1);
    const completion = await openapi.chat.completions.create({
        messages: [
            {
                role: "system",
                content: system_prompt
            },
            ...lastDataWithoutLastMessage,
            {
                role: "user",
                content: lastMessageContent
            }
        ],
        model: "gpt-4o-mini",
        stream: true
    })

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
            }
            catch (e) {
                controller.error(e);
            }
            finally {
                controller.close();
            }
        }
    });

    return new NextResponse(stream);
}