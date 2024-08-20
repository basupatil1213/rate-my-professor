import { Pinecone } from "@pinecone-database/pinecone"

export const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY!,
})

export const index = pc.index("rag").namespace("ns1")