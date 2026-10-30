#!/usr/bin/env node

import { Command } from "commander";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { readFile, readdir } from "node:fs/promises"; // adding readdir to list files in a folder
import { QdrantClient } from "@qdrant/js-client-rest";
import { v4 as uuidv4 } from "uuid";
import OpenAI from "openai";
import { join } from "node:path"; // so I can combine folder path and filename

const client = new QdrantClient({ host: "localhost", port: 6333 });

const openai = new OpenAI({
  baseURL: "http://localhost:11434/v1/",
  apiKey: "ollama",
  dangerouslyAllowBrowser: true,
});

const program = new Command();
program.name("npx tsx src/main.ts").description("Qdrant CLI").version("1.0.0");

export function hello(name: string, options: any) {
  const message = `Hello, ${name}!`;
  return options.uppercase ? message.toUpperCase() : message;
}

const embed = async (input: string) =>
  openai.embeddings
    .create({
      model: "qwen3-embedding",
      input: input,
      dimensions: 384,
    })
    .then((result) => result.data[0].embedding);

/** Commander demonstration */
program
  .command("hello")
  .argument("<name>")
  .action((name, options) => {
    console.log(hello(name, options));
  });

/** Chunking a document */
const makeChunksFromFile = async (filepath: string) => {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
    separators: ["\n\n\n", "\n\n", "\n", ". ", " "],
  });

  const document = await readFile(filepath, "utf8");
  const chunks = await splitter.splitText(document);
  return chunks;
};

program
  .command("split")
  .description("Split file at <path> into chunks and print.")
  .argument("<path>", "file path")
  .action(async (path, _options) => {
    const chunks = await makeChunksFromFile(path);
    chunks.forEach((c, ix) => console.log(ix, "\n", c));
  });

program
  .command("createCollection")
  .description("Create a collection")
  .argument("<name>", "collection name")
  .action(async (name) => {
    await client.createCollection(name, {
      vectors: { size: 384, distance: "Cosine" },
    });
    console.log(`Succesfully created collection: ${name}`);
  });

program
  .command("addData")
  .description("Chunk data at <path> and add it to a collection.")
  .argument("<collection>", "collection name")
  .argument("<path>", "file path")
  .action(async (collection, path) => {
    const chunks = await makeChunksFromFile(path);
    const points = await Promise.all(
      chunks.map(async (chunk) => {
        const embedding = await embed(chunk);
        return {
          id: uuidv4(),
          vector: embedding,
          payload: { text: chunk },
        };
      }),
    );
    console.log(
      `Done chunking into ${chunks.length} documents. Adding them into collection: ${collection}...`,
    );
    await client.upsert(collection, { wait: true, points: points });
    console.log(
      `Succesfully added ${chunks.length} document into collection: ${collection}`,
    );
  });

// adding data from a whole folder at once
program
  .command("addFolder")
  .description("Chunk data from multiple files in a folder at <path> and add it to a collection.")
  .argument("<collection>", "collection name")
  .argument("<path>", "folder path")
  .action(async (collection, path) => {
    const files = await readdir(path) // getting filenames
    const textFiles = files.filter((file) => file.endsWith(".txt")); // only txt files
    
    const allPoints = [] // array for collecting all the points

    for (const fileName of textFiles) { // looping over all the files in the folder
      const filePath = join(path, fileName); // getting path to file
      
      const chunks = await makeChunksFromFile(filePath);

      const pointsFile = await Promise.all(
        chunks.map(async (chunk) => {
          const embedding = await embed(chunk);
          return {
            id:uuidv4(),
            vector: embedding,
            payload: {  
              text: chunk,
              source: fileName, // added additional data
            },
          };
        }),
      );

      allPoints.push(...pointsFile) // adding file points to all points array

      console.log(
        `Done chunking ${fileName} into ${chunks.length} documents.`,
      );
    }
    
    await client.upsert(collection, { wait: true, points: allPoints });
    console.log(
      `Succesfully added ${allPoints.length} document into collection: ${collection}`,
    );
  });

// adding idioms from JSON
program
  .command("addIdioms")
  .description("Add idioms from a JSON file to a collection")
  .argument("<collection>", "collection name")
  .argument("<filepath>", "JSON file path")
  .action(async (collection, filepath) => {
      
    // reading JSON file + converting to Java objects
    const file = await readFile(filepath, "utf-8");
    const idioms = JSON.parse(file)

    // checking
    console.log(`Loaded ${idioms.length} idioms.`);
    console.log("First idiom:", idioms[0].idiom);
    console.log("Meaning:", idioms[0].meaning);
    console.log("Number of examples:", idioms[0].examples.length);

    // creating embeddings
    const batchSize = 10;

    for (let i = 0; i < idioms.length; i += batchSize) {
      const batch = idioms.slice(i, i + batchSize);
      

      const points = await Promise.all(
        batch.map(async (item, index) => {
          const embeddingText = `${item.idiom}. ${item.meaning}`;
          const embedding = await embed(embeddingText);

          return {
            id: i + index,
            vector: embedding,
            payload: {
              idiom: item.idiom,
              meaning: item.meaning,
              explanation: item.idiom_explanation,
              example: item.examples,
              register: item.register,
              difficulty: item.difficulty,
            },
          };
        }),
      );

      await client.upsert(collection, {
        wait: true,
        points: points,
      });

      console.log(
          `Imported ${Math.min(i + batchSize, idioms.length)} / ${idioms.length} idioms`
      );
    }

  });

program
  .command("queryCollection")
  .description("Query the collection")
  .argument("<collection>", "collection name")
  .argument("<query>", "text of the query")
  .action(async (collection, query) => {
    const embedding = await embed(query);
    const results = await client.query(collection, {
      with_payload: true,
      query: embedding,
      limit: 5,
    });
    console.log(results.points);
  });

program.parse();
