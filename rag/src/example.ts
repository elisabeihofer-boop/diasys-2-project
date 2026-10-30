import { Command } from "commander";
import { QdrantClient } from "@qdrant/js-client-rest";
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://localhost:11434/v1/",
  apiKey: "ollama",
  dangerouslyAllowBrowser: true,
});

const client = new QdrantClient({ host: "localhost", port: 6333 });

const program = new Command();
program
  .name("npx tsx src/example.ts")
  .description("Qdrant cli-based tutorial")
  .version("1.0.0");

export function hello(name: string, options: any) {
  const message = `Hello, ${name}!`;
  return options.uppercase ? message.toUpperCase() : message;
}

program
  .command("hello")
  .argument("<name>")
  .option("--uppercase")
  .action((name, options) => {
    console.log(hello(name, options));
  });

program
  .command("createCollection")
  .argument("<name>")
  .action(async (name) => {
    await client.createCollection(name, {
      vectors: { size: 384, distance: "Cosine" },
    });
    console.log(`Successfully created collection: ${name}`);
  });

const embed = async (input: string) =>
  openai.embeddings
    .create({
      model: "qwen3-embedding",
      input: input,
      dimensions: 384,
    })
    .then((result) => result.data[0].embedding);

const menu = [
  [
    "Pad Thai with Tofu",
    "Stir-fried rice noodles with tofu, bean sprouts, scallions, and crushed peanuts in traditional tamarind sauce. ",
  ],
  [
    "Grilled Chicken Satay",
    "Marinated chicken skewers grilled over an open flame and served with creamy peanut sauce and cucumber relish.",
  ],
  [
    "Green Curry with Vegetables",
    "Thai green curry with coconut milk, seasonal vegetables, bamboo shoots, basil, and jasmine rice. ",
  ],
  [
    "Crispy Spring Rolls",
    "Golden-fried spring rolls filled with cabbage, carrots, glass noodles, and mushrooms, served with sweet chili sauce. ",
  ],
  [
    "Beef Massaman Curry",
    "Slow-cooked beef in a rich coconut curry with potatoes, onions, roasted peanuts, and aromatic spices.",
  ],
  [
    "Tom Yum Shrimp Soup",
    "Hot and sour soup with shrimp, mushrooms, lemongrass, galangal, kaffir lime leaves, and fresh chili.",
  ],
  [
    "Thai Basil Chicken",
    "Minced chicken stir-fried with garlic, chili, green beans, and Thai basil, served over steamed jasmine rice.",
  ],
  [
    "Mango Avocado Salad",
    "Fresh mango and avocado with mixed greens, cucumber, red onion, cashews, and a tangy lime dressing. ",
  ],
  [
    "Garlic Pepper Beef",
    "Tender slices of beef stir-fried with garlic, black pepper, onions, and bell peppers, served with jasmine rice.",
  ],
  [
    "Pineapple Fried Rice",
    "Jasmine rice stir-fried with pineapple, egg, cashews, carrots, peas, and scallions. ",
  ],
  [
    "Coconut Chicken Soup",
    "Creamy coconut milk soup with chicken, mushrooms, lemongrass, galangal, lime leaves, and fresh cilantro.",
  ],
  [
    "Spicy Eggplant Stir-Fry",
    "Wok-fried eggplant with tofu, bell peppers, garlic, chili, and Thai basil in a savory soy sauce. ",
  ],
  [
    "Crispy Duck with Tamarind Sauce",
    "Crispy roasted duck served with steamed vegetables and a sweet and tangy tamarind glaze.",
  ],
  [
    "Shrimp Drunken Noodles",
    "Wide rice noodles stir-fried with shrimp, chili, garlic, vegetables, and fragrant Thai basil.",
  ],
  [
    "Vegetable Red Curry",
    "Red coconut curry with tofu, zucchini, bell peppers, bamboo shoots, green beans, and Thai basil. ",
  ],
  [
    "Lemongrass Grilled Salmon",
    "Grilled salmon marinated with lemongrass, garlic, lime, and herbs, served with jasmine rice and vegetables.",
  ],
  [
    "Cashew Chicken",
    "Wok-fried chicken with roasted cashews, bell peppers, onions, scallions, and a savory chili sauce.",
  ],
  [
    "Tofu Larb",
    "Minced tofu tossed with lime juice, chili, toasted rice, red onion, mint, and fresh herbs. ",
  ],
  [
    "Thai Banana Fritters",
    "Crispy fried banana slices coated with sesame and coconut, served warm with coconut cream. ",
  ],
  [
    "Mango Sticky Rice",
    "Sweet coconut sticky rice served with fresh ripe mango and a drizzle of coconut cream. ",
  ],
];

program
  .command("addData")
  .argument("<name>")
  .action(async (name) => {
    const points = await Promise.all(
      menu.map(async (item, index) => {
        const embedding = await embed(item.join(" "));
        return {
          id: index,
          vector: embedding,
          payload: {
            name: item[0],
            note: item[1],
          },
        };
      }),
    );
    await client.upsert(name, { points });
    console.log(`Inserted ${points.length} points into a collection ${name}`);
  });

program
  .command("query")
  .argument("<name>")
  .argument("<query>")
  .action(async (name, query) => {
    const embedding = await embed(query);
    const result = await client.query(name, {
      query: embedding,
      with_payload: true,
      limit: 5,
    });
    console.log(result.points);
  });

program.parse();
