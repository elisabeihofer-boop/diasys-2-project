import { assign, createActor, fromPromise, setup } from "xstate";
import { Settings, speechstate } from "speechstate";
import { PROXY_KEY, NLU_KEY } from "./credentials";
import { DMContext, DMEvents, IdiomRecord, Message, NLUObject, IdiomExample } from "./types"; // added Message
import OpenAI from "openai";
import { QdrantClient } from "@qdrant/js-client-rest"; // importing qdrant

const REGION = "northeurope";

const openai = new OpenAI({
  baseURL: "http://localhost:11434/v1/",
  apiKey: "ollama",
  dangerouslyAllowBrowser: true,
});

// embedding helper function: converting text into 384-dimensional vector
const embed = async (input: string) =>
  openai.embeddings
    .create({
      model: "qwen3-embedding",
      input,
      dimensions: 384,
    })
    .then((result) => result.data[0].embedding);

// creating qdrant client
const qdrant = new QdrantClient({
  host: "localhost",
  port: 6333,
});

// Azure access via FLoV proxy
const azureProxyCredentials = {
  proxyUrl: "https://rndserv.flov.gu.se:4000/api/token",
  key: PROXY_KEY,
};

// NLU access-> needed for intent
const azureLanguageCredentials = {
  endpoint: "https://lab-gusbeihel.cognitiveservices.azure.com/language/:analyze-conversations?api-version=2024-11-15-preview",
  key: NLU_KEY, 
  deploymentName: "idiom_explainer",
  projectName: "idiom_explainer",
  };

const settings: Settings = {
  azureLanguageCredentials: azureLanguageCredentials,
  azureCredentials: azureProxyCredentials,
  azureRegion: REGION,
  asrDefaultCompleteTimeout: 0,
  asrDefaultNoInputTimeout: 5000,
  locale: "en-US",
  ttsDefaultVoice: "en-US-DavisNeural",
  bargeIn: false, // ENABLE ????
};

// checking the confidence score for the intents
function getIntentConfidence(nlu: NLUObject | null, intent: string): number {
  return nlu?.intents.find(i => i.category === intent)?.confidenceScore ?? 0;
};

const dmMachine = setup({
  types: {
    context: {} as DMContext,
    events: {} as DMEvents,
  },
  actions: {
    "spst.speak": ({ context }, params: { utterance: string }) =>
      context.spstRef.send({
        type: "SPEAK",
        value: {
          utterance: params.utterance,
        },
      }),
    "spst.listen": ({ context }) =>
      context.spstRef.send({
        type: "LISTEN",
        value: { nlu: true }, // local activation of NLU
      }),
  },
  actors: {

    // generation step: current dialogue history -> LLM + returns text response
    fetchLLM: fromPromise<string, { messages: Message[] }>(
      async ({ input }) => {
        //console.log("Actor input:", input);

        const response = await openai.chat.completions.create({
          model: "llama3.2:latest",
          messages: input.messages,
        });

      //console.log("LLM response:", response);

      return response.choices[0].message.content ?? "";
      },
    ),

    // retrieval step: input: user query -> embedded, output: relevant text from qdrant
    queryRAG: fromPromise<IdiomRecord[], { query: string }>(
      async ({ input }) => {
        const embedding = await embed(input.query);

        // querying idioms collection
        const result = await qdrant.query("idioms", {
          query: embedding,
          with_payload: true,
          limit: 3,
        });
      
        // converting returned payloads into array
        const retrievedIdioms: IdiomRecord[] = result.points
          .filter((point) => point.payload != null)
          .map((point) => {
            const payload = point.payload!;

            return {
              idiom: String(payload.idiom ?? ""),
              meaning: String(payload.meaning ?? ""),
              explanation: String(payload.explanation ?? ""),
              register: String(payload.register ?? ""),
              difficulty: String(payload.difficulty ?? ""),
              example: Array.isArray(payload.example)
                ? payload.example as unknown as IdiomExample[]
                : [],
            };
          });

        //console.log("Qdrant payload:", retrievedIdioms);

        return retrievedIdioms;

      },
    ),
  },
}).createMachine({
  context: ({ spawn }) => ({
    spstRef: spawn(speechstate, { input: settings }),

    // Adding interpretation
    interpretation: null as NLUObject | null,

    // storing ASR results for current turn -> resets before going back to listening
    lastResult: null,

    // storing idiom under discussion
    currentIdiom: "",

    // dialogue history across multiple turns
    messages: [
      {
        role: "system",
        content:
          "You are a helpful assistant for University of Gothenburg students.",
      },
    ] as Message[],
    retrievedIdioms: [] as IdiomRecord[], // to store converted Qdrant arrays for current query
    noInputCount: 0, // counting how often there is no input from the user
  }),
  id: "DM",
  initial: "Prepare",
  states: {
    Prepare: {
      entry: ({ context }) => context.spstRef.send({ type: "PREPARE" }),
      on: { ASRTTS_READY: "WaitToStart" },
    },
    WaitToStart: {
      on: { 
        CLICK: {
          actions: assign({
            noInputCount: 0, // resetting no input counter at restart
          }),
          target: "Greeting",
        },
      },
    },
    Greeting: {
      initial: "Prompt",
      on: {
        LISTEN_COMPLETE: [

          
          // detecting intent + recognizing utterance
          // recognition successful -> RAG pipeline
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "new_idiom" &&
            // checking for confidence (should be over 60%)
              getIntentConfidence(context.interpretation, "new_idiom") > 0.6,
            target: "Retrieve"
          },

          // handling no input
          // third time no input (consecutively)
          { target: ".NoInputFinal",
            guard: ({ context }) => context.noInputCount >= 3,
          },
          // first or second time no input
          {
            target: ".NoInput",
          },
        ],
      },
      states: {
        Prompt: {
          entry: { type: "spst.speak", params: { utterance: `Welcome to: What do they actually mean? - your personal idiom explainer` } },
          on: { SPEAK_COMPLETE: "Ask" },
        },
        NoInput: {
          entry: {
            type: "spst.speak",
            params: ({ context }) => ({
                utterance:
                  context.noInputCount === 1
                    ? `I didn't hear anything. Please try again.` // if count === 1
                    : `I still didn't hear anything. Please try one more time`, // if count === 2
            }),
          },
          on: { SPEAK_COMPLETE: "Ask" },
        },
        NoInputFinal: {
          entry: {
            type: "spst.speak",
            params: { utterance: `I didn't hear anything. Click to try again! ` }, 
          },
          on: { SPEAK_COMPLETE: "#DM.WaitToStart" }, // sending the user back to the beginning after three times of no input
        },
        Ask: {
          entry: [
            assign({ 
              lastResult: null, // resetting lastResult
              interpretation: null, // resetting intent
            }), 
            { type: "spst.listen" },
          ],
          on: {
            RECOGNISED: {
              actions: [
                assign({interpretation: ({ event }) => event.nluValue ?? null}),
              
                assign(({ context, event }) => {
                  const utterance = event.value[0].utterance; // extracting what the user said

                  return {
                    lastResult: event.value,
                    messages: [
                      ...context.messages, // appending to convo history
                      {
                        role: "user" as const, // as user message
                        content: utterance,
                      },
                    ],
                    noInputCount: 0, // resetting no input counter if utterance recognised
                  };
                }),
              ],
            },
            ASR_NOINPUT: {

              // handling no input
              actions:
                assign(({ context }) => ({
                    lastResult: null, // no Input -> clear lastResult
                    noInputCount: context.noInputCount + 1, // increasing the count by one
              })),
            },
          },
        },
      },
    },
    Retrieve: { // invoking queryRAG to pass latest user utterance
      invoke: {
        src: "queryRAG",
        input: ({ context }) => ({
  		    query: context.lastResult![0].utterance,
        }),

        // keeping the latest two user utterances for retrieval
        /*
        input: ({ context }) => {
          // keeping only user messages
          const userMessages = context.messages.filter(
            (message) => message.role === "user",
          );

          // taking the two most recent user messages
          const recentUserMessages = userMessages.slice(-2);

          // combining them
          const query = recentUserMessages
            .map((message) => message.content)
            .join("\n");

          // control log
          console.log("RAG query:", query);

          return {
            query: query,
          };
        },
        */
        onDone: {
          actions: [
            
            // storing the retrieved info (for augmentation step)
            assign({
              retrievedIdioms: ({ event }) => event.output,
              currentIdiom: ({ event }) => event.output[0]?.idiom ?? "",
            }),
          
            ({ event }) => {
              console.log("Retrieved idioms:", event.output);  // control log
            },
          ],

          target: "GetCompletion",
        },
      },
    },
    GetCompletion: {
      invoke: {
        src: "fetchLLM",
        input: ({ context }) => ({
          messages: [
            {
              role: "system" as const,

              // augmentation step: retrieved qdrant context -> system prompt
              content: `
                You are an assistant for language learners of English. You explain idioms.
                
                Use the retrieved information below to answer the user's question.
                If the answer is not supported by the retrieved information, 
                please let the user know that you do not have enough information to answer the question.

                RETRIEVED IDIOM INFORMATION START

                ${JSON.stringify(context.retrievedIdioms, null, 2)}

                RETRIEVED IDIOM INFORMATION END

                Explain the idiom in simple English.
                If the user provided context, explain what the idiom means in that context.

                The retrieved examples may contain both literal and idiomatic uses of the expression.
                Use the user's context (if available) to determine which interpretation is appropriate.

                When providing an example of idiomatic usage, prefer examples that clearly demonstrate the figurative meaning.
                Do not assume every retrieved example is idiomatic.

                Keep the responses concise and efficient.
              `,
            },
            ...context.messages
          ],
        }),
        onDone: {
          actions: assign({
            messages: ({ context, event }) => [
              ...context.messages,
              {
                role: "assistant",
                content: event.output,
              },
            ],
        }),
        target: "SpeakResponse",
        },
      },
    },
    SpeakResponse: {
      entry: {
        type: "spst.speak",
        params: ({ context }) => ({
          utterance: context.messages[context.messages.length - 1].content,
        }),
      },
      on: {
        SPEAK_COMPLETE: {
          target: "Greeting.Ask",
        },
      },
    },
  },
});

const dmActor = createActor(dmMachine, {}).start();

dmActor.subscribe((state) => {
  console.group("State update");
  console.log("State value:", state.value);
  console.log("State context:", state.context);
  console.log("No-input count:", state.context.noInputCount); // checking the no input counter
  console.groupEnd();
});

export function setupButton(element: HTMLButtonElement) {
  element.addEventListener("click", () => {
    dmActor.send({ type: "CLICK" });
  });
  dmActor.subscribe((snapshot) => {
    const meta: { view?: string } = Object.values(
      snapshot.context.spstRef.getSnapshot().getMeta(),
    )[0] || {
      view: undefined,
    };
    element.innerHTML = `${meta.view}`;
  });
}
