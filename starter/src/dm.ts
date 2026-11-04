import { assign, createActor, fromPromise, setup } from "xstate";
import { Settings, speechstate } from "speechstate";
import { PROXY_KEY, NLU_KEY } from "./credentials";
import { DMContext, DMEvents, IdiomRecord, Message, NLUObject, IdiomExample, ExtractedRequest, DialogueIntent } from "./types";
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

// creating Qdrant client
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

    // using the LLM to extract idiom and context from user utterance
    extractRequest: fromPromise<ExtractedRequest, { messages: Message[] }>(
      async ({ input }) => {

      const response = await openai.chat.completions.create({
        model: "llama3.2:latest",
        messages: input.messages,
        response_format: { type: "json_object" },
      });

      const content = response.choices[0].message.content ?? "{}";

      console.log("LLM response:", content);

      return JSON.parse(content) as ExtractedRequest;
    }),

    // retrieval step: input: user query -> embedded, output: relevant text from qdrant
    queryRAG: fromPromise<IdiomRecord[], { query: string }>(
      async ({ input }) => {
        const embedding = await embed(input.query);

        // querying idioms collection
        const result = await qdrant.query("idioms", {
          query: embedding,
          with_payload: true,
          limit: 1,
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

    // adding interpretation (for intents)
    interpretation: null as NLUObject | null,

    // for storing ASR results for current turn -> resets before going back to listening
    lastResult: null,

    // for storing idiom under discussion
    currentIdiom: "",

    // for storing context provided by user
    userContext: "",

    // used to distiguish tasks in LLM prompt
    dialogueIntent: "explain_idiom" as DialogueIntent,

    // for storing selected RAG example (by system)
    selectedExample: null as IdiomExample | null,
    usedExampleIndices: [] as number[],

    // dialogue history across multiple turns
    messages: [
      {
        role: "system",
        content:
          "You are a helpful idiom explainer for English language learners.",
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
    }, // Prepare end
    WaitToStart: {
      // waiting until the button is clicked
      on: { 
        CLICK: {
          actions: assign({
            noInputCount: 0, // resetting no input counter at restart
          }),
          target: "Greeting",
        },
      },
    }, // WaitToStart end
    Greeting: {
      // greeting + initial user ask is analyzed
      initial: "Prompt",
      on: {
        LISTEN_COMPLETE: [
          
          // detecting intent (NLU) + recognizing utterance

          // user asks about new idiom
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "new_idiom" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "new_idiom") > 0.65,
            target: "ExtractRequest" // -> RAG pipeline
          },
          // user provides context
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "user_context" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "user_context") > 0.65,
            target: "GetContext" // ???
          },
          // user wants the system to provide a context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "request_example" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "request_example") > 0.65,
            target: "GiveContext" // ???
          },
          // user wants to hear different context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "different_example" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "different_example") > 0.65,
            target: "GiveContext" // different state ???
          },
          // user wants more in depth explanation of the current idiom
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "more_details" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "more_details") > 0.65,
            target: "GiveDetails" // ???
          },

          // handling unclear input

          { target: "Clarify",
            guard: ({ context }) => !!context.lastResult, // CHECK !!!!
          },

          // handling no input
          // ... globally (user needs to be sent back to state they came from) !!!!

          // third time no input (consecutively)
          { target: ".NoInputFinal",
            guard: ({ context }) => context.noInputCount >= 3,
          },
          // first or second time no input
          {
            target: ".NoInput",
          },
        ],
      }, // on: LISTEN_COMPLETE end
      states: {
        Prompt: { // greeting the user
          entry: { type: "spst.speak", params: { utterance: `Welcome to: What do they actually mean? - your personal idiom explainer. Which idiom is on your mind today?` } },
          on: { SPEAK_COMPLETE: "Ask" },
        }, // Prompt end
        NoInput: { // handling first and second time no input
          entry: {
            type: "spst.speak",
            params: ({ context }) => ({
                utterance:
                  context.noInputCount === 1
                    ? `I didn't hear anything. Please try again.` // if count === 1
                    : `I still didn't hear anything. Please try one more time`, // if count === 2 (ALSO IF 0 !!!)
            }),
          },
          on: { SPEAK_COMPLETE: "Ask" },
        }, // NoInput end
        NoInputFinal: { // handling third time no input
          entry: {
            type: "spst.speak",
            params: { utterance: `I didn't hear anything. Click to try again! ` }, 
          },
          on: { SPEAK_COMPLETE: "#DM.WaitToStart" }, // sending the user back to the beginning after three times of no input
        }, // NoInputFinal end
        Ask: { // handling content of user ask
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
                assign({interpretation: ({ event }) => event.nluValue ?? null}), // extracting intent
              
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
            }, // RECOGNISED end
            ASR_NOINPUT: {

              // handling no input counter
              actions:
                assign(({ context }) => ({
                    lastResult: null, // no Input -> clear lastResult
                    noInputCount: context.noInputCount + 1, // increasing the count by one
              })),
            }, // ASR_NOINPUT end
          }, // on: end
        }, // Ask end
      },
    }, // Greeting end
    Clarify: {
      // the system did not understand / the user utterance did not match any intent
      // the system asks the user to repeat their ask

      entry: {
        type: "spst.speak",
        params: {
          utterance:
            "Sorry, I didn't understand. Could you rephrase your question?",
        },
      },
      on: {
        SPEAK_COMPLETE: "Greeting.Ask", // change to whatever state it came from !!!!
      },
    }, // Clarify end
    ExtractRequest: {
      // sending the user utterance to LLM to extract the new idiom and possibly the context
      
      // setting dialogue intent (unnecessary here?) ????
      entry: assign({
        dialogueIntent: "explain_idiom"
       }),

      // prompting LLM
      invoke: {
        src: "extractRequest",
        input: ({ context }) => ({
          messages: [
            {
              role: "system" as const,
              content: `
                Extract the idiom and any situation or context provided by the user.

                Do NOT make up a context if none is provided!

                Return ONLY a JSON object with:
                - idiom: the expression the user asks about
                - context: the situation in which they heard it

                If no context is provided, use the JSON value null, NOT the string "null".
                If no idiom is identifiable, use JSON value null, NOT the string "null".
                Do not change anything about the user's context.

                Do not explain the idiom.

                Example: 
                {"idiom":"break the ice", "context":"My teacher said it yesterday."}

                Example:
                {"idiom": "hit the nail on the head", "context": null}
              `
            },
            {
              role: "user",
              content: context.lastResult?.[0]?.utterance ?? "",
            },
          ],
        }),

        // saving current idiom and user context
        onDone: [
          {
            guard: ({ event }) => !!event.output.idiom?.trim(), // going to Retrieval only if idiom was extracted
            actions: assign({
              currentIdiom: ({ event }) => event.output.idiom ?? "", // resetting/overwriting !!!!
              userContext: ({ event }) => { // resetting/overwriting !!!!
                // accounting for cases when the LLM might return "null" instead of null
                const value = event.output.context;

                if (
                  value == null ||
                  typeof value !== "string" ||
                  value.trim().toLowerCase() === "null"
                ) {
                  return "";
                }
                return value.trim();
              },

              // resetting upon successful extraction of new idiom
              dialogueIntent: "explain_idiom",
              selectedExample: null,
              usedExampleIndices: [],
              retrievedIdioms: [],
            }),
            target: "Retrieve", // ???
          },
          {
            target: "Clarify",
          },
        ],
      }, // CHECK !!!!
    }, // ExtractRequest end
    Retrieve: { 
      // invoking queryRAG to pass latest user utterance
      invoke: {
        src: "queryRAG",
        input: ({ context }) => ({
  		    query: context.currentIdiom, // querying using currentIdiom
        }),

        // keeping the latest two user utterances for retrieval !!!
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
              //currentIdiom: ({ event }) => event.output[0]?.idiom ?? "", // !!!
            }),
          
            ({ event }) => {
              console.log("Retrieved idioms:", event.output);  // control log
            },
          ],

          target: "CheckContext",
        },
      },
    }, // Retrieve end
    CheckContext: {
      // checking if the user has already provided context
      always: [
        {
          guard: ({ context }) => context.userContext.trim().length > 0,

          // setting dialogue intent
          actions: assign({
            dialogueIntent: "explain_context"
          }),

          target: "GetCompletion", // sending to LLM
        },
        {
          target: "AskContext", // if not: ask for context
        },
      ],
    }, // CheckContext end
    AskContext: {
      // user has not provided context, so system asks for context
      // before general explanation !!!

      entry: {
        type: "spst.speak",
        params: {
          utterance: "Where did you hear this expression, or how was it used?",
        },
      },
      on: {
        SPEAK_COMPLETE: "GetContext",
      },
    }, // AskContext end
    GetContext: {
      // the user provides context for the idiom
      // store context
      // give to LLM
      // (LLM uses retrieved RAG data (but does not retrieve again))

      entry: [
        assign({
          lastResult: null,
          interpretation: null,
        }),
        { type: "spst.listen" },
      ],
      on: {
        // saving utterance contents
        RECOGNISED: {
          actions: [
            assign({interpretation: ({ event }) => event.nluValue ?? null}), // extracting intent
          
            assign(({ context, event }) => ({
                lastResult: event.value, // updating lastResult with what user said
                messages: [
                  ...context.messages, // appending to convo history
                  {
                    role: "user" as const, // as user message
                    content: event.value[0].utterance,
                  },
                ],
                noInputCount: 0, // resetting no input counter if utterance recognised !!!
            })),
          ],
        },

        // deciding next dialogue step
        LISTEN_COMPLETE: [
          // user provides context
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "user_context" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "user_context") > 0.65,

            // storing user context
            actions: assign({
              userContext: ({ context }) =>
                context.lastResult?.[0]?.utterance ?? "",
              // setting dialogue intent
              dialogueIntent: "explain_context",
            }),

            target: "GetCompletion" // sending to LLM
          },

          // user does not have context !!!! train (Azure) !!!
          /**{
            guard: ({ context }) =>
              !!context.lastResult &&
              context.interpretation?.topIntent === "no_context" &&
              getIntentConfidence(context.interpretation, "no_context") > 0.65,
            target: "GiveContext",
          }, **/

          // user wants the system to provide a context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "request_example" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "request_example") > 0.65,
            target: "GiveContext"
          },

          // else:
          {
            target: "Clarify" // Add smth to allow saying no (context) + "Alright, then I will give you an example." !!!!
          },
        ],
      },

    }, // GetContext end
    GiveContext: {
      // the user wants to hear an example context from the system
      // or: the user wants to hear another example context from the system

      entry: [

          // setting dialogue intent
          assign({
            dialogueIntent: "request_example",
          }),

          assign(({ context }) => {

          // accessing retrieved Qdrant examples (filtering by currentIdiom)
          const record = context.retrievedIdioms.find(
            (item) =>
              item.idiom.toLowerCase() ===
              context.currentIdiom.toLowerCase()
          );

          // filtering out literal examples from the dataset
          const examples = (record?.example ?? []).filter((example) => {
            const meaning = example.meaning.toLowerCase();
            const explanation = example.explanation.toLowerCase();

            return (
              !meaning.includes("literal") &&
              !meaning.includes("literally") &&
              !explanation.includes("literal") &&
              !explanation.includes("literally")
            );
          });
        
          // finding first unused example
          const nextIndex = examples.findIndex(
            (_, index) => !context.usedExampleIndices.includes(index)
          );

          // storing that example (or returning null if none left)
          if (nextIndex === -1) {
            return {
              selectedExample: null, // UPGRADE !!!!
            };
          }

          // recording all selected example indices
          return {
            selectedExample: examples[nextIndex],
            usedExampleIndices: [
              ...context.usedExampleIndices,
              nextIndex,
            ],
          };
        }),

        // debugging logs
        ({ context }) => {
          console.log("GiveContext entered");
          console.log("Selected example:", context.selectedExample);
          console.log("Used indices:", context.usedExampleIndices);
        },
      ],

      always: {
        target: "GetCompletion",
      },
    }, // GiveContext end
    GiveDetails: {
      // the user wants to know more details about the idiom (register, difficulty, why it's idiomatic)

      // setting dialogue intent
      entry: assign({
        dialogueIntent: "more_details"
      }),

    }, // GiveDetails end
    GetCompletion: {
      // the LLM is prompted with the retrieved context
      invoke: {
        src: "fetchLLM",

        // making retrieved data for current idiom usable for conditional prompting
        input: ({ context }) => {
          const record = context.retrievedIdioms.find(
            (item) =>
              item.idiom.toLowerCase() ===
              context.currentIdiom.toLowerCase()
          );

          return {
          messages: [
            {
              role: "system" as const,

              // augmentation: retrieved qdrant context -> (conditional) system prompt
              content: `
                You are an English idiom tutor for language learners.
                You give short, concise answers.

                CURRENT IDIOM:
                ${context.currentIdiom}

                GENERAL MEANING:
                ${record?.meaning ?? "Unknown"}

                IDIOM EXPLANATION:
                ${record?.explanation ?? "Not available"}

                REGISTER:
                ${record?.register ?? "Unknown"}

                CURRENT TASK:
                ${context.dialogueIntent}

                ${
                  context.dialogueIntent === "request_example"
                    ? `
                      CURRENT IDIOM:
                      ${context.currentIdiom}

                      GENERAL MEANING:
                      ${record?.meaning ?? "Unknown"}

                      SELECTED EXAMPLE:
                      ${JSON.stringify(context.selectedExample, null, 2)}

                      Read the content of the line below SELECTED EXAMPLE.

                      Explain the meaning of the selected example in one or two short sentences.
                      How does the selected example relate to the general meaning of the current idiom?

                      ONLY explain!
                    `
                    : context.dialogueIntent === "explain_context"
                    ? `
                      USER'S CONTEXT:
                      ${context.userContext}

                      Explain what the idiom means in this situation.
                      Explain the speaker's likely intention.
                      Do not invent details about the situation.
                    `
                    : `
                      Explain the general meaning of the idiom.
                      Give a short, clear explanation.
                    `
                }

                Keep your response concise and suitable for spoken dialogue.
              `,
            },
            ...context.messages
          ],
        };
      },
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
    }, // GetCompletion end
    SpeakResponse: {
      // the system speaks the LLM generated response

      entry: {
        type: "spst.speak",
        params: ({ context }) => {
          const explanation =
            context.messages[context.messages.length - 1].content;

          // ensuring that the selected example is spoken exactly as in the data
          if (
            context.dialogueIntent === "request_example" &&
            context.selectedExample
          ) {
            return {
              utterance:
                `Here's an example: ${context.selectedExample.sentence} ` +
                explanation,
            };
          }
          
          // otherwise just LLM-generated text
          return {
            utterance: explanation,
          };
        },
      },
      on: {
        SPEAK_COMPLETE: {
          target: "AnythingMore",
        },
      },
    }, // SpeakResponse end
    AnythingMore: {
      entry: {
        type: "spst.speak",
        params: {
          utterance: "Anything more?"
        },
      },
      on: {
        SPEAK_COMPLETE: "Greeting.Ask",
      },
    }, // AnythingMore end
    Done: {
      entry: {
        type: "spst.speak",
        params: {
          utterance: "Thank you for exploring idioms with me! Click to ask again."
        },
      },
      on: {
        SPEAK_COMPLETE: "WaitToStart",
      },

    }, // Done end
  },
});

const dmActor = createActor(dmMachine, {}).start();

dmActor.subscribe((state) => {
  console.group("State update");

  console.log("State value:", state.value);
  console.log("State context:", state.context);
  console.log("Dialogue intent:", state.context.dialogueIntent);

  console.log("Detected intent:", state.context.interpretation?.topIntent);

  console.log("Current idiom:", state.context.currentIdiom);
  console.log("User context:", state.context.userContext);

  console.log("Selected example:", state.context.selectedExample);
  console.log("Used indices:", state.context.usedExampleIndices);

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
