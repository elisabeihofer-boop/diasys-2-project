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
  deploymentName: "IdiomExplainer",
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

    // has the system already delivered an explanation?
    hasExplainedIdiom: false,

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

      // resetting dialogue context for next session
      entry: assign({
        currentIdiom: "",
        userContext: "",
        retrievedIdioms: [],
        selectedExample: null,
        usedExampleIndices: [],
        dialogueIntent: "explain_idiom",
        hasExplainedIdiom: false,

        interpretation: null,
        lastResult: null,
        noInputCount: 0,

        messages: [
          {
            role: "system",
            content: "You are a helpful idiom explainer for English language learners.",
          },
        ],
      }),

      // waiting until the button is clicked
      on: { 
        CLICK: {
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

            // setting dialogue intent
            actions: assign({
              dialogueIntent: "explain_idiom",

              // preserving context if no idiom stored (context provided first)
              // clearing (old) context if idiom already stored (new idiom)
              // !!!!
              userContext: ({ context }) =>
                context.currentIdiom ? "" : context.userContext,
            }),  

            target: "ExtractRequest" // -> RAG pipeline directly
          },

          // user provides context
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "user_context" &&
              getIntentConfidence(context.interpretation, "user_context") > 0.65,
            
            // saving utterance as user context immediately
            actions: assign({
              userContext: ({ context }) =>
                context.lastResult?.[0]?.utterance ?? "",
              dialogueIntent: "explain_context", // setting dialogue intent
            }),
              
            target: "CheckIdiom" // ??? (changed from GetContext)
          },

          // user wants the system to provide a context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "request_example" &&
              getIntentConfidence(context.interpretation, "request_example") > 0.65,
            
            // setting dialogue intent
            actions: assign({
              dialogueIntent: "request_example",
            }),
            
            target: "CheckIdiom" // ??? (changed from GiveContext)
          },

          // user wants to hear different context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "different_example" &&
              getIntentConfidence(context.interpretation, "different_example") > 0.65,
            
            // setting dialogue intent
            actions: assign({
              dialogueIntent: "request_example",
            }),
            
            target: "CheckIdiom" // ??? (changed from GiveContext)
          },

          // user wants more in depth explanation of the current idiom
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "more_details" &&
              getIntentConfidence(context.interpretation, "more_details") > 0.65,
            
            // setting dialogue intent
            actions: assign({
              dialogueIntent: "more_details"
            }),

            target: "CheckIdiom" // ??? (changed from GiveDetails)
          },

          // user says no (from AnythingMore)
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "negation" &&
              getIntentConfidence(context.interpretation, "negation") > 0.65,
            target: "Done"
          },

          // handling unclear input
          { target: "Clarify",
            guard: ({ context }) => !!context.lastResult, // CHECK !!!!
          },

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
          entry: { type: "spst.speak", params: { utterance: `Welcome to Idiomagic - your personal idiom explainer. Which idiom is on your mind today?` } },
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
                    interpretation: null, // clear interpretation !!!!
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
            "Sorry, I didn't understand. Could you rephrase that?",
        },
      },
      on: {
        SPEAK_COMPLETE: "Greeting.Ask", // change to whatever state it came from !!!!
      },
    }, // Clarify end
    ExtractRequest: {
      // sending the user utterance to LLM to extract the new idiom and possibly the context

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
                NEVER infer a context from the idiom's meaning.

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
              userContext: ({ context, event }) => { // resetting/overwriting !!!!
                // accounting for cases when the LLM might return "null" instead of null
                
                // prioritizing previously saved context (from Greeting; in case context came before idiom) !!!

                const value = event.output.context;

                if (
                  value == null ||
                  typeof value !== "string" ||
                  value.trim().toLowerCase() === "null"
                ) {
                  return context.userContext;
                }

                return context.userContext || value.trim();
              },

              // resetting upon successful extraction of new idiom
              selectedExample: null,
              usedExampleIndices: [],
              retrievedIdioms: [],
              hasExplainedIdiom: false,
            }),
            target: "Retrieve", // ???
          },
          {
            target: "Clarify",
          },
        ],
        // in case something went wrong with invoking the LLM
        onError: {
          actions: ({ event }) => {
            console.error("Extraction failed:", event.error);
          },
          target: "Error",
        },
      }, // CHECK !!!!
    }, // ExtractRequest end
    Retrieve: { 
      // invoking queryRAG to pass latest user utterance
      invoke: {
        src: "queryRAG",
        input: ({ context }) => ({
  		    query: context.currentIdiom, // querying using currentIdiom
        }),

        onDone: [

          // storing the retrieved info (for augmentation step)
          {
            guard: ({ context }) =>
              context.dialogueIntent === "request_example",

            actions: assign({
              retrievedIdioms: ({ event }) => event.output,
            }),

            target: "GiveContext", // giving the user an example if originally detected intent was request_example or different_example
          },
          {
            actions: assign({
              retrievedIdioms: ({ event }) => event.output,
            }),

            target: "CheckContext",
          },
        ],

        onError: {
          actions: ({ event }) => {
            console.error("Retrieval failed:", event.error);
          },
          target: "Error",
        },
      },
    }, // Retrieve end
    CheckIdiom: {
      // checking if the user has provided an idiom (and an idiom has been retrieved)

      always: [

        // no idiom provided
        {
          guard: ({ context }) => 
            !context.currentIdiom.trim() ||
            context.retrievedIdioms.length === 0,

          target: "AskIdiom", // asking for idiom
        },

        // user wants an example
        {
          guard: ({ context }) =>
            context.interpretation?.topIntent === "request_example" ||
            context.interpretation?.topIntent === "different_example",

          target: "GiveContext", // sending to select example
        },

        // user wants more details
        {
          guard: ({ context }) =>
            context.interpretation?.topIntent === "more_details",

          target: "GiveDetails",
        },

        // otherwise: continue on default pipeline
        {
          target: "CheckContext",
        }
      ],
    }, // CheckIdiom end
    AskIdiom: {
      // the user has not yet provided an idiom, so the system asks for it

      entry: {
        type: "spst.speak",
        params: {
          utterance: "Which idiom would you like to know about?",
        },
      },
      on: {
        SPEAK_COMPLETE: "GetIdiom", // sending to get idiom
      },
    }, // AskIdiom end
    GetIdiom: {
      // user provides idiom later in the dialogue

      entry: [
        assign({
          lastResult: null,
          interpretation: null,
        }),
        { type: "spst.listen" },
      ],

      on: {
        RECOGNISED: {
          actions: [
            assign({ interpretation: ({ event }) => event.nluValue ?? null }),
            
            assign(({ context, event }) => ({
              lastResult: event.value,
              messages: [
                ...context.messages,
                {
                  role: "user" as const,
                  content: event.value[0].utterance,
                },
              ],
              noInputCount: 0,
            })),
          ],
        },

        LISTEN_COMPLETE: [
          {
            guard: ({ context }) => !!context.lastResult &&
              context.interpretation?.topIntent === "new_idiom" &&
              getIntentConfidence(context.interpretation, "new_idiom") > 0.65,

            target: "ExtractRequest",
          },
          {
            target: "Clarify", // sending back to beginning basically
          },
        ],
      },
    }, // GetIdiom end
    CheckContext: {
      // checking if the user has already provided context
      always: [
        {
          guard: ({ context }) => context.userContext.trim().length > 0,

          // setting dialogue intent !!!
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
      // before general explanation by LLM

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
      // give to LLM (LLM uses retrieved RAG data (but does not retrieve again))

      initial: "Listen",

      on: {
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

          // user does not have context
          {
            guard: ({ context }) =>
              !!context.lastResult &&
              context.interpretation?.topIntent === "no_context" &&
              getIntentConfidence(context.interpretation, "no_context") > 0.65,
            target: "ConfirmExample",
          },

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
            guard: ({ context }) => !!context.lastResult,
            target: "Clarify"
          },

          // third consecutive time no input
          {
            guard: ({ context }) =>
              !context.lastResult && context.noInputCount >= 3,
            target: ".NoInputFinal",
          },

          // first or second no input
          {
            guard: ({ context }) => !context.lastResult,
            target: ".NoInput",
          },
        ],
      },

      states: {
        Listen: {
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

            // or handling no input
            ASR_NOINPUT: {
              // handling no input counter
              actions: assign(({ context }) => ({
                    lastResult: null, // no Input -> clear lastResult
                    interpretation: null, // clear interpretation
                    noInputCount: context.noInputCount + 1, // increasing the count by one
              })),
            }, // ASR_NOINPUT end
          },
        },
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
          on: { SPEAK_COMPLETE: "Listen" },
        }, // NoInput end        
        NoInputFinal: { // handling third time no input
          entry: {
            type: "spst.speak",
            params: { utterance: `I didn't hear anything. Click to try again! ` }, 
          },
          on: { SPEAK_COMPLETE: "#DM.WaitToStart" }, // sending the user back to the beginning after three times of no input
        }, // NoInputFinal end
      }, // states end  
    }, // GetContext end
    ConfirmExample: {
      // system lets the user know it will provide an example

      entry: {
        type: "spst.speak",
        params: {
          utterance: "No worries, I'll give you an example.",
        },
      },
      on: {
        SPEAK_COMPLETE: "GiveContext",
      },
    }, // ConfirmExample end
    GiveContext: {
      // the user wants to hear an example context from the system
      // or: the user wants to hear another example context from the system

      entry: [

          // setting dialogue intent (in case GiveContext is not reached from Greeting)
          assign({
            dialogueIntent: "request_example",
          }),

          assign(({ context }) => {

          // accessing retrieved Qdrant example (only one retrieved)
          const record = context.retrievedIdioms[0];

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

      always: [
        // checking if an example was selected
        {
          guard: ({ context }) =>
            context.selectedExample !== null,
          target: "GetCompletion"
        },
        {
          target: "NoExample",
        },
      ],
    }, // GiveContext end
    NoExample: {
      // the system informs the user that no example was found

      entry: {
        type: "spst.speak",
        params: {
          utterance: "Sorry, there's no example available actually."
        },
      },
      on: {
        SPEAK_COMPLETE: [
          // user has not received any explanation yet
          {
            guard: ({ context }) =>
              !context.hasExplainedIdiom,

            actions: assign({
              dialogueIntent: "explain_idiom",
            }),

            target: "GetCompletion",
          },
          // user has already received an explanation (or at least one example)
          {
            target: "AnythingMore",
          },
        ],
      },
    }, // NoExample end
    GiveDetails: {
      // the user wants to know more details about the idiom (register, difficulty, why it's idiomatic)

      // sending straight to generation step
      // all data already available through retrieved idiom data
      always: {
        target: "GetCompletion",
      },

    }, // GiveDetails end
    GetCompletion: {
      // the LLM is prompted with the retrieved context
      invoke: {
        src: "fetchLLM",

        // making retrieved data for current idiom usable for conditional prompting
        input: ({ context }) => {
          const record = context.retrievedIdioms[0];

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

                DIFFICULTY:
                ${record?.difficulty ?? "Unknown"}

                CURRENT TASK:
                ${context.dialogueIntent}

                ${
                  context.dialogueIntent === "request_example"
                    ? `
                      SELECTED EXAMPLE:
                      ${context.selectedExample?.sentence ?? ""}

                      MEANING OF THIS EXAMPLE:
                      ${context.selectedExample?.meaning ?? ""}

                      EXPLANATION OF THIS EXAMPLE:
                      ${context.selectedExample?.explanation ?? ""}

                      Your task is to help an English language learner understand how the idiom is used in the selected example.

                      Explain:
                      - What the idiom means in this particular sentence.
                      - Why the speaker uses the expression in this situation.
                      - The tone or intention, if relevant.

                      Use the example's meaning and explanation as guidance.

                      IMPORTANT:
                      - Do not repeat or quote the selected example sentence.
                      - Do not invent another example or situation.
                      - Do not simply restate the example.
                      - Explain in natural, simple English.
                      - Give approximately 2 short sentences.

                      Output ONLY the explanation.
                    `

                    : context.dialogueIntent === "explain_context"
                    ? `
                      USER'S CONTEXT:
                      ${context.userContext}

                      Explain what the idiom means in this situation.
                      Explain the speaker's likely intention. The speaker might not be the user.
                      Do not invent details about the situation.
                    `
                    : context.dialogueIntent === "more_details"
                    ? `
                      The learner wants additional information about the idiom "${context.currentIdiom}".

                      Use the retrieved information above to explain:

                      1. Why the expression is idiomatic or figurative, based on IDIOM EXPLANATION.

                      2. Its register and when it is appropriate to use, based on REGISTER.

                      3. Its difficulty and how hard it is for language learners to master, based on DIFFICULTY.

                      4. Any additional nuances supported by the retrieved meaning and explanation.

                      Do not simply repeat the general definition.
                      Do not invent origins, cultural facts, or usage restrictions that are not in the retrieved data.
                      If you do not have sufficient data, say so.
                      Do not invent facts that are not backed up by the data you received.

                      Explain in 3-4 clear sentences suitable for an English language learner.
                    `

                    : `
                      Explain the general meaning of the idiom.
                      Give a short, clear explanation.
                    `
                }

                Keep your response concise and suitable for spoken dialogue.
              `,
            },
            // excluding conversation history when explaining selected example
            ...(context.dialogueIntent === "request_example"
              ? [
                {
                  role: "user" as const,
                  content: "Please explain the selected example.",
                },
              ]
              : context.messages)
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

        onError: {
          actions: ({ event }) => {
            console.error("Completion failed:", event.error);
          },
          target: "Error",
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
          // recording that user has received an explanation or at least one example
          actions: assign({
            hasExplainedIdiom: true,
          }),
          target: "AnythingMore",
        },
      },
    }, // SpeakResponse end
    AnythingMore: {
      // system asks if there's anything else the user wants to know

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
    Error: {
      // handling possible invocation failures

      entry: {
        type: "spst.speak",
        params: {
          utterance:
            "Sorry, something went wrong. Please try again.",
        },
      },
      on: {
        SPEAK_COMPLETE: "Greeting.Ask",
      },
    }, // Error end
    Done: {
      // end state

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

// logs
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

// user interface

// changing button text
export function setupButton(element: HTMLButtonElement) {
  element.addEventListener("click", () => {
    dmActor.send({ type: "CLICK" });
  });

  dmActor.subscribe((snapshot) => {
    const meta: { view?: string } = Object.values(
      snapshot.context.spstRef.getSnapshot().getMeta(),
    )[0] || { view: undefined };

    // displaying "Click" during WaitToStart
    if (snapshot.matches("WaitToStart")) {
      element.textContent = "Click";
    } else {
      element.textContent = meta.view ?? "";
    }
  });
}

// updating dialogue context values
export function setupDialogueDisplay() {
  const idiomElement = document.querySelector<HTMLParagraphElement>(
    "#current-idiom"
  );

  const contextElement = document.querySelector<HTMLParagraphElement>(
    "#user-context"
  );

  const exampleElement = document.querySelector<HTMLParagraphElement>(
    "#selected-example"
  );

  dmActor.subscribe((snapshot) => {
    const {
      currentIdiom,
      userContext,
      selectedExample
    } = snapshot.context;

    if (idiomElement) {
      idiomElement.textContent =
        currentIdiom || "No idiom selected yet";
    }

    if (contextElement) {
      contextElement.textContent =
        userContext || "No context provided yet";
    }

    if (exampleElement) {
      exampleElement.textContent =
        selectedExample?.sentence || "No example selected yet";
    }
  });
}
