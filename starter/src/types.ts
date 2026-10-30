import { Hypothesis, SpeechStateExternalEvent } from "speechstate";
import { AnyActorRef } from "xstate";

export type Message = {
  role: "assistant" | "user" | "system";
  content: string;
};

export interface DMContext {
  spstRef: AnyActorRef;
  lastResult: Hypothesis[] | null;
  // nextUtterance: string;

  interpretation: NLUObject | null; 
  
  messages: Message[]; // for LLM
  //retrievedContext: string; // for RAG

  noInputCount: number; // to handle no input

  // keeping track of the conversation
  currentIdiom: string; // idiom under discussion
  userContext: string; // context for idiom provided by user
  retrievedIdioms: IdiomRecord[]; // results from Qdrant (RAG)
  dialogueIntent: string; // what user wants to do
  usedExampleIndices: []; // examples (from Qdrant) already used
  needsContext: boolean; // whether system should ask user for context

}

// structure for each idiom example
export interface IdiomExample {
  sentence: string;
  meaning: string;
  explanation: string; 
  context_type: string;
  tags?: string;
}

// full retrieved idiom structure (array)
export interface IdiomRecord {
  idiom: string;
  meaning: string; 
  explanation: string;
  register: string;
  difficulty: string;
  example: IdiomExample[];
}

// Adding the Intent object type
export interface Intent { // intents array in NLUObject
  category: string;
  confidenceScore: number;
}

// Adding the NLUObject (contains Intent)
export interface NLUObject { // interpretation in DMContext
  intents: Intent[];
  projectKind: string;
  topIntent: string;
}

export type DMEvents = SpeechStateExternalEvent | { type: "CLICK" } | {type: "DONE"};
