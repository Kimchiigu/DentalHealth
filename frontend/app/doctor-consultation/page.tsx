"use client";

import React, { useState } from "react";
import Chatbot from "@/components/pages/chatbot-page";
import { ScrollArea } from "@/components/ui/scroll-area";
import ReactMarkdown from "react-markdown";

export default function Home() {
  const [conversations, setConversations] = useState<
    { prompt: string; response: string }[]
  >([]);
  const [showAnswers, setShowAnswers] = useState(false);

  const handleAddAnswer = (response: string, prompt: string) => {
    setConversations((prev) => [...prev, { prompt, response }]);
    setShowAnswers(true); // Show the answers box after the first prompt
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-100 dark:bg-gray-800">
      <header className="p-6 mb-3">
        <h1 className="text-2xl font-semibold text-gray-800 dark:text-white">
          DH-1 Chatbot
        </h1>
      </header>

      <main className="flex flex-col items-center px-4">
        {showAnswers && (
          <ScrollArea className="h-[30rem] w-full flex flex-col items-center justify-center rounded-md border p-4">
            <div className="w-full  mb-6">
              {/* Conversations Section */}
              <div
                id="answer-place"
                className="bg-white dark:bg-gray-900 p-6 rounded-lg shadow-lg w-full"
              >
                <div className="space-y-4">
                  {conversations.map((item, index) => (
                    <div key={index} className="flex flex-col space-y-2">
                      {/* User Prompt */}
                      <div className="flex justify-end">
                        <div className="bg-blue-500 text-white p-4 rounded-lg shadow-md max-w-sm">
                          <p>{item.prompt}</p>
                        </div>
                      </div>

                      {/* Chatbot Response */}
                      <div className="flex justify-start">
                        <div className="bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-300 p-4 rounded-lg shadow-md max-w-3xl my-5 border border-gray-200 dark:border-gray-700">
                          <ReactMarkdown>{item.response}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </ScrollArea>
        )}

        {/* Chatbot Placeholder */}
        <Chatbot onSubmitAnswer={handleAddAnswer} />
      </main>
    </div>
  );
}
