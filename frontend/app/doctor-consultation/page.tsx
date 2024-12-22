"use client";

import React, { useState } from "react";
import Chatbot from "@/components/pages/chatbot-page";
import Footer from  "@/components/pages/footer"

export default function Home() {
  const [answers, setAnswers] = useState<string[]>([]);

  const handleAddAnswer = (answer: string) => {
    setAnswers((prev) => [...prev, answer]);
  };

  return (
    <>

      <Chatbot onSubmitAnswer={handleAddAnswer} />

      <div id="answer-place" className="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg shadow-lg mb-12 max-w-3xl mx-auto">

        <h3 className="text-xl font-bold text-gray-800 dark:text-white mb-4">Chatbot Answers</h3>

        <div className="space-y-2">
          {answers.length > 0 ? (
            answers.map((answer, index) => (
              <div
                key={index}
                className="p-4 bg-white dark:bg-gray-900 rounded-lg shadow-md border border-gray-200 dark:border-gray-700"
              >
                <p className="text-gray-700 dark:text-gray-300">{answer}</p>
              </div>
            ))
          ) : (
            <p className="text-gray-500 dark:text-gray-400">No answers yet. Ask something!</p>
          )}
        </div>

      </div>

      <Footer/>

    </>
  );
}
