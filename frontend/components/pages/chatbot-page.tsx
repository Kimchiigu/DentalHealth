"use client";

import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { PlaceholdersAndVanishInput } from "../ui/placeholders-and-vanish-input";

interface ChatbotProps {
  onSubmitAnswer: (response: string, prompt: string) => void;
}

export default function Chatbot({ onSubmitAnswer }: ChatbotProps) {
  const placeholders = [
    "What is oral health?",
    "How does caries formed?",
    "Give me tips to take care of my teeth?",
    "What healthy food should I eat?",
  ];

  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [thinkingTime, setThinkingTime] = useState(0); // Keeps track of thinking time
  const [showTitle, setShowTitle] = useState(true); // Controls title visibility

  useEffect(() => {
    let timer: NodeJS.Timeout;

    if (loading) {
      timer = setInterval(() => {
        setThinkingTime((prev) => prev + 1);
      }, 1000);
    } else {
      setThinkingTime(0); // Reset the timer when loading is false
    }

    return () => clearInterval(timer);
  }, [loading]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (inputValue.trim() === "") {
      alert("Please enter a question!");
      return;
    }

    const userPrompt = inputValue; // Capture the user's prompt
    setLoading(true);

    try {
      const response = await fetch("http://localhost:11434/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "deepseek-r1:8b", // Replace with your model name if different
          messages: [{ role: "user", content: inputValue }],
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      let botResponse = "";
      let isThinking = false;

      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // Parse each chunk as JSON
        const lines = chunk.trim().split("\n");
        for (const line of lines) {
          const data = JSON.parse(line);

          // Check if the bot is thinking
          if (data.message?.content.includes("<think>")) {
            isThinking = true;
            continue;
          }

          // Stop processing after </think>
          if (data.message?.content.includes("</think>")) {
            isThinking = false;
            continue;
          }

          // Only add content when the bot is not "thinking"
          if (isThinking === false) {
            botResponse += data.message?.content || "";
          }
        }
      }

      // Pass both the response and the user prompt to the parent
      onSubmitAnswer(botResponse.trim(), userPrompt);

      // Hide the title after the first prompt
      setShowTitle(false);
    } catch (error) {
      console.error("Error fetching response from Ollama:", error);
      onSubmitAnswer(
        "Sorry, something went wrong. Please try again.",
        inputValue
      );
    } finally {
      setLoading(false);
      setInputValue("");
    }
  };

  return (
    <div
      className={`w-full flex flex-col justify-center items-center px-4 transition-all duration-500 ${
        showTitle ? "h-[35rem]" : "h-[5rem]"
      }`}
    >
      {/* Title and Subtitle */}
      {showTitle && (
        <>
          <h2 className="text-xl text-center sm:text-5xl dark:text-white text-black">
            DH-1 Chatbot
          </h2>
          <p className="mt-5 mb-10 sm:mb-20">Powered by DeepSeek R1-8B</p>
        </>
      )}

      {/* Placeholder and Loading State */}
      <PlaceholdersAndVanishInput
        placeholders={placeholders}
        value={inputValue}
        onChange={handleChange}
        onSubmit={onSubmit}
      />
      {loading && (
        <div className="mt-4 text-gray-500 dark:text-gray-400">
          DH-1 is thinking... ({thinkingTime}s)
        </div>
      )}
    </div>
  );
}
