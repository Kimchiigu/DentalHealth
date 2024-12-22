"use client";

import React, { useState } from "react";
import { PlaceholdersAndVanishInput } from "../ui/placeholders-and-vanish-input";

interface ChatbotProps {
  onSubmitAnswer: (answer: string) => void;
}

export default function Chatbot({ onSubmitAnswer }: ChatbotProps) {
  const placeholders = [
    "What is DentalHealth?",
    "Who is the founder of AI?",
    "Who is the founder of DentalHealth?",
    "What is the features of DentalHealth?",
  ];

  const [inputValue, setInputValue] = useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {

    e.preventDefault();

    if (inputValue.trim() === "") {
      alert("Please enter a question!");
      return;
    }

    // Simulate bot response (replace with actual API call if needed)
    // const botResponse = `This is the answer to: "${inputValue}"`;

    const botResponse = "DentalHealth is the world leading Website integrated with AI to help people maintain and checking their oral health!"

    // Send response back to parent component
    onSubmitAnswer(botResponse);

    // Clear the input
    setInputValue("");

  };

  return (
    <div className="h-[40rem] flex flex-col justify-center items-center px-4">
      <h2 className="mb-10 sm:mb-20 text-xl text-center sm:text-5xl dark:text-white text-black">
        DH1 Chatbot
      </h2>
      <PlaceholdersAndVanishInput
        placeholders={placeholders}
        value={inputValue}
        onChange={handleChange}
        onSubmit={onSubmit}
      />
    </div>
  );
}
