"use client";

import React, { useEffect } from "react";
import { gsap } from "gsap";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import { AuroraBackground } from "@/components/ui/aurora-background";
import { FocusCards } from "@/components/ui/focus-cards";

export default function Home() {

  const cards = [
    {
      title: "Doctor Consultation",
      src: "/assets/doctor-consultation.png",
      desc: "Consult with our 24/7 professional doctors around the world",
    },
    {
      title: "Teeth Checking",
      src: "/assets/teeth-checking.png",
      desc: "Check your teeth conditions using our latest AI technology",
    },
    {
      title: "Medical Knowledge",
      src: "/assets/medical-knowledge.png",
      desc: "Explore more about your health in our user-friendly website",
    },
  ];

  return (
    <AuroraBackground className="">
      <div className="py-10 z-20 w-full flex flex-col items-center justify-center min-h-screen">
        <h1 className="font-bold text-2xl md:text-4xl">
          Welcome to DentalHealth!
        </h1>
        <h2 className="text-md md:text-lg mb-8">
          Choose the type of the services that you want!
        </h2>
        <FocusCards cards={cards} />
      </div>
    </AuroraBackground>
  );

}
