"use client";

import React from "react";

// import {useEffect} from "react"
// import { gsap } from "gsap";
// import {
//   Card,
//   CardDescription,
//   CardHeader,
//   CardTitle,
// } from "@/components/ui/card";

import { motion } from "framer-motion";
import { HeroHighlight, Highlight } from "@/components/ui/hero-highlight";

import { AuroraBackground } from "@/components/ui/aurora-background";
import { FocusCards } from "@/components/ui/focus-cards";
import BentoGrid from "@/components/pages/bento-grid";
import WoobleCard from '@/components/pages/wooble-card';
import Globe from  "@/components/pages/globe"
import Footer from  "@/components/pages/footer"

// Constants

import {cards} from "@/constants/const"

export default function Home() {

  return (

    <>

      <HeroHighlight>

        <motion.h1
          initial={{
            opacity: 0,
            y: 20,
          }}
          animate={{
            opacity: 1,
            y: [20, -5, 0],
          }}
          transition={{
            duration: 0.5,
            ease: [0.4, 0.0, 0.2, 1],
          }}
          className="text-3xl px-4 md:text-4xl lg:text-5xl font-bold text-neutral-700 dark:text-white max-w-4xl leading-relaxed lg:leading-snug text-center mx-auto "
        >
          
          Welcome to DentalHealth

          <br />

          <Highlight className="text-black dark:text-white">
            Good Teeth, Nice Smile
          </Highlight>

        </motion.h1>
      </HeroHighlight>

      {/* <BentoGrid/> */}

      <WoobleCard/>

      <AuroraBackground className="">
        <div className="py-10 z-20 w-full flex flex-col items-center justify-center min-h-screen">
          <h1 className="font-bold text-2xl md:text-4xl dark:text-slate-50">
            ❤️‍🔥 DentalHealth Features ❤️‍🔥
          </h1>
          <h2 className="text-md md:text-lg mb-8 dark:text-slate-200">
            Choose the type of the services that you want!
          </h2>
          <FocusCards cards={cards} />
        </div>
      </AuroraBackground>

      <Globe/>

      <Footer/>

    </>

  );

}
