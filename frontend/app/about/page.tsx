"use client";

import React from "react";
import { motion } from "framer-motion";
import { HeroHighlight, Highlight } from "@/components/ui/hero-highlight";
import TeamCard from  "@/components/pages/team-card"
import Footer from  "@/components/pages/footer"
// import Globe from  "@/components/pages/globe"
// import LampLanding from "@/components/pages/lamp-landing"

export default function About() {

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
              
              Team behind

              <br />

              <Highlight className="text-black dark:text-white">
                DentalHealth
              </Highlight>

            </motion.h1>

          </HeroHighlight>

          {/* <LampLanding/> */}

          <TeamCard/>

          {/* <Globe/> */}

          <Footer/>

      </>

    );

}