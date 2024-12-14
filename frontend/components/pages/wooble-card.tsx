"use client";
import Image from "next/image";
import React from "react";
import { WobbleCard } from "@/components/ui/wobble-card";

export default function WobbleCardDemo() {

  return (

    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 max-w-7xl mx-auto w-full">

      <WobbleCard
        containerClassName="col-span-1 lg:col-span-2 h-full bg-pink-800 min-h-[500px] lg:min-h-[300px]"
        className=""
      >

        <div className="max-w-xs">
          <h2 className="text-left text-balance text-base md:text-xl lg:text-3xl font-semibold tracking-[-0.015em] text-white">
            Welcome to DentalHealth 🪥
          </h2>
          <p className="mt-4 text-left  text-base/6 text-neutral-200">
          A perfect place to consultate about your teeth condition for cheap price! Not only that, we also provides insane technology that will help you know your teeth better.
          </p>
        </div>

        <Image
          src="/assets/teeth2.png"
          width={500}
          height={500}
          alt="linear demo image"
          className="absolute -right-4 lg:-right-[10%] filter -bottom-10 object-contain rounded-2xl"
          unoptimized
        />

      </WobbleCard>


      <WobbleCard containerClassName="col-span-1 min-h-[300px]">
        <h2 className="max-w-80  text-left text-balance text-base md:text-xl lg:text-3xl font-semibold tracking-[-0.015em] text-white">
          Good Teeth, Nice Smile 🌟
        </h2>
        <p className="mt-4 max-w-[26rem] text-left  text-base/6 text-neutral-200">
          We offer best features that you will definitely like it. Not only quality but we offer it with cheap price!
        </p>
      </WobbleCard>


      <WobbleCard containerClassName="col-span-1 lg:col-span-3 bg-blue-900 min-h-[500px] lg:min-h-[600px] xl:min-h-[300px] mb-20">
        <div className="max-w-sm">
          <h2 className="max-w-sm md:max-w-lg  text-left text-balance text-base md:text-xl lg:text-3xl font-semibold tracking-[-0.015em] text-white">
            DentalHealth Experience 🔥
          </h2>
          <p className="mt-4 max-w-[26rem] text-left  text-base/6 text-neutral-200">
          DentalHealth already trusted by many people and we love to help you too! So what are you waiting for?? Just using DentalHealth right now!
          </p>
        </div>
        <Image
          src="/assets/teeth5.png"
          width={500}
          height={500}
          alt="linear demo image"
          className="absolute -right-10 md:-right-[30%] lg:-right-[8%] -bottom-2 object-contain rounded-2xl"
          unoptimized
        />
      </WobbleCard>

    </div>
  );
}
