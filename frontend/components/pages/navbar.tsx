"use client";

import React from "react";
import { FloatingNav } from "@/components/ui/floating-navbar";
import { IconHome, IconMessage, IconUser } from "@tabler/icons-react";

export default function FloatingNavDemo() {

  const navItems = [
    
    {
      name: "Home",
      link: "/",
      icon: <IconHome className="h-4 w-4 text-neutral-500 dark:text-white" />,
    },

    {
        name: "About Us",
        link: "/about",
        icon: (
          <IconMessage className="h-4 w-4 text-neutral-500 dark:text-white" />
        ),
    },

    {
      name: "Consultation",
      link: "/",
      icon: <IconUser className="h-4 w-4 text-neutral-500 dark:text-white" />,
    },

    {
      name: "Teeth Checking",
      link: "/teeth-checking",
      icon: <IconUser className="h-4 w-4 text-neutral-500 dark:text-white" />,
    },

    {
      name: "Medical Knowledge",
      link: "/article",
      icon: <IconUser className="h-4 w-4 text-neutral-500 dark:text-white" />,
    },
    
  ];

  return (
    <div className="relative  w-full">
      <FloatingNav navItems={navItems} />
    </div>
  );

}