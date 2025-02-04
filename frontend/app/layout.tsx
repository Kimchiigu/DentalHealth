import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

import Navbar from "@/components/pages/navbar";
// import Footer from "@/components/pages/footer";

import AnimatedCursor from "@/components/ui/custom-cursor";
import { ThemeProvider } from "@/context/ThemeContext";
import ThemeToggleButton from "@/components/ui/toggle";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});

const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "DentalHealth",
  description: "Keep your teeth healthy!",
  icons: "/frontend/public/dental-health-logo.png",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}

        <ThemeProvider>
          <AnimatedCursor />
          <Navbar />
          <ThemeToggleButton />
          <Toaster />
          {/* <Footer/> */}
        </ThemeProvider>
      </body>
    </html>
  );
}
