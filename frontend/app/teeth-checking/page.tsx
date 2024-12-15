"use client";

import React, { useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { FileUpload } from "@/components/ui/file-upload";
import { useToast } from "@/hooks/use-toast";
import axios from "axios"; // Import axios for API calls
import { BackgroundBeamsWithCollision } from "@/components/ui/background-beams-with-collision";

import Footer from "@/components/pages/footer"

export default function TeethChecking() {

  const [files, setFiles] = useState<File[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [classification, setClassification] = useState<string | null>(null);
  const { toast } = useToast();

  const handleFileUpload = async (files: File[]) => {
    const isImage = files.every((file) => file.type.startsWith("image/"));

    if (!isImage) {
      toast({
        title: "Invalid File Type",
        description: "Please upload a valid image file.",
        variant: "destructive",
      });
      setFiles([]);
      setSelectedImage(null);
      return;
    }

    setFiles(files);

    if (files.length > 0) {
      const file = files[0];
      const previewUrl = URL.createObjectURL(file);
      setSelectedImage(previewUrl);

      try {
        const formData = new FormData();
        formData.append("file", file);

        // Send image to backend API
        const response = await axios.post(
          "http://localhost:8000/predict",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        // Set the classification result
        setClassification(`Predicted Class: ${response.data.predicted_class}`);
      } catch (error) {
        console.error("Error uploading file:", error);
        toast({
          title: "Error",
          description: "An error occurred while uploading the file.",
          variant: "destructive",
        });
      }
    } else {
      setSelectedImage(null);
    }
  };

  useEffect(() => {
    return () => {
      if (selectedImage) {
        URL.revokeObjectURL(selectedImage);
      }
    };
  }, [selectedImage]);

  return (

    <>

      <div className="flex min-h-screen flex-row items-center justify-center">
        <div className="flex flex-col min-h-screen items-center justify-center w-1/2">
          <FileUpload onChange={handleFileUpload} />
        </div>

        {/* Preview Area */}
        <div className="flex flex-col items-center justify-center w-1/2 bg-gray-100 min-h-[90vh] shadow-inner my-5 mx-5 rounded-xl">
          <BackgroundBeamsWithCollision className="min-h-[90vh] rounded-xl z-0">
            {selectedImage ? (
              <div className="z-20">
                <h1 className="mb-2">Image Preview:</h1>
                <img
                  src={selectedImage}
                  alt="Preview"
                  className="max-w-full max-h-96 rounded-xl shadow-xl"
                />
                {classification && (
                  <p className="mt-4 text-xl text-gray-700">{classification}</p>
                )}
              </div>
            ) : (
              <h1>Image is empty :(</h1>
            )}
          </BackgroundBeamsWithCollision>
        </div>

        

      </div>

      <Footer/>

    </>

  );
}
