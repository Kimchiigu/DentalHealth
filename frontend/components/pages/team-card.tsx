import { AnimatedTestimonials } from "@/components/ui/animated-testimonials";

export default function AnimatedTestimonialsDemo() {

  const testimonials = [

    {
      quote:
        "The attention to detail and innovative features have completely transformed our workflow. This is exactly what we've been looking for.",
      name: "Christopher Hardy Gunawan",
      designation: "Product Manager at TechFlow",
      src: "/assets/teeth1.png",
    },

    {
      quote:
        "Implementation was seamless and the results exceeded our expectations. The platform's flexibility is remarkable.",
      name: "Olivia Tiffany",
      designation: "CTO at InnovateSphere",
      src: "/assets/teeth1.png",
    },

    {
      quote:
        "This solution has significantly improved our team's productivity. The intuitive interface makes complex tasks simple.",
      name: "Christine Kosasih",
      designation: "Operations Director at CloudScale",
      src: "/assets/teeth1.png",
    },

    {
        quote:
          "This solution has significantly improved our team's productivity. The intuitive interface makes complex tasks simple.",
        name: "Stanley Nathanael Wijaya",
        designation: "Operations Director at CloudScale",
        src: "/assets/profile/stanz.png",
      },

  ];

  return <AnimatedTestimonials testimonials={testimonials} />;

}
