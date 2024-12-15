"use client";

import Footer from "@/components/pages/footer"

export default function ArticlePage({ params }) {

  const { id } = params;

  const articleData = {
    1: {
      title: "Daily Dental Care Guide",
      sections: [
        {
          image: "/assets/teeth1.png",
          text: "Maintaining healthy teeth requires a consistent daily routine. Brushing twice a day with fluoride toothpaste helps remove plaque and food particles, preventing decay. Flossing reaches the spaces between teeth that a brush can't, ensuring thorough cleaning. Regular dental checkups are essential for identifying potential problems early and keeping your smile bright.",
          reverse: false,
        },
        {
          image: "/assets/teeth2.png",
          text: "Choosing the right toothbrush matters. Soft-bristled brushes are gentle on enamel and gums while being effective at cleaning. Electric toothbrushes can further enhance the cleaning process, especially for individuals with limited dexterity. Coupled with proper technique, this is a cornerstone of dental health.",
          reverse: true,
        },
        {
          image: "/assets/teeth3.png",
          text: "A healthy diet supports dental care. Foods rich in calcium, like dairy products, strengthen teeth, while avoiding sugary snacks minimizes the risk of cavities. Drink plenty of water to wash away food particles and maintain hydration, which aids saliva production for natural cleaning.",
          reverse: false,
        },
      ],
    },
  
    2: {
      title: "How to Prevent Cavities",
      sections: [
        {
          image: "/assets/teeth2.png",
          text: "Cavities occur when acids produced by bacteria erode tooth enamel. These acids are often a result of consuming sugary or starchy foods. Prevention starts with minimizing these foods and practicing good oral hygiene to disrupt bacterial growth and acid production.",
          reverse: false,
        },
        {
          image: "/assets/teeth1.png",
          text: "Fluoride is a natural cavity fighter. It strengthens enamel, making it more resistant to acid attacks. Use fluoride toothpaste and consider fluoride treatments from your dentist for added protection. Tap water containing fluoride also contributes to dental health.",
          reverse: true,
        },
        {
          image: "/assets/teeth3.png",
          text: "Dental sealants provide another layer of defense. These thin coatings applied to the chewing surfaces of molars block out food and bacteria, reducing the risk of decay. Sealants are particularly effective for children and teenagers.",
          reverse: false,
        },
      ],
    },
  
    3: {
      title: "What is Tartar and How to Remove It?",
      sections: [
        {
          image: "/assets/teeth3.png",
          text: "Tartar, or dental calculus, forms when plaque hardens on teeth. This usually occurs when plaque is not removed effectively through brushing and flossing. Once hardened, tartar cannot be removed with regular home care and requires professional cleaning by a dentist or hygienist.",
          reverse: false,
        },
        {
          image: "/assets/teeth2.png",
          text: "Tartar buildup not only affects the appearance of your teeth but also contributes to gum disease. It creates rough surfaces that attract more plaque and bacteria, exacerbating oral health problems. Regular cleanings are key to controlling tartar.",
          reverse: true,
        },
        {
          image: "/assets/teeth1.png",
          text: "Prevention is better than treatment. Brushing with an anti-tartar toothpaste and flossing daily can significantly reduce plaque buildup. Keeping up with routine dental visits ensures early intervention before tartar becomes problematic.",
          reverse: false,
        },
      ],
    },
  
    4: {
      title: "Causes and Treatments for Bleeding Gums",
      sections: [
        {
          image: "/assets/teeth4.png",
          text: "Bleeding gums are often a sign of gingivitis, an early stage of gum disease. They may result from poor oral hygiene, which allows plaque to accumulate and irritate the gums. Addressing bleeding gums early prevents progression to more serious periodontal disease.",
          reverse: false,
        },
        {
          image: "/assets/teeth2.png",
          text: "Improved oral hygiene can reverse gingivitis. Brushing gently along the gumline, flossing daily, and using an antiseptic mouthwash can reduce inflammation and promote healing. Be consistent to see long-term benefits.",
          reverse: true,
        },
        {
          image: "/assets/teeth3.png",
          text: "If bleeding persists, consult your dentist. They can perform a deep cleaning to remove plaque and tartar below the gumline. In advanced cases, periodontal treatments like scaling and root planing may be necessary to restore gum health.",
          reverse: false,
        },
      ],
    },
  
    5: {
      title: "Fluoride for Strong Teeth",
      sections: [
        {
          image: "/assets/teeth5.png",
          text: "Fluoride plays a vital role in preventing cavities. It strengthens tooth enamel by promoting remineralization and making enamel more resistant to acid attacks. Fluoride is commonly found in toothpaste and drinking water.",
          reverse: false,
        },
        {
          image: "/assets/teeth2.png",
          text: "Children benefit greatly from fluoride as their teeth develop. Regular exposure ensures their teeth grow strong and healthy. Fluoride treatments, especially in areas with non-fluoridated water, can make a significant difference.",
          reverse: true,
        },
        {
          image: "/assets/teeth1.png",
          text: "Overexposure to fluoride can lead to fluorosis, which causes white spots on teeth. Use the recommended amount of fluoride toothpaste and supervise children to avoid swallowing it unnecessarily.",
          reverse: false,
        },
      ],
    },
  
    6: {
      title: "Toothpaste Selection Tips",
      sections: [
        {
          image: "/assets/teeth1.png",
          text: "Choosing the right toothpaste depends on your specific dental needs. For sensitive teeth, toothpaste with desensitizing agents like potassium nitrate can alleviate discomfort. Whitening toothpaste helps remove surface stains for a brighter smile.",
          reverse: false,
        },
        {
          image: "/assets/teeth2.png",
          text: "If you’re prone to cavities, use a fluoride toothpaste to strengthen enamel and prevent decay. For those with gum issues, toothpaste containing antibacterial agents like triclosan can reduce inflammation and plaque.",
          reverse: true,
        },
        {
          image: "/assets/teeth3.png",
          text: "Consult your dentist for personalized recommendations. Some toothpaste contains abrasives that might not suit everyone, especially those with weakened enamel. Finding the right balance ensures optimal oral health.",
          reverse: false,
        },
      ],
    },
  };

  const article = articleData[id] || {
    title: "Article Not Found",
    sections: [],
  };

  return (

    <>
    
        <div className="container mx-auto p-8">

        <h1 className="text-4xl font-extrabold text-center mb-8">{article.title}</h1>

        {article.sections.map((section, index) => (

            <div
            key={index}
            className={`flex flex-col md:flex-row items-center mb-12 ${
                section.reverse ? "md:flex-row-reverse" : ""
            }`}
            >

            <div className="md:w-1/2 mb-4 md:mb-0">
                <img
                src={section.image}
                alt={article.title}
                className="rounded-lg shadow-lg w-full h-auto"
                />
            </div>

            <div className="md:w-1/2 md:px-8">
                <p className="text-lg text-gray-700 leading-relaxed dark:text-white">{section.text}</p>
            </div>

            </div>

        ))}

        </div>

        <Footer/>

    </>

  );

}
