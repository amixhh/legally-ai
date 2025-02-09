import React from 'react';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

const Feature = ({ title, description }) => (
  <div className="p-8 rounded-2xl bg-white/20 backdrop-blur-lg shadow-lg hover:shadow-2xl transition-shadow text-center border border-white/30">
    <h3 className="text-2xl font-medium text-black mb-4">{title}</h3>
    <p className="text-gray-600">{description}</p>
  </div>
);

const LandingPage = () => {
  const features = [
    {
      title: "Legal Advice & Case Analysis",
      description: "Utilizing AI-powered insights to analyze case outcomes and provide clear, data-driven explanations and advice."
    },
    {
      title: "Legal Documents Summarization",
      description: "Streamlining the review of complex legal texts by extracting key insights quickly and efficiently."
    },
    {
      title: "Intelligent Legal Query System",
      description: "Delivering precise answers to legal questions."
    }
  ];

  return (
    <div className={`min-h-screen bg-white text-black ${inter.className}`}> 
      <nav className="fixed top-0 left-0 w-full flex justify-between items-center px-8 py-6 bg-white/30 backdrop-blur-lg shadow-lg border-b border-white/20 z-50">
        <div className="text-3xl font-bold text-black">Legally AI</div>
        <div className="space-x-8">
          <a
            href={process.env.NEXT_PUBLIC_STREAMLIT_URL}
            className="bg-black hover:bg-gray-800 text-white px-6 py-3 rounded-full font-medium transition-all shadow-md"
          >
            Get Started
          </a>
        </div>
      </nav>

      <main className="pt-24">
        <section className="text-center py-24 px-4">
          <h1 className="text-5xl font-semibold mb-4">An AI Legal Assistant</h1>
          <h2 className="text-3xl text-gray-700 mb-8">Made for Billions!</h2>
          <p className="text-lg text-gray-600 mb-12 max-w-3xl mx-auto">
            We are an AI-powered legal assistance platform that provides free legal advice for all.
          </p>
          <div className="flex justify-center space-x-4">
            <a
              href={process.env.NEXT_PUBLIC_STREAMLIT_URL}
              className="px-6 py-4 bg-black hover:bg-gray-800 text-white rounded-full font-medium transition-all shadow-lg"
            >
              Try Now
            </a>
            <a
              href="#features"
              className="px-6 py-4 bg-black hover:bg-gray-800 text-white rounded-full font-medium transition-all shadow-md"
            >
              See Features
            </a>
          </div>
          <p className="text-gray-600 mt-8">Chat, Summary, QnA and more.</p>
        </section>

        <section id="features" className="py-24 px-4 bg-gray-50 border-t border-gray-200">
          <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <Feature key={index} {...feature} />
            ))}
          </div>
        </section>
      </main>

      <footer className="bg-gray-100 text-black py-8 px-8 border-t border-gray-200">
        <div className="max-w-6xl mx-auto flex flex-col items-center">
          <p className="text-gray-600">© Copyright 2025 Legally AI</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;