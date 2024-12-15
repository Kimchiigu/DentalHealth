export default function Custom404() {
    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 text-white">
        <h1 className="text-9xl font-extrabold mb-4">404</h1>
        <h2 className="text-3xl md:text-4xl font-semibold mb-4">Page Not Found</h2>
        <p className="text-center max-w-md mb-6">
            Sorry, the page you are looking for doesn’t exist. It might have been removed or the URL might be incorrect.
        </p>
        <a
            href="/"
            className="px-6 py-3 bg-white text-blue-600 font-semibold rounded-lg shadow-md hover:bg-gray-100"
        >
            Go Back Home
        </a>
        <img
            src="/assets/404-illustration.png"
            alt="404 Illustration"
            className="w-full max-w-md mt-8"
        />
        </div>
    );
}
  