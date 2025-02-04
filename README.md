<h1 align="center"> DentalHealth </h1> <br>
<p align="center">
  <a href="">
    <img alt="DentalHealth" title="DentalHealth" src="https://res.cloudinary.com/dxcn5osfu/image/upload/f_auto,q_auto/v1/DentalHealth/fvifsotbrscbknw64mgm" width="200">
  </a>
</p>

<p align="center">
  Computer Vision and Deep Learning for Dental Screening
</p>

# Artificial Intelligence: DentalHealth 🦷🪥

This repository contains the AI Project Class of BINUS University with the Project Title "DentalHealth" 🔥🔥

## Table of Contents
1. [Project Overview](#project-overview)
2. [Mockup Screenshot](#mockup-screenshot)
3. [Web Demo](#web-demo)
4. [Prerequisite](#prerequisite)
5. [How to Run Project Locally](#how-to-run-project-locally)
6. [Owner](#owner)

## Project Overview

### Project Name
DentalHealth

### Explanation

DentalHealth is a perfect place to consult about your dental health for an affordable price! Not only that, we also provide advanced technology to help you understand your teeth better. Trusted by many, DentalHealth is here to assist you too! So what are you waiting for? Start using DentalHealth now! ❤️‍🔥✨

## Mockup Screenshot

* Home Page - Main Feature Selection

<p align="center">
  <img src = "https://res.cloudinary.com/dxcn5osfu/image/upload/f_auto,q_auto/v1/DentalHealth/xe2dixyvuw0ludqoh3zc" width=700>
</p>

* Chatbot Page

<p align="center">
  <img src = "https://res.cloudinary.com/dxcn5osfu/image/upload/f_auto,q_auto/v1/DentalHealth/fsjtmrxun4z2g7w0fl2g" width=700>
</p>

* Teeth Checking Page

<p align="center">
  <img src = "https://res.cloudinary.com/dxcn5osfu/image/upload/f_auto,q_auto/v1/DentalHealth/sjujnrrzhampna1hplke" width=700>
</p>

* Article Page

<p align="center">
  <img src = "https://res.cloudinary.com/dxcn5osfu/image/upload/f_auto,q_auto/v1/DentalHealth/yczw7gxrlxd4f7uzw5hk" width=700>
</p>

## Web Demo

* Teeth Checking - Utilizing CNN Model

<p align="center">
  <img src = "https://res.cloudinary.com/dxcn5osfu/image/upload/f_auto,q_auto/v1/DentalHealth/rxk7uo4nq6bj4853qzq3" width=700>
</p>

* Chatbot - Powered by Deepseek R1:8b

<p align="center">
  <img src = "https://res.cloudinary.com/dxcn5osfu/image/upload/f_auto,q_auto/v1/DentalHealth/i1zatnvyy8akk0gcfjga" width=700>
</p>

### Technology and Infrastructure

This project is built using modern web development tools and frameworks:
- **Frontend Framework**: [Next.js](https://nextjs.org/) (React-based framework for server-side rendering and static site generation)
- **Styling**: [TailwindCSS](https://tailwindcss.com/) (Utility-first CSS framework for fast UI development)
- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python framework for building APIs)
- **AI Model**: PyTorch-based deep learning models
- **Programming Languages**: JavaScript/TypeScript (frontend), Python (backend)
- **Package Manager**: npm or yarn for frontend; pip for backend
- **Deployment**: [Vercel](https://vercel.com/) for frontend, and [AWS/Google Cloud/Heroku] for backend (optional)

## Prerequisite

Before running this project, ensure the following tools are installed on your system:

1. **Node.js and npm:**
   - Install Node.js (version **16.x** or newer) which includes npm.  
     Download Node.js from [here](https://nodejs.org/).
   - Verify installation by running:

     ```bash
     node -v
     npm -v
     ```

2. **Python:**
   - Install Python (version **3.9** or newer).  
     Download Python from [here](https://www.python.org/).
   - Verify installation by running:

     ```bash
     python --version
     pip --version
     ```

3. **Git:**
   - Install Git to clone the repository from GitHub.  
     Download Git from [here](https://git-scm.com/).  
     Verify installation using:

     ```bash
     git --version
     ```

4. **Code Editor:**
   - Install [Visual Studio Code (VS Code)](https://code.visualstudio.com/) or any editor of your choice.  
     Ensure you have the following extensions for VS Code:
     - **ESLint**
     - **Prettier - Code formatter**
     - **Python**
     - **TailwindCSS IntelliSense**

5. **Virtual Environment (Python):**
   - Install `virtualenv` or `venv` to create isolated environments for Python dependencies:

     ```bash
     pip install virtualenv
     ```

6. **Browser:**
   - Use a modern browser like [Google Chrome](https://www.google.com/chrome/) or [Firefox](https://www.mozilla.org/).

## How to Run Project Locally

Follow these steps to set up and run the project locally on your machine:

---

### Frontend (Next.js)

#### 1. Navigate to the Frontend Directory
   - Navigate to the `frontend` folder:

     ```bash
     cd frontend
     ```

#### 2. Install Dependencies
   - Install the required Node.js packages:

     ```bash
     npm install
     ```

#### 3. Start the Development Server
   - Run the following command to start the development server:

     ```bash
     npm run dev
     ```

   - By default, the application will run on `http://localhost:3000`.

---

### Backend (AI Model API)

#### 1. Navigate to the Model Directory
   - Open a new terminal and navigate to the `model` folder:

     ```bash
     cd model
     ```

#### 2. Set Up Python Environment
   - Create a virtual environment to isolate dependencies:

     ```bash
     python -m venv venv
     ```

   - Activate the virtual environment:
     - On **Windows**:
       ```bash
       venv\Scripts\activate
       ```
     - On **macOS/Linux**:
       ```bash
       source venv/bin/activate
       ```

#### 3. Install Python Dependencies
   - Install the required libraries listed in `requirements.txt`:

     ```bash
     pip install -r requirements.txt
     ```

#### 4. Start the Backend API
   - Run the Python API server:

     ```bash
     python modelAPI.py
     ```

   - By default, the API will run on `http://localhost:8000`.

---

### Integration and Testing

#### 1. Connect Frontend with Backend
   - Update the frontend API configuration file (e.g., `.env` or API service file) to point to the backend server at `http://localhost:8000`.

#### 2. Test the Application
   - Open your browser and visit [http://localhost:3000](http://localhost:3000).
   - Use the features of the application, which interact with the AI model hosted on the backend.

---

### Deployment (Optional)
#### Frontend:
   - Deploy the frontend to [Vercel](https://vercel.com/).  
     Follow [these steps](https://vercel.com/docs).

#### Backend:
   - Deploy the Python API to platforms like [AWS](https://aws.amazon.com/), [Google Cloud](https://cloud.google.com/), or [Heroku](https://www.heroku.com/).

---

## Owner

This Repository is created by:
- 2702217125 - Stanley Nathanael Wijaya
- 2702211185 - Christopher Hardy Gunawan
- 2702217043 - Christine Kosasih
- 2702218153 - Olivia Tiffany

<code> Striving for Excellence ❤️‍🔥❤️‍🔥 </code>
