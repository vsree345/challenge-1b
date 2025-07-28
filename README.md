# Persona-Driven Document Intelligence (Adobe Hackathon - Challenge 1B)

This project is a sophisticated solution for Challenge 1B of the Adobe India Hackathon. It implements an advanced Retrieval-Augmented Generation (RAG) pipeline to analyze multiple document collections and extract relevant, persona-driven insights.

## Key Features

Our solution goes beyond simple semantic search to deliver highly accurate and user-centric results.

*   **Advanced Two-Stage Retrieval:** To solve the critical "negative constraint" problem (e.g., finding "vegetarian" recipes), we use a two-stage pipeline.
    1.  **Keyword Filtering:** First, it filters out irrelevant content based on keywords (like removing non-vegetarian items).
    2.  **Semantic Ranking:** Then, it applies a powerful sentence-transformer model to rank the remaining "safe" content.

*   **AI-Powered Title Generation:** Instead of using unreliable heuristics, the system uses a local generative model (`google/flan-t5-base`) to create clean, concise, and human-readable titles for each extracted section.

*   **Global Ranking:** The pipeline pools content from all documents in a collection and ranks them globally, ensuring the final output contains the absolute best results from the entire set.

*   **Unified & Configurable Architecture:** A single, robust Python script (`run_analysis.py`) handles all collections, configured via command-line arguments for professional and scalable execution.

## How to Run the Project

1.  **Install Dependencies:**
    First, ensure all required libraries and the models are installed by running the commands below.
    ```
    pip install -r requirements.txt
    python download_model.py
    python download_model1.py

    Build the docker image using the following command

    docker build --platform linux/amd64 -t adobe1b:offlinefinal . 

    ```

2.  **Run the Analysis Script:**
    Execute the script from the terminal, providing the path to the desired collection directory.


    *   **For Collection 1 (Travel Planning):**
        ```
        python run_analysis.py ./Collection_1/
        or 
        docker run --rm `-v "${PWD}/input:/app/input" ` -v "${PWD}/output:/app/output" `--network none ` adobe1b:offlinefinal input/Collection_1
        ```

    *   **For Collection 2 (Adobe Acrobat Learning):**
        ```
        python run_analysis.py ./Collection_2/
        or
        docker run --rm `        
        -v "${PWD}/input:/app/input" `
        -v "${PWD}/output:/app/output" `
        --network none `
        adobe1b:offlinefinal input/Collection_2
        ```

    *   **For Collection 3 (Recipe Collection):**
        ```
        python run_analysis.py ./Collection_3/
        or
        docker run --rm `        
        -v "${PWD}/input:/app/input" `
        -v "${PWD}/output:/app/output" `
        --network none `
        adobe1b:offlinefinal input/Collection_3
        ```
    The script will process the documents and generate a `challenge1b_output.json` file inside the respective collection directory.

## Project Structure
adobe-hackathon-challenge1b-master/
├── Collection_1/
│   ├── challenge1b_input.json
│   ├── challenge1b_output.json         # (generated after execution)
│   └── pdfs/
│       ├── all the pdf dcocuments will be visble here
├── Collection_2/
│   ├── challenge1b_input.json
│   └── pdfs/
│       ├── all the pdf dcocuments will be visble here
├── Collection_3/
│   ├── challenge1b_input.json
│   └── pdfs/
│       ├── all the pdf dcocuments will be visble here
├── models/                             
│   ├── all-MiniLM-L6-v2/
│   └── flan-t5-base/
├── run_analysis.py                     # Core RAG logic
├── download_model1.py                  # Script to download required models
├── Dockerfile                          # For containerized offline execution
├── requirements.txt                    # Python dependencies
├── README.md
└── approach_explanation.md


## Conclusion

This project shows an effective and smart way to analyze documents based on user personas. By using a multi-stage retrieval process, it goes beyond the limits of traditional semantic search.

The main strengths of this approach are:
*   **Accuracy:** The two-step process of filtering keywords and then ranking semantically guarantees that the results are contextually relevant and meet important criteria, such as the "vegetarian" requirement.
*   **Usability:** By using a generative AI model to create clear and descriptive titles, the final JSON output changes from raw data into a refined and usable report.

In the end, this solution highlights a practical and scalable design for developing future document intelligence systems that truly understand what users want.
