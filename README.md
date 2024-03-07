# Project Name

## Description

This system is designed to create multiple-choice questions using Generative AI technology, particularly utilizing LangChain and the OpenAI API. The tool allows users to upload text or PDF files by drag and drop them into the provided area. Once the file is uploaded, users can specify the subject of the content in the file and adjust the difficulty level of the questions to be generated. Essentially, it's a platform that automates the process of generating multiple-choice questions based on the content of a given text or PDF file, with customization options for subject and difficulty level.
Note : This model's maximum data context length is 16385 tokens.

## Installation

1. Clone the repository: `git clone https://github.com/m-rachman/MCQ.git`
2. Navigate to the project directory: `cd project-directory`
3. Install dependencies: `pip install -r requirements.txt`

### Built With

This project utilizes the OpenAI API with GPT-3.5-Turbo model for advanced natural language processing capabilities. Langchain facilitates seamless integration of various programming languages, while Streamlit enables intuitive and interactive web-based user interfaces. Together, they empower developers to create dynamic and efficient applications.

[![Langchain][Langchain]][Langchain-url]
[![OpenAI][OpenAI]][OpenAI-url]
[![streamlit][streamlit]][streamlit-url]




[Langchain]: data:image/png;base64,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
[Langchain-url]: https://www.langchain.com/
[OpenAI]: https://freelogopng.com/images/all_img/1681142315open-ai-logo.png
[OpenAI-url]: https://openai.com/
[streamlit]: https://seeklogo.com/images/S/streamlit-logo-B405F7E2FC-seeklogo.com.png
[streamlit-url]: https://vuejs.org/
