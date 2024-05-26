# An Intelligent and Integrated Platform for Quality Control and Fault Diagnosis in 3D Printing

This project presents an intelligent and integrated platform for quality control and fault diagnosis in 3D printing. The platform leverages state-of-the-art technologies, including large language models (LLMs), knowledge bases, and computer vision, to address the limitations of traditional quality control methods and provide a comprehensive solution for real-time monitoring, defect detection, and user interaction.

## Key Features

- **Real-time Monitoring and Defect Detection**: The platform integrates the YOLOv8 object detection algorithm for real-time monitoring and defect detection in 3D printed parts. It can accurately identify and localize common defects such as spaghetti, stringing, zits, under-extrusion, and warping.

- **Conversational Fault Diagnosis**: A GPT-based chatbot is fine-tuned on a dataset of 3D printing troubleshooting conversations to provide intelligent and interactive fault diagnosis. Users can engage in natural language conversations with the chatbot to receive accurate and informative guidance on resolving printing issues.

- **Knowledge Base Integration**: The platform utilizes LLMs and the Langchain framework to construct a localized knowledge base specific to 3D printing. This knowledge base serves as a foundation for providing accurate and context-aware information to users through the chatbot interface.

- **Multimodal Diagnostic Approach**: The platform combines visual data from the monitoring system with user inputs through the chatbot interface to enable a comprehensive and accurate diagnosis of 3D printing issues. This multimodal approach leverages both objective data and subjective user information for effective problem-solving.

- **Notification and User Interaction**: The platform includes real-time notification functionalities to alert users of detected defects and anomalies. It also provides a user-friendly chatbot interface for intuitive interaction and seamless integration with the monitoring system and knowledge base.

## Dataset

The project utilizes a diverse dataset of 3D printed part images, carefully curated to represent a wide range of defects and printing scenarios. The dataset includes samples from various materials (PLA, ABS, PETG, nylon) and printing technologies (FDM, SLA, SLS). The images are annotated by experts to ensure accurate labeling of defects.

The dataset is augmented using techniques such as rotation, flipping, scaling, and translation to enhance the model's robustness and generalization capabilities. It is divided into training, validation, and testing subsets to facilitate model development and evaluation.

## Technologies Used

- **YOLOv8**: A state-of-the-art object detection algorithm used for real-time defect detection in 3D printed parts.
- **Large Language Models (LLMs)**: GPT-based models fine-tuned on domain-specific data for conversational fault diagnosis and knowledge base construction.
- **Langchain**: A framework for building applications with LLMs, used for integrating the knowledge base and chatbot components.
- **Computer Vision**: Image processing and analysis techniques used for data preprocessing, augmentation, and visualization.
- **Natural Language Processing (NLP)**: Techniques used for text processing, intent recognition, and dialogue management in the chatbot component.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/yourusername/3d-printing-quality-control.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Prepare the dataset by following the instructions in the `data` directory.
4. Train the YOLOv8 model using the provided scripts in the `models` directory.
5. Fine-tune the GPT model on the 3D printing troubleshooting dataset.
6. Integrate the trained models with the chatbot and knowledge base components.
7. Launch the platform and interact with the chatbot interface for fault diagnosis and quality control.

For detailed instructions and documentation, please refer to the `docs` directory.

## Contributions

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments

We would like to thank the experts who contributed to the annotation of the dataset and the open-source community for their valuable tools and resources that made this project possible.

## Demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/UoPQNJK_C9s?si=zez0ozMOEJrSzQfV" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


For any questions or inquiries, please contact the project maintainer at `office@3dprinter.quest`.