# yolov9-segment

This service will use a YOLOv9 model to apply semantic segmentation to an orthophoto image.

## Install

1) You need to have the Core engine installed before using this service.

    ```bash
    # Clone the Core engine repository
    git clone git@github.com:swiss-ai-center/core-engine
    ```

    Documentation for the installation of the Core engine can be found [here](https://docs.swiss-ai-center.ch/tutorials/start-the-core-engine/)

    a) install the Core engine Backend with Docker Compose

    ```bash
    # Change directory to the Core engine Backend
    cd core-engine/backend

    # Build the Docker image
    docker compose build

    # Start the Core engine Backend
    docker compose up
    ```

    b) install the Core engine Frontend with Docker Compose

    ```bash
    # Change directory to the Core engine Frontend
    cd ..
    cd core-engine/frontend

    # Build the Docker image
    docker compose build

    # Start the Core engine Frontend
    docker compose up
    ```

    Access the Core engine Frontend on <http://localhost:3000>

2) Install this service

    ```bash
    # Clone this service repository
    git clone https://github.com/denisiglesiasgarcia/yolov8-service

    # Build the Docker image
    docker compose build

    # Start the service
    docker compose up
    ```
    





