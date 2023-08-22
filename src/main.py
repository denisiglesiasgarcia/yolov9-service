import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from pydantic import Field
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import (
    FieldDescriptionType,
    ExecutionUnitTagName,
    ExecutionUnitTagAcronym,
)
from common_code.common.models import FieldDescription, ExecutionUnitTag

# Imports required by the service's model
import io
import numpy as np
import json
from PIL import Image
from ultralytics import YOLO

settings = get_settings()


class Results:
    def __init__(self, results_temp):
        self.top5 = results_temp[0].probs.top5
        self.top5conf = results_temp[0].probs.top5conf.numpy()
        self.results = {
            "top5": {
                results_temp[0].names[top]: conf
                for top, conf in zip(self.top5, self.top5conf)
            }
        }

    def tojson(self):
        return json.dumps(str(self.results))


class MyService(Service):
    """
    Yolov8 service model
    """

    # Any additional fields must be excluded for Pydantic to work
    model_detect: object = Field(exclude=True)
    model_seg: object = Field(exclude=True)
    model_pose: object = Field(exclude=True)
    model_class: object = Field(exclude=True)

    def __init__(self):
        super().__init__(
            name="Yolov8 Service",
            slug="yolov8-service",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_PNG,
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
                FieldDescription(name="type", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_JSON]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING,
                ),
            ],
        )

        self.model_detect = YOLO("model/yolov8x.pt")
        self.model_seg = YOLO("model/yolov8x-seg.pt")
        self.model_pose = YOLO("model/yolov8x-pose.pt")
        self.model_class = YOLO("model/yolov8x-cls.pt")

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        raw = data["image"].data
        input_type = data["image"].type
        process_type = data["type"].data.decode("utf-8")
        if process_type not in ["detect", "segment", "pose", "classify"]:
            raise Exception("Model type not supported.")
        # ... do something with the raw data
        buff = io.BytesIO(raw)
        img_pil = Image.open(buff)
        img = np.array(img_pil)

        if process_type == "detect":
            results = self.model_detect(img)[0]
        elif process_type == "segment":
            results = self.model_seg(img)[0]
        elif process_type == "pose":
            results = self.model_pose(img)[0]
        elif process_type == "classify":
            results = Results(self.model_class(img))

        task_data = TaskData(
            data=results.tojson(), type=FieldDescriptionType.APPLICATION_JSON
        )

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {"result": task_data}


api_description = """
This service will use Yolov8 to analyse the image content according to the selected model type :
- detect: object detection
- segment: object segmentation
- pose: human pose estimation
- classify: image classification
"""
api_summary = """
Yolov8 service
"""

# Define the FastAPI application with information
app = FastAPI(
    title="Yolov8 API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "CSIA-PME",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)


service_service: ServiceService | None = None


@app.on_event("startup")
async def startup_event():
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(
                    my_service, engine_url
                )
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())


@app.on_event("shutdown")
async def shutdown_event():
    # Global variable
    global service_service
    my_service = MyService()
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)