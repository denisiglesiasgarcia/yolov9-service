import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import torch
import platform
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
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
from contextlib import asynccontextmanager

# Imports required by the service's model
import io
import os
import numpy as np
import json
from PIL import Image
from ultralytics import YOLO
import cv2 as cv
from common_code.tasks.service import get_extension

settings = get_settings()

def setup_device():
    """
    Setup the computation device (GPU/CPU) based on availability
    """
    if platform.system() == 'Darwin':  # Check if running on macOS
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
            print("Using MPS (Metal Performance Shaders) device")
            return device
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA device")
        return device
    
    print("Using CPU device")
    return 'cpu'

def to_device(tensor, device):
    """
    Safely move a tensor to the specified device
    """
    if device == 'mps':
        # Ensure the tensor is in the correct format for MPS
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
    return tensor.to(device)

class MyService(Service):
    """
    Yolov9 model with GPU support
    """

    _logger: Logger
    _model_seg: object
    _device: str

    def __init__(self):
        super().__init__(
            name="Yolov9-segment",
            slug="yolov9-segment",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.IMAGE_JPEG]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_RECOGNITION,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/yolov8/",
        )
        self._logger = get_logger(settings)
        
        # Setup device
        self._device = setup_device()
        self._logger.info(f"Using device: {self._device}")
        
        # Load model
        model_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "model/best.pt"
        )
        self._model_seg = YOLO(model_path)
        
        # Move model to appropriate device
        if self._device != 'cpu':
            self._model_seg.to(self._device)

    def process(self, data):
        raw = data["image"].data
        input_type = data["image"].type
        buff = io.BytesIO(raw)
        # open buff with cv 
        img_pred = cv.imdecode(np.frombuffer(buff.read(), np.uint8), cv.IMREAD_COLOR)

        colors_pred = [
            (255, 255, 0),  # Pred Rooftop
            (0.0, 165, 255),  # Pred Solar panel
        ]

        # Run inference on the source
        try:
            if self._device != 'cpu':
                # Convert image to tensor and move to device
                img_tensor = torch.from_numpy(img_pred.transpose(2, 0, 1)).float()
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                img_tensor = to_device(img_tensor, self._device)
                
                # Run inference
                results = self._model_seg(img_tensor)
                
                # Move results back to CPU for processing
                if self._device == 'mps':
                    results = [r.cpu() for r in results]
            else:
                results = self._model_seg(img_pred)
            
        except Exception as e:
            self._logger.error(f"Error during inference: {str(e)}")
            # Fallback to CPU if GPU inference fails
            self._logger.info("Falling back to CPU inference")
            results = self._model_seg(img_pred)
 
        for r in results:
            masks = r.masks.xy if r.masks is not None else []  # list of masks
            mask_class = r.boxes.cls.cpu().numpy()  # class of the mask
            confidence_score = r.boxes.conf.cpu().numpy()  # confidence of the mask

            for mask, c, conf_score in zip(masks, mask_class, confidence_score):
                mask = np.array(mask, np.int32)
                c = int(c)  # Convert c to an integer
                overlay = img_pred.copy()
                cv.fillPoly(overlay, [mask], color=colors_pred[c] + (128,))
                img_pred = cv.addWeighted(overlay, 0.5, img_pred, 0.5, 0)

                # Calculate the centroid of the mask
                moments = cv.moments(mask)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])

                    # Add label text at the centroid
                    conf_score = round(conf_score, 2)
                    if c==0:
                        label_text = "rooftop (" + str(conf_score) + ")"
                        label_color = colors_pred[0]
                    elif c==1:
                        label_text = "solar-panel (" + str(conf_score) + ")"
                        label_color = colors_pred[1]
                    label_font_scale = 1.0
                    label_thickness = 2
                    label_bg_color = (0, 0, 0)
                    label_text_size, _ = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
                    label_x = cx - label_text_size[0] // 2
                    label_y = cy + label_text_size[1] // 2

                    cv.rectangle(img_pred, (label_x - 5, label_y - label_text_size[1] - 5),
                                (label_x + label_text_size[0] + 5, label_y + 5), label_bg_color, cv.FILLED)

                    cv.putText(img_pred, label_text, (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX,
                            label_font_scale, label_color, label_thickness)

        # Add legend for predicted image
        pred_legend_font_scale = 1.0
        pred_legend_thickness = 2
        pred_legend_color = (255, 255, 255)
        pred_legend_bg_color = (50, 50, 5)
        pred_legend_text = "Predicted"
        pred_legend_text_size, _ = cv.getTextSize(pred_legend_text, cv.FONT_HERSHEY_SIMPLEX, pred_legend_font_scale, pred_legend_thickness)
        pred_legend_x = img_pred.shape[1] - pred_legend_text_size[0] - 20
        pred_legend_y = 30
        cv.rectangle(img_pred, (pred_legend_x - 10, pred_legend_y - pred_legend_text_size[1] - 10),
                    (pred_legend_x + pred_legend_text_size[0] + 10, pred_legend_y + 10), pred_legend_bg_color, -1)
        cv.putText(img_pred, pred_legend_text, (pred_legend_x, pred_legend_y), cv.FONT_HERSHEY_SIMPLEX,
                    pred_legend_font_scale, pred_legend_color, pred_legend_thickness)

        color_rooftop_pred = (255, 255, 0)  # Blue (BGR format)
        color_solar_panel_pred = (0, 165, 255)  # Orange (BGR format)
        pred_legend_items = [
            ("Rooftop", color_rooftop_pred),
            ("Solar-panel", color_solar_panel_pred)
        ]

        for i, (text, color) in enumerate(pred_legend_items):
            text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, pred_legend_font_scale, pred_legend_thickness)
            text_x = img_pred.shape[1] - text_size[0] - 20
            text_y = pred_legend_y + (i + 1) * (text_size[1] + 20)
            cv.rectangle(img_pred, (text_x - 10, text_y - text_size[1] - 10),
                        (text_x + text_size[0] + 10, text_y + 10), pred_legend_bg_color, -1)
            cv.putText(img_pred, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, pred_legend_font_scale, color, pred_legend_thickness)

        # Encode the image
        guessed_extension = get_extension(input_type)
        is_success, out_buff = cv.imencode(guessed_extension, img_pred)

        return {
            "result": TaskData(
                data=out_buff.tobytes(),
                type=input_type,
            )
        }

service_service: ServiceService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    global service_service

    # Startup
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

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)

api_description = """
This service will use Yolov9 to analyse orthophotos and apply semantic segmentation on rooftops and solar panels.
"""

api_summary = """
Yolov9 trained model to segment.
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="Yolov9 API.",
    description=api_description,
    version="1.0.0",
    contact={
        "name": "Swiss AI Center",
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